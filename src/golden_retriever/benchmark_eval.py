from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, Field

from golden_retriever.dataset import BenchmarkTask
from golden_retriever.eval import f1 as f1_score
from golden_retriever.eval import final_answer_found as contains_final_answer
from golden_retriever.eval import precision as precision_score
from golden_retriever.eval import recall as recall_score
from golden_retriever.eval import trajectory_recall as trajectory_recall_score


class BenchmarkScore(BaseModel):
    """Comparable retrieval metrics for a Context-1-style benchmark task."""

    task_id: str
    suite: str
    recall: float
    precision: float
    f1: float
    trajectory_recall: float
    final_answer_found: bool
    returned_doc_ids: list[str] = Field(default_factory=list)
    returned_urls: list[str] = Field(default_factory=list)
    unknown_doc_ids: list[str] = Field(default_factory=list)


def canonicalize_url(url: str) -> str:
    """Normalize benchmark URLs for positive-URL scoring.

    Public suites like Seal-0 and FRAMES score URLs as evidence handles. The
    model/harness may emit different host casing, query parameter order, or
    fragments, so normalize those away while preserving meaningful paths.
    """

    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def score_benchmark_output(
    task: BenchmarkTask,
    *,
    returned_doc_ids: Iterable[str] = (),
    returned_urls: Iterable[str] = (),
    output_texts: Iterable[str] = (),
    encountered_doc_ids: Iterable[str] = (),
    valid_doc_ids: Iterable[str] | None = None,
) -> BenchmarkScore:
    """Score one benchmark task using Context-1-style retrieval metrics."""

    returned_docs = _unique(returned_doc_ids)
    returned_url_values = _unique(canonicalize_url(url) for url in returned_urls)
    positive_docs = _unique(task.positive_doc_ids)
    positive_urls = _unique(canonicalize_url(url) for url in task.positive_urls)

    returned_targets = [*returned_docs, *returned_url_values]
    positive_targets = [*positive_docs, *positive_urls]

    valid_doc_set = set(valid_doc_ids) if valid_doc_ids is not None else None
    unknown_doc_ids = [doc_id for doc_id in returned_docs if valid_doc_set is not None and doc_id not in valid_doc_set]

    # The final output is also part of the trajectory: if a model returns a
    # positive doc without it being listed in intermediate tool encounters, it
    # still found that evidence during the episode from the scorer's view.
    encountered = _unique([*encountered_doc_ids, *returned_docs])
    trajectory_recall = trajectory_recall_score(encountered, positive_docs) if positive_docs else 0.0

    answer_found = contains_final_answer(output_texts, task.answer) if task.answer else False

    return BenchmarkScore(
        task_id=task.task_id,
        suite=task.suite,
        recall=recall_score(returned_targets, positive_targets),
        precision=precision_score(returned_targets, positive_targets),
        f1=f1_score(returned_targets, positive_targets),
        trajectory_recall=trajectory_recall,
        final_answer_found=answer_found,
        returned_doc_ids=returned_docs,
        returned_urls=returned_url_values,
        unknown_doc_ids=unknown_doc_ids,
    )
