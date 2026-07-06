from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from golden_retriever.benchmark_eval import canonicalize_url
from golden_retriever.dataset import BenchmarkTask, EvidenceDocument, RetrievalTask


def slugify(text: str, fallback: str = "doc") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.casefold()).strip("-")
    return slug[:80] or fallback


def _doc_text(title: str | None, text: str | None, url: str | None = None) -> str:
    parts = []
    if title:
        parts.append(f"# {title}")
    if url:
        parts.append(f"Source URL: {url}")
    if text:
        parts.append(text)
    return "\n\n".join(parts).strip() + "\n"


def hotpot_row_to_task(row: dict[str, Any]) -> tuple[RetrievalTask, dict[str, str]]:
    task_id = f"hotpotqa-{row['id']}"
    titles = row["context"]["title"]
    sentences = row["context"]["sentences"]
    positive_titles = list(dict.fromkeys(row["supporting_facts"]["title"]))
    title_to_doc_id: dict[str, str] = {}
    docs: dict[str, str] = {}
    for title, sents in zip(titles, sentences, strict=False):
        doc_id = f"hotpotqa/{row['id']}/{slugify(title)}.md"
        title_to_doc_id[title] = doc_id
        docs[doc_id] = _doc_text(title, " ".join(sents))
    positive_doc_ids = {title_to_doc_id[title] for title in positive_titles if title in title_to_doc_id}
    supporting = [
        EvidenceDocument(doc_id=doc_id, role="positive", document_quotes=[row["answer"]], clue_quotes=[])
        for doc_id in title_to_doc_id.values()
        if doc_id in positive_doc_ids
    ]
    distractors = [
        EvidenceDocument(doc_id=doc_id, role="distractor", document_quotes=[], clue_quotes=[])
        for doc_id in title_to_doc_id.values()
        if doc_id not in positive_doc_ids
    ]
    task = RetrievalTask(
        task_id=task_id,
        domain="hotpotqa",
        difficulty=1 if row.get("level") == "easy" else 2,
        hop_count=max(1, len(positive_doc_ids)),
        question=row["question"],
        answer=row["answer"],
        clues=[row["question"]],
        supporting_documents=supporting,
        distractor_documents=distractors,
        metadata={"source_suite": "hotpotqa", "type": row.get("type"), "level": row.get("level")},
    )
    return task, docs


def _stable_doc_id(prefix: str, idx: int, doc: dict[str, Any]) -> str:
    title = slugify(str(doc.get("title") or "doc"))
    url = str(doc.get("url") or "")
    digest = hashlib.sha1((url + title + str(idx)).encode()).hexdigest()[:8]
    return f"{prefix}/{idx:06d}/{title}-{digest}.md"


def longseal_row_to_task(row: dict[str, Any], idx: int, doc_field: str = "30_docs") -> tuple[RetrievalTask, dict[str, str]]:
    docs_in = row.get(doc_field) or row.get("20_docs") or row.get("12_docs") or []
    gold_urls = {canonicalize_url(str(url)) for url in row.get("urls") or [] if url}
    gold_urls.update(canonicalize_url(str(doc.get("url"))) for doc in row.get("golds") or [] if doc.get("url"))
    docs: dict[str, str] = {}
    supporting: list[EvidenceDocument] = []
    distractors: list[EvidenceDocument] = []
    for doc_pos, doc in enumerate(docs_in):
        doc_id = _stable_doc_id("longseal", idx, doc | {"pos": doc_pos})
        docs[doc_id] = _doc_text(doc.get("title"), doc.get("text"), doc.get("url"))
        role = "positive" if doc.get("url") and canonicalize_url(str(doc.get("url"))) in gold_urls else "distractor"
        ev = EvidenceDocument(
            doc_id=doc_id,
            role=role,
            document_quotes=[row.get("answer", "")] if role == "positive" else [],
            clue_quotes=[],
        )
        (supporting if role == "positive" else distractors).append(ev)
    # Some LongSeal rows have gold docs not present in the chosen retrieved set.
    # Add them so the task is still scoreable while preserving the retrieved-set docs.
    if not supporting:
        for gold_pos, doc in enumerate(row.get("golds") or []):
            doc_id = _stable_doc_id("longseal", idx * 1000 + gold_pos, doc)
            docs[doc_id] = _doc_text(doc.get("title"), doc.get("text"), doc.get("url"))
            supporting.append(EvidenceDocument(doc_id=doc_id, role="positive", document_quotes=[row.get("answer", "")], clue_quotes=[]))
    task = RetrievalTask(
        task_id=f"longseal-{idx:06d}",
        domain="longseal",
        difficulty=3,
        hop_count=max(1, len(supporting)),
        question=row["question"],
        answer=row["answer"],
        clues=[row["question"]],
        supporting_documents=supporting,
        distractor_documents=distractors,
        metadata={"source_suite": "longseal", "topic": row.get("topic"), "doc_field": doc_field},
    )
    return task, docs


def parse_frames_row(row: dict[str, str], idx: int) -> BenchmarkTask:
    urls: list[str] = []
    for key, value in row.items():
        if key.startswith("wikipedia_link_") and value and value.strip():
            urls.append(value.strip())
    raw_links = row.get("wiki_links", "")
    if raw_links:
        try:
            parsed = ast.literal_eval(raw_links)
            if isinstance(parsed, list):
                urls.extend(str(url) for url in parsed if url)
        except Exception:
            pass
    deduped = list(dict.fromkeys(urls))
    return BenchmarkTask(
        task_id=f"frames-{idx:06d}",
        suite="frames",
        question=row["Prompt"],
        answer=row.get("Answer") or None,
        positive_urls=deduped,
        metadata={"reasoning_types": row.get("reasoning_types")},
    )


def _write_retrieval_dataset(tasks: list[RetrievalTask], docs: dict[str, str], output_root: Path, suite: str) -> dict[str, Any]:
    suite_root = output_root / suite
    corpus_root = suite_root / "corpus"
    dataset_path = suite_root / "tasks.jsonl"
    corpus_root.mkdir(parents=True, exist_ok=True)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    for doc_id, text in docs.items():
        path = corpus_root / doc_id
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    with dataset_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(task.model_dump_json() + "\n")
    return {"suite": suite, "tasks": len(tasks), "docs": len(docs), "dataset": str(dataset_path), "corpus_root": str(corpus_root)}


def prepare_hotpotqa(output_root: Path, limit: int | None = None) -> dict[str, Any]:
    import pyarrow.parquet as pq

    path = hf_hub_download("hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet", repo_type="dataset", local_dir="data/benchmarks/raw/hf")
    rows = pq.read_table(path).to_pylist()
    if limit:
        rows = rows[:limit]
    tasks: list[RetrievalTask] = []
    docs: dict[str, str] = {}
    for row in rows:
        task, row_docs = hotpot_row_to_task(row)
        tasks.append(task)
        docs.update(row_docs)
    return _write_retrieval_dataset(tasks, docs, output_root, "hotpotqa")


def prepare_longseal(output_root: Path, limit: int | None = None) -> dict[str, Any]:
    import pyarrow.parquet as pq

    path = hf_hub_download("vtllms/sealqa", "longseal.parquet", repo_type="dataset", local_dir="data/benchmarks/raw/hf")
    rows = pq.read_table(path).to_pylist()
    if limit:
        rows = rows[:limit]
    tasks: list[RetrievalTask] = []
    docs: dict[str, str] = {}
    for idx, row in enumerate(rows):
        try:
            task, row_docs = longseal_row_to_task(row, idx)
        except ValueError as exc:
            if "supporting_documents" in str(exc):
                continue
            raise
        tasks.append(task)
        docs.update(row_docs)
    return _write_retrieval_dataset(tasks, docs, output_root, "longseal")


def prepare_frames(output_root: Path, limit: int | None = None) -> dict[str, Any]:
    path = hf_hub_download("google/frames-benchmark", "test.tsv", repo_type="dataset", local_dir="data/benchmarks/raw/hf")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if limit:
        rows = rows[:limit]
    tasks = [parse_frames_row(row, idx) for idx, row in enumerate(rows)]
    suite_root = output_root / "frames"
    suite_root.mkdir(parents=True, exist_ok=True)
    out = suite_root / "tasks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(task.model_dump_json() + "\n")
    return {"suite": "frames", "tasks": len(tasks), "dataset": str(out), "note": "URL-positive benchmark; corpus materialization requires Wikipedia fetching."}


def prepare_seal0(output_root: Path, limit: int | None = None) -> dict[str, Any]:
    import pyarrow.parquet as pq

    path = hf_hub_download("vtllms/sealqa", "seal-0.parquet", repo_type="dataset", local_dir="data/benchmarks/raw/hf")
    rows = pq.read_table(path).to_pylist()
    if limit:
        rows = rows[:limit]
    tasks = [
        BenchmarkTask(
            task_id=f"seal0-{idx:06d}",
            suite="seal0",
            question=row["question"],
            answer=row.get("answer"),
            positive_urls=row.get("urls") or [],
            metadata={"topic": row.get("topic"), "freshness": row.get("freshness")},
        )
        for idx, row in enumerate(rows)
    ]
    suite_root = output_root / "seal0"
    suite_root.mkdir(parents=True, exist_ok=True)
    out = suite_root / "tasks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(task.model_dump_json() + "\n")
    return {"suite": "seal0", "tasks": len(tasks), "dataset": str(out), "note": "URL-positive web benchmark; full run requires browsing/scraping backend."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare public Context-1 benchmark datasets for golden-retriever.")
    parser.add_argument("--suite", choices=["hotpotqa", "longseal", "frames", "seal0", "all"], default="all")
    parser.add_argument("--output-root", default="data/benchmarks/materialized")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    root = Path(args.output_root)
    suites = ["hotpotqa", "longseal", "frames", "seal0"] if args.suite == "all" else [args.suite]
    reports = []
    for suite in suites:
        reports.append(globals()[f"prepare_{suite}"](root, args.limit))
    print(json.dumps({"reports": reports}, indent=2))


if __name__ == "__main__":
    main()
