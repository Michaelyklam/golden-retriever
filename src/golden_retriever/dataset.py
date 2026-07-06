from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator


class EvidenceDocument(BaseModel):
    """A positive or distractor document attached to a retrieval task."""

    doc_id: str
    role: Literal["positive", "distractor"]
    document_quotes: list[str] = Field(default_factory=list)
    clue_quotes: list[str] = Field(default_factory=list)


class RetrievalTask(BaseModel):
    """One Context-1-style retrieval task."""

    task_id: str
    domain: str
    difficulty: int = Field(ge=0)
    hop_count: int = Field(ge=1)
    question: str
    answer: str
    clues: list[str] = Field(min_length=1)
    supporting_documents: list[EvidenceDocument] = Field(min_length=1)
    distractor_documents: list[EvidenceDocument] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def roles_match_containers(self) -> "RetrievalTask":
        for doc in self.supporting_documents:
            if doc.role != "positive":
                raise ValueError(f"supporting document {doc.doc_id} must have role='positive'")
        for doc in self.distractor_documents:
            if doc.role != "distractor":
                raise ValueError(f"distractor document {doc.doc_id} must have role='distractor'")
        return self


BenchmarkSuite = Literal[
    "generated",
    "browsecomp_plus",
    "seal0",
    "longseal",
    "frames",
    "hotpotqa",
    "hle",
]


class BenchmarkDocument(BaseModel):
    """A document or chunk exposed to a full-scale benchmark harness."""

    doc_id: str
    text: str
    url: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkTask(BaseModel):
    """One task from a comparable Context-1-style benchmark suite."""

    task_id: str
    suite: BenchmarkSuite
    question: str
    answer: str | None = None
    positive_doc_ids: list[str] = Field(default_factory=list)
    positive_urls: list[str] = Field(default_factory=list)
    corpus_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def scored_suites_have_positive_targets(self) -> "BenchmarkTask":
        if self.suite != "hle" and not self.positive_doc_ids and not self.positive_urls:
            raise ValueError("BenchmarkTask requires positive_doc_ids or positive_urls for scored retrieval suites")
        return self


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def load_jsonl(path: str | Path) -> list[RetrievalTask]:
    tasks: list[RetrievalTask] = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                tasks.append(RetrievalTask.model_validate_json(line))
            except ValidationError as exc:
                raise ValueError(f"Invalid task at {path}:{line_no}: {exc}") from exc
    return tasks


def validate_quote_grounding(task: RetrievalTask, corpus_root: str | Path) -> list[str]:
    """Check that declared document quotes literally appear in corpus files.

    This is the deterministic half of the Context-1 verification approach. The
    LLM extraction step should produce document_quotes; this function refuses
    tasks whose quotes cannot be grounded in source text.
    """

    root = Path(corpus_root)
    errors: list[str] = []
    for doc in [*task.supporting_documents, *task.distractor_documents]:
        path = root / doc.doc_id
        if not path.exists():
            errors.append(f"{task.task_id}: missing document {doc.doc_id}")
            continue
        text = normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
        for quote in doc.document_quotes:
            if normalize_text(quote) not in text:
                errors.append(f"{task.task_id}: quote not found in {doc.doc_id}: {quote!r}")
    return errors


def validate_dataset(path: str | Path, corpus_root: str | Path | None = None) -> list[str]:
    errors: list[str] = []
    try:
        tasks = load_jsonl(path)
    except ValueError as exc:
        return [str(exc)]
    if corpus_root is not None:
        for task in tasks:
            errors.extend(validate_quote_grounding(task, corpus_root))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate golden-retriever JSONL retrieval tasks.")
    parser.add_argument("path", help="JSONL dataset path")
    parser.add_argument("--corpus-root", help="Optional corpus root for quote-grounding checks")
    args = parser.parse_args()

    errors = validate_dataset(args.path, args.corpus_root)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        raise SystemExit(1)
    print(json.dumps({"ok": True}, indent=2))


if __name__ == "__main__":
    main()
