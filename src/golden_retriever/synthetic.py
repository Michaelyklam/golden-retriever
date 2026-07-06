from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .dataset import RetrievalTask

SUPPORTED_SUFFIXES = {".md", ".txt", ".rst"}
TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class FactCandidate:
    doc_id: str
    quote: str


def _tokens(text: str) -> set[str]:
    return {t.casefold() for t in TOKEN_RE.findall(text) if len(t) > 3}


def _iter_text_files(corpus_root: str | Path) -> list[Path]:
    root = Path(corpus_root)
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]


def _sentences_from_text(text: str) -> list[str]:
    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("|") or line.lower().startswith("source:"):
            continue
        sentences.extend(s.strip() for s in SENTENCE_RE.split(line) if s.strip())
    return sentences


def extract_fact_candidates(corpus_root: str | Path, min_chars: int = 80, max_chars: int = 260) -> list[FactCandidate]:
    """Extract deterministic source-quote candidates from a local corpus."""

    root = Path(corpus_root)
    candidates: list[FactCandidate] = []
    for path in _iter_text_files(root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        doc_id = path.relative_to(root).as_posix()
        for sentence in _sentences_from_text(text):
            if min_chars <= len(sentence) <= max_chars:
                candidates.append(FactCandidate(doc_id=doc_id, quote=sentence))
                break
    return candidates


def select_distractors(target: FactCandidate, candidates: list[FactCandidate], count: int = 3) -> list[FactCandidate]:
    """Pick hard distractors by token overlap with the target quote."""

    target_tokens = _tokens(target.quote)
    scored: list[tuple[int, str, FactCandidate]] = []
    for candidate in candidates:
        if candidate.doc_id == target.doc_id:
            continue
        overlap = len(target_tokens & _tokens(candidate.quote))
        scored.append((overlap, candidate.doc_id, candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, _, candidate in scored[:count]]


def _answer_from_quote(quote: str) -> str:
    # The answer only needs to appear in the positive document for retrieval
    # evaluation. Use a concise quote prefix for deterministic smoke data.
    words = quote.split()
    return " ".join(words[: min(8, len(words))]).strip(" ,.;:")


def _clue_from_quote(quote: str) -> str:
    return f"Find evidence for this claim: {quote}"


def _task_from_candidate(
    candidate: FactCandidate,
    all_candidates: list[FactCandidate],
    index: int,
    domain: str,
    distractor_count: int,
) -> RetrievalTask:
    distractors = select_distractors(candidate, all_candidates, count=distractor_count)
    answer = _answer_from_quote(candidate.quote)
    clue = _clue_from_quote(candidate.quote)
    return RetrievalTask(
        task_id=f"{domain}-{index:06d}",
        domain=domain,
        difficulty=1 + min(len(distractors), 3),
        hop_count=1,
        question=f"Which document contains evidence for the claim: {candidate.quote}",
        answer=answer,
        clues=[clue],
        supporting_documents=[
            {
                "doc_id": candidate.doc_id,
                "role": "positive",
                "document_quotes": [candidate.quote],
                "clue_quotes": [clue],
            }
        ],
        distractor_documents=[
            {
                "doc_id": distractor.doc_id,
                "role": "distractor",
                "document_quotes": [distractor.quote],
                "clue_quotes": ["hard distractor with overlapping retrieval terms"],
            }
            for distractor in distractors
        ],
        metadata={"source": "deterministic_local_generator_v0"},
    )


def generate_tasks(
    corpus_root: str | Path,
    limit: int | None = None,
    distractor_count: int = 3,
    domain: str = "localdocs",
) -> list[RetrievalTask]:
    candidates = extract_fact_candidates(corpus_root)
    if limit is not None:
        candidates = candidates[:limit]
    return [
        _task_from_candidate(candidate, candidates, index + 1, domain=domain, distractor_count=distractor_count)
        for index, candidate in enumerate(candidates)
    ]


def write_tasks_jsonl(tasks: list[RetrievalTask], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task.model_dump(), ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic local retrieval tasks.")
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--distractors", type=int, default=3)
    parser.add_argument("--domain", default="localdocs")
    args = parser.parse_args()

    tasks = generate_tasks(
        corpus_root=args.corpus_root,
        limit=args.limit,
        distractor_count=args.distractors,
        domain=args.domain,
    )
    write_tasks_jsonl(tasks, args.output)
    print(json.dumps({"tasks": len(tasks), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
