from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from golden_retriever.dataset import RetrievalTask, load_jsonl
from golden_retriever.model_eval import SYSTEM_PROMPT
from golden_retriever.sft_data import render_gold_completion, write_sft_jsonl


def _load_suite_tasks(benchmark_root: Path, suite: str, limit: int | None = None) -> tuple[list[RetrievalTask], Path]:
    suite_root = benchmark_root / suite
    dataset_path = suite_root / "tasks.jsonl"
    corpus_root = suite_root / "corpus"
    if not dataset_path.exists():
        raise FileNotFoundError(f"missing benchmark task file: {dataset_path}")
    if not corpus_root.exists():
        raise FileNotFoundError(f"missing benchmark corpus root: {corpus_root}")
    tasks = load_jsonl(dataset_path)
    if limit is not None:
        tasks = tasks[:limit]
    return tasks, corpus_root


def _read_doc(corpus_root: Path, doc_id: str, max_chars: int) -> str:
    path = corpus_root / doc_id
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]


def _render_candidate_prompt_fast(task: RetrievalTask, corpus_root: Path, max_chars_per_doc: int) -> str:
    ordered_ids: list[str] = []
    for doc in [*task.supporting_documents, *task.distractor_documents]:
        if doc.doc_id not in ordered_ids:
            ordered_ids.append(doc.doc_id)
    rendered_docs = []
    for doc_id in ordered_ids:
        text = _read_doc(corpus_root, doc_id, max_chars_per_doc)
        if text:
            rendered_docs.append(f'<Document id="{doc_id}">\n{text}\n</Document>')
    return "\n\n".join(
        [
            f"Query: {task.question}",
            "Relevant clues:",
            *[f"- {clue}" for clue in task.clues],
            "",
            "Candidate documents:",
            *rendered_docs,
            "",
            "Return only the supporting document ids in the requested XML-like format. Do not return distractors.",
        ]
    )


def _build_examples_fast(
    tasks: list[RetrievalTask],
    corpus_root: Path,
    include_thinking: bool,
    max_chars_per_doc: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for task in tasks:
        examples.append(
            {
                "task_id": task.task_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _render_candidate_prompt_fast(task, corpus_root, max_chars_per_doc)},
                    {"role": "assistant", "content": render_gold_completion(task, include_thinking=include_thinking)},
                ],
            }
        )
    return examples


def build_bootstrap_corpus(
    *,
    benchmark_root: str | Path = "data/benchmarks/materialized",
    output_root: str | Path = "data/sft/context1-bootstrap-v1",
    suites: list[str] | None = None,
    limit_per_suite: int | None = None,
    train_fraction: float = 0.9,
    max_chars_per_doc: int = 1200,
    include_thinking: bool = True,
    seed: int = 20260706,
) -> dict[str, Any]:
    """Build first keyless Context-1-style bootstrap SFT data.

    This uses public benchmark corpora already materialized into the project
    schema. It is label-supervised rather than Codex-teacher-supervised: a fast,
    reproducible first stage that gets the training pipeline moving without
    Anthropic/OpenAI API keys. Codex-backed trajectory rollouts can later replace
    or augment these silver examples.
    """

    benchmark_root = Path(benchmark_root)
    output_root = Path(output_root)
    selected_suites = suites or ["longseal", "hotpotqa"]
    rng = random.Random(seed)
    all_examples: list[dict[str, Any]] = []
    suite_reports: list[dict[str, Any]] = []
    for suite in selected_suites:
        tasks, corpus_root = _load_suite_tasks(benchmark_root, suite, limit=limit_per_suite)
        examples = _build_examples_fast(
            tasks,
            corpus_root,
            include_thinking=include_thinking,
            max_chars_per_doc=max_chars_per_doc,
        )
        for example in examples:
            example.setdefault("metadata", {})
            example["metadata"].update(
                {
                    "source_suite": suite,
                    "teacher": "gold_label_bootstrap",
                    "prompt_scope": "task-candidates",
                    "max_chars_per_doc": max_chars_per_doc,
                }
            )
        all_examples.extend(examples)
        suite_reports.append({"suite": suite, "tasks": len(tasks), "examples": len(examples), "corpus_root": str(corpus_root)})

    rng.shuffle(all_examples)
    split_at = len(all_examples) if train_fraction >= 1 else max(1, int(len(all_examples) * train_fraction))
    train_examples = all_examples[:split_at]
    eval_examples = all_examples[split_at:]
    train_path = output_root / "train.jsonl"
    eval_path = output_root / "eval.jsonl"
    manifest_path = output_root / "manifest.json"
    write_sft_jsonl(train_examples, train_path)
    write_sft_jsonl(eval_examples, eval_path)
    report = {
        "output_root": str(output_root),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "manifest_path": str(manifest_path),
        "suites": suite_reports,
        "examples_total": len(all_examples),
        "examples_train": len(train_examples),
        "examples_eval": len(eval_examples),
        "include_thinking": include_thinking,
        "max_chars_per_doc": max_chars_per_doc,
        "teacher": "gold_label_bootstrap",
        "note": "Keyless first-stage corpus. No Anthropic/OpenAI API keys. Codex CLI teacher trajectories are the next augmentation layer.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a keyless Context-1 bootstrap SFT corpus from materialized public benchmarks.")
    parser.add_argument("--benchmark-root", default="data/benchmarks/materialized")
    parser.add_argument("--output-root", default="data/sft/context1-bootstrap-v1")
    parser.add_argument("--suite", action="append", dest="suites", help="Suite to include. Repeatable. Defaults to longseal + hotpotqa.")
    parser.add_argument("--limit-per-suite", type=int)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--max-chars-per-doc", type=int, default=1200)
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args()
    report = build_bootstrap_corpus(
        benchmark_root=args.benchmark_root,
        output_root=args.output_root,
        suites=args.suites,
        limit_per_suite=args.limit_per_suite,
        train_fraction=args.train_fraction,
        max_chars_per_doc=args.max_chars_per_doc,
        include_thinking=not args.no_thinking,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
