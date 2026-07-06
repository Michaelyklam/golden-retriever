from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dataset import RetrievalTask, load_jsonl
from .model_eval import SYSTEM_PROMPT, render_corpus_prompt, render_task_candidate_prompt


def _compact_reason(text: str, max_chars: int = 180) -> str:
    reason = " ".join(text.split())
    if len(reason) <= max_chars:
        return reason
    return reason[: max_chars - 1].rstrip() + "…"


def render_gold_thinking(task: RetrievalTask) -> str:
    positive_ids = ", ".join(doc.doc_id for doc in task.supporting_documents)
    distractor_ids = ", ".join(doc.doc_id for doc in task.distractor_documents[:3]) or "none"
    primary_quote = task.supporting_documents[0].document_quotes[0] if task.supporting_documents[0].document_quotes else task.answer
    return "\n".join(
        [
            "<think>",
            f"The query asks for evidence matching: {_compact_reason(primary_quote, 120)}",
            f"The grounded supporting document id is {positive_ids}.",
            f"Distractors considered but rejected: {distractor_ids}.",
            "Return only the supporting document tag, preserving the exact id.",
            "</think>",
        ]
    ) + "\n\n"


def render_gold_completion(task: RetrievalTask, include_thinking: bool = False) -> str:
    """Render the supervised target for exact document-id selection."""

    tags: list[str] = []
    for doc in task.supporting_documents:
        quote = doc.document_quotes[0] if doc.document_quotes else task.answer
        tags.append(
            f'<Document id="{doc.doc_id}"><Justification>'
            f'Contains grounded quote: {_compact_reason(quote)}'
            f'</Justification></Document>'
        )
    completion = "\n".join(tags)
    if include_thinking:
        return render_gold_thinking(task) + completion
    return completion


def _render_prompt(task: RetrievalTask, corpus_root: str | Path, prompt_scope: str, max_chars_per_doc: int = 6000) -> str:
    if prompt_scope == "full-corpus":
        return render_corpus_prompt(task, corpus_root, max_chars_per_doc=max_chars_per_doc)
    if prompt_scope == "task-candidates":
        return render_task_candidate_prompt(task, corpus_root, max_chars_per_doc=max_chars_per_doc)
    raise ValueError(f"Unsupported prompt scope: {prompt_scope}")


def build_sft_examples(
    tasks: list[RetrievalTask],
    corpus_root: str | Path,
    include_thinking: bool = False,
    prompt_scope: str = "full-corpus",
    max_chars_per_doc: int = 6000,
) -> list[dict[str, Any]]:
    """Build chat-format SFT examples from retrieval tasks.

    The user prompt is intentionally rendered with the same helper as the eval
    harness so the first fine-tune lap directly targets the measured base-model
    failure: exact final-tag emission for closed-corpus selection.
    """

    examples: list[dict[str, Any]] = []
    for task in tasks:
        examples.append(
            {
                "task_id": task.task_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _render_prompt(task, corpus_root, prompt_scope, max_chars_per_doc=max_chars_per_doc)},
                    {"role": "assistant", "content": render_gold_completion(task, include_thinking=include_thinking)},
                ],
            }
        )
    return examples


def write_sft_jsonl(examples: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chat-format SFT data from retrieval tasks.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-thinking", action="store_true")
    parser.add_argument("--prompt-scope", choices=["full-corpus", "task-candidates"], default="full-corpus")
    parser.add_argument("--max-chars-per-doc", type=int, default=6000)
    args = parser.parse_args()

    tasks = load_jsonl(args.dataset)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    examples = build_sft_examples(
        tasks,
        args.corpus_root,
        include_thinking=args.include_thinking,
        prompt_scope=args.prompt_scope,
        max_chars_per_doc=args.max_chars_per_doc,
    )
    write_sft_jsonl(examples, args.output)
    print(json.dumps({"examples": len(examples), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
