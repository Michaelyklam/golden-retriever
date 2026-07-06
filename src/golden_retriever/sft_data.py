from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dataset import RetrievalTask, load_jsonl
from .model_eval import SYSTEM_PROMPT, render_corpus_prompt


def _compact_reason(text: str, max_chars: int = 180) -> str:
    reason = " ".join(text.split())
    if len(reason) <= max_chars:
        return reason
    return reason[: max_chars - 1].rstrip() + "…"


def render_gold_completion(task: RetrievalTask) -> str:
    """Render the supervised target for exact document-id selection."""

    tags: list[str] = []
    for doc in task.supporting_documents:
        quote = doc.document_quotes[0] if doc.document_quotes else task.answer
        tags.append(
            f'<Document id="{doc.doc_id}"><Justification>'
            f'Contains grounded quote: {_compact_reason(quote)}'
            f'</Justification></Document>'
        )
    return "\n".join(tags)


def build_sft_examples(tasks: list[RetrievalTask], corpus_root: str | Path) -> list[dict[str, Any]]:
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
                    {"role": "user", "content": render_corpus_prompt(task, corpus_root)},
                    {"role": "assistant", "content": render_gold_completion(task)},
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
    args = parser.parse_args()

    tasks = load_jsonl(args.dataset)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    examples = build_sft_examples(tasks, args.corpus_root)
    write_sft_jsonl(examples, args.output)
    print(json.dumps({"examples": len(examples), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
