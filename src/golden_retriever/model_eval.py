from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .dataset import RetrievalTask, load_jsonl
from .eval import f1, final_answer_found, precision, recall

DOC_TAG_RE = re.compile(r"<Document\s+id\s*=\s*[\"']?([^\"'\s>]+)[\"']?\s*>", re.IGNORECASE)


SYSTEM_PROMPT = """You are a retrieval subagent. Return the document ids most relevant to the query.
Do not answer the question directly. Return only ranked document tags in this format:
<Document id="document_id"><Justification>Brief reason.</Justification></Document>
"""


def parse_document_ids(text: str) -> list[str]:
    """Extract ranked Context-1-style document ids from model output."""

    seen: set[str] = set()
    ids: list[str] = []
    for match in DOC_TAG_RE.finditer(text):
        doc_id = match.group(1)
        if doc_id not in seen:
            ids.append(doc_id)
            seen.add(doc_id)
    return ids


def _load_corpus_documents(corpus_root: str | Path) -> list[tuple[str, str]]:
    root = Path(corpus_root)
    docs: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".rst"}:
            continue
        rel = path.relative_to(root).as_posix()
        docs.append((rel, path.read_text(encoding="utf-8", errors="ignore")))
    return docs


def render_corpus_prompt(task: RetrievalTask, corpus_root: str | Path, max_chars_per_doc: int = 6000) -> str:
    """Render a small-corpus baseline prompt for base-model evaluation.

    This is intentionally a simple closed-corpus selection baseline, not the
    final observe/reason/act tool loop. It gives us a measured MiniCPM5 base
    checkpoint before generating SFT traces.
    """

    docs = _load_corpus_documents(corpus_root)
    rendered_docs = []
    for doc_id, text in docs:
        text = text[:max_chars_per_doc]
        rendered_docs.append(f'<Document id="{doc_id}">\n{text}\n</Document>')
    return "\n\n".join(
        [
            f"Query: {task.question}",
            "Relevant clues:",
            *[f"- {clue}" for clue in task.clues],
            "",
            "Corpus:",
            *rendered_docs,
            "",
            "Return the most relevant document ids only in the requested XML-like format.",
        ]
    )


def call_openai_compatible(
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: int = 120,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Model request failed: HTTP {exc.code}: {body}") from exc
    return data["choices"][0]["message"]["content"]


def evaluate_task_from_text(task: RetrievalTask, corpus_root: str | Path, output_text: str) -> dict[str, Any]:
    parsed_ids = parse_document_ids(output_text)
    positive_ids = [doc.doc_id for doc in task.supporting_documents]
    corpus_by_id = dict(_load_corpus_documents(corpus_root))
    returned_ids = [doc_id for doc_id in parsed_ids if doc_id in corpus_by_id]
    unknown_ids = [doc_id for doc_id in parsed_ids if doc_id not in corpus_by_id]
    output_texts = [corpus_by_id.get(doc_id, "") for doc_id in returned_ids]
    return {
        "task_id": task.task_id,
        "question": task.question,
        "answer": task.answer,
        "positive_doc_ids": positive_ids,
        "returned_doc_ids": returned_ids,
        "unknown_doc_ids": unknown_ids,
        "final_answer_found": final_answer_found(output_texts, task.answer),
        "recall": recall(returned_ids, positive_ids),
        "precision": precision(returned_ids, positive_ids),
        "f1": f1(returned_ids, positive_ids),
        "raw_output": output_text,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"num_tasks": 0}
    return {
        "num_tasks": len(results),
        "final_answer_found": sum(1 for r in results if r["final_answer_found"]) / len(results),
        "recall": sum(float(r["recall"]) for r in results) / len(results),
        "precision": sum(float(r["precision"]) for r in results) / len(results),
        "f1": sum(float(r["f1"]) for r in results) / len(results),
    }


def evaluate_dataset(
    dataset_path: str | Path,
    corpus_root: str | Path,
    base_url: str,
    model: str,
    limit: int | None = None,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    tasks = load_jsonl(dataset_path)
    if limit is not None:
        tasks = tasks[:limit]
    results: list[dict[str, Any]] = []
    started = time.time()
    for task in tasks:
        prompt = render_corpus_prompt(task, corpus_root)
        output = call_openai_compatible(base_url=base_url, model=model, prompt=prompt, max_tokens=max_tokens)
        results.append(evaluate_task_from_text(task, corpus_root, output))
    elapsed = time.time() - started
    return {
        "model": model,
        "dataset": str(dataset_path),
        "corpus_root": str(corpus_root),
        "mode": "closed_corpus_selection_baseline",
        "elapsed_seconds": elapsed,
        "summary": summarize_results(results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model on golden-retriever retrieval tasks.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="openbmb/MiniCPM5-1B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    report = evaluate_dataset(
        dataset_path=args.dataset,
        corpus_root=args.corpus_root,
        base_url=args.base_url,
        model=args.model,
        limit=args.limit,
        max_tokens=args.max_tokens,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
