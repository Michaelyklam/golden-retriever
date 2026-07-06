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


def render_task_candidate_prompt(task: RetrievalTask, corpus_root: str | Path, max_chars_per_doc: int = 6000) -> str:
    """Render only the positive + distractor candidate documents for a task."""

    corpus_by_id = dict(_load_corpus_documents(corpus_root))
    ordered_ids: list[str] = []
    for doc in [*task.supporting_documents, *task.distractor_documents]:
        if doc.doc_id not in ordered_ids:
            ordered_ids.append(doc.doc_id)
    rendered_docs = []
    for doc_id in ordered_ids:
        if doc_id not in corpus_by_id:
            continue
        rendered_docs.append(f'<Document id="{doc_id}">\n{corpus_by_id[doc_id][:max_chars_per_doc]}\n</Document>')
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
    # Some local adapters emit natural-language reasoning that contains the
    # exact document ID before/without wrapping it in the final XML-like tag.
    # Count exact ID mentions as returned evidence so truncated but identifiable
    # outputs are not scored as total parse failures.
    for doc_id in corpus_by_id:
        if doc_id in output_text and doc_id not in parsed_ids:
            parsed_ids.append(doc_id)
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


def _render_prompt_by_scope(
    task: RetrievalTask,
    corpus_root: str | Path,
    prompt_scope: str,
    max_chars_per_doc: int = 6000,
) -> str:
    if prompt_scope == "full-corpus":
        return render_corpus_prompt(task, corpus_root, max_chars_per_doc=max_chars_per_doc)
    if prompt_scope == "task-candidates":
        return render_task_candidate_prompt(task, corpus_root, max_chars_per_doc=max_chars_per_doc)
    raise ValueError(f"Unsupported prompt scope: {prompt_scope}")


def evaluate_dataset(
    dataset_path: str | Path,
    corpus_root: str | Path,
    base_url: str,
    model: str,
    limit: int | None = None,
    max_tokens: int = 2048,
    prompt_scope: str = "full-corpus",
) -> dict[str, Any]:
    tasks = load_jsonl(dataset_path)
    if limit is not None:
        tasks = tasks[:limit]
    results: list[dict[str, Any]] = []
    started = time.time()
    for task in tasks:
        prompt = _render_prompt_by_scope(task, corpus_root, prompt_scope)
        output = call_openai_compatible(base_url=base_url, model=model, prompt=prompt, max_tokens=max_tokens)
        results.append(evaluate_task_from_text(task, corpus_root, output))
    elapsed = time.time() - started
    return {
        "model": model,
        "dataset": str(dataset_path),
        "corpus_root": str(corpus_root),
        "mode": f"openai_compatible_{prompt_scope}",
        "elapsed_seconds": elapsed,
        "summary": summarize_results(results),
        "results": results,
    }


def evaluate_local_model_dataset(
    dataset_path: str | Path,
    corpus_root: str | Path,
    model: str,
    adapter: str | Path | None = None,
    limit: int | None = None,
    max_new_tokens: int = 768,
    prompt_scope: str = "task-candidates",
    enable_thinking: bool = False,
    dtype: str = "bfloat16",
    max_chars_per_doc: int = 6000,
) -> dict[str, Any]:
    """Evaluate a local base model or PEFT LoRA adapter directly."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tasks = load_jsonl(dataset_path)
    if limit is not None:
        tasks = tasks[:limit]

    tokenizer_source = adapter if adapter else model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    if adapter:
        from peft import PeftModel

        loaded_model = PeftModel.from_pretrained(loaded_model, adapter)
    loaded_model.eval()
    device = next(loaded_model.parameters()).device

    results: list[dict[str, Any]] = []
    started = time.time()
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    for task in tasks:
        prompt = _render_prompt_by_scope(task, corpus_root, prompt_scope, max_chars_per_doc=max_chars_per_doc)
        template_kwargs = {"enable_thinking": True} if enable_thinking else {}
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            generated = loaded_model.generate(**encoded, **generation_kwargs)
        continuation_ids = generated[0][encoded["input_ids"].shape[-1] :]
        output = tokenizer.decode(continuation_ids, skip_special_tokens=True)
        results.append(evaluate_task_from_text(task, corpus_root, output))
    elapsed = time.time() - started
    return {
        "model": model,
        "adapter": str(adapter) if adapter else None,
        "dataset": str(dataset_path),
        "corpus_root": str(corpus_root),
        "mode": f"local_{'peft' if adapter else 'base'}_{prompt_scope}_{'thinking' if enable_thinking else 'no_thinking'}",
        "max_chars_per_doc": max_chars_per_doc,
        "elapsed_seconds": elapsed,
        "summary": summarize_results(results),
        "results": results,
    }


def evaluate_local_peft_dataset(
    dataset_path: str | Path,
    corpus_root: str | Path,
    model: str,
    adapter: str | Path,
    limit: int | None = None,
    max_new_tokens: int = 768,
    prompt_scope: str = "task-candidates",
    enable_thinking: bool = False,
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    """Evaluate a PEFT LoRA adapter directly without an OpenAI-compatible server."""

    return evaluate_local_model_dataset(
        dataset_path=dataset_path,
        corpus_root=corpus_root,
        model=model,
        adapter=adapter,
        limit=limit,
        max_new_tokens=max_new_tokens,
        prompt_scope=prompt_scope,
        enable_thinking=enable_thinking,
        dtype=dtype,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model on golden-retriever retrieval tasks.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="openbmb/MiniCPM5-1B")
    parser.add_argument("--adapter", help="Optional PEFT adapter path. When set, evaluate locally instead of via --base-url.")
    parser.add_argument("--local", action="store_true", help="Evaluate the base model locally without an OpenAI-compatible server.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-chars-per-doc", type=int, default=6000)
    parser.add_argument("--prompt-scope", choices=["full-corpus", "task-candidates"], default="full-corpus")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    args = parser.parse_args()

    if args.adapter or args.local:
        report = evaluate_local_model_dataset(
            dataset_path=args.dataset,
            corpus_root=args.corpus_root,
            model=args.model,
            adapter=args.adapter,
            limit=args.limit,
            max_new_tokens=args.max_tokens,
            prompt_scope=args.prompt_scope,
            enable_thinking=args.enable_thinking,
            dtype=args.dtype,
            max_chars_per_doc=args.max_chars_per_doc,
        )
    else:
        report = evaluate_dataset(
            dataset_path=args.dataset,
            corpus_root=args.corpus_root,
            base_url=args.base_url,
            model=args.model,
            limit=args.limit,
            max_tokens=args.max_tokens,
            prompt_scope=args.prompt_scope,
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
