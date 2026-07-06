# Base MiniCPM5-1B evaluation — closed-corpus smoke

Date: 2026-07-06

Model: `openbmb/MiniCPM5-1B` base

Serving stack:

- vLLM server on `http://127.0.0.1:8000/v1`
- BF16
- `VLLM_USE_FLASHINFER_SAMPLER=0`
- GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM
- Driver: 580.159.03

## Scope

This is the first base-model retrieval checkpoint before data preparation and fine-tuning.

It is **not yet** the full Context-1 observe/reason/act tool loop. It is a closed-corpus document-selection baseline: the model sees a small corpus in the prompt and must return ranked `<Document id="...">` tags. This tests whether the base model can follow the retrieval-subagent output format and select relevant documents from a constrained corpus.

## Commands

```bash
golden-retriever-validate data/base_smoke/tasks.jsonl \
  --corpus-root data/base_smoke/corpus

golden-retriever-model-eval \
  --dataset data/base_smoke/tasks.jsonl \
  --corpus-root data/base_smoke/corpus \
  --model openbmb/MiniCPM5-1B \
  --base-url http://127.0.0.1:8000/v1 \
  --max-tokens 2048 \
  --output results/base-minicpm5/base-smoke-closed-corpus.json
```

## Results

Dataset: `data/base_smoke/tasks.jsonl`

```json
{
  "num_tasks": 3,
  "final_answer_found": 1.0,
  "recall": 1.0,
  "precision": 0.7222222222222222,
  "f1": 0.7619047619047619
}
```

Secondary sanity dataset: `data/seed/tasks.jsonl`

```json
{
  "num_tasks": 2,
  "final_answer_found": 1.0,
  "recall": 1.0,
  "precision": 0.25,
  "f1": 0.4
}
```

The seed dataset has label leakage because several repo docs mention the same Context-1 tools; treat it as schema smoke only, not a meaningful benchmark.

## Observations

- The base model can identify the right evidence on this tiny closed-corpus setup: final-answer-found and recall are both 1.0 on the base-smoke set.
- Precision is imperfect because the model sometimes emits additional relevant-looking but non-positive documents.
- MiniCPM5 often reasons for a while before emitting final document tags. A 512-token output budget truncated outputs before final tags on some examples; `--max-tokens 2048` fixed that for this checkpoint.
- The model repeats the requested XML-like format inside its reasoning, which can produce placeholder IDs such as `document_id`. The evaluator now filters unknown document IDs before scoring and records them as `unknown_doc_ids`.

## Decision for next experiment

The measured failure mode is **selection precision / output discipline**, not basic recall on small corpora.

Next data-prep target:

1. Build a larger generated dataset with explicit positives and distractors.
2. Capture traces that reward returning only sufficient positive evidence.
3. Add a stricter output parser/eval mode for final-answer tags only once we move from closed-corpus prompting to the real tool loop.
