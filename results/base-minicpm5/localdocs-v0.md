# localdocs-v0 base MiniCPM5 evaluation

Date: 2026-07-06

## Step status

- Step 1: deterministic synthetic task generator added.
- Step 2: first generated local-docs dataset created and quote-grounding validated.
- Step 3: base MiniCPM5-1B evaluated on the generated dataset.

## Dataset

- Corpus: `data/generated/localdocs-v0/corpus/`
- Tasks: `data/generated/localdocs-v0/tasks.jsonl`
- Task count: 12
- Generator: `golden-retriever-generate`
- Validation: `golden-retriever-validate data/generated/localdocs-v0/tasks.jsonl --corpus-root data/generated/localdocs-v0/corpus`

## Command

```bash
golden-retriever-model-eval \
  --dataset data/generated/localdocs-v0/tasks.jsonl \
  --corpus-root data/generated/localdocs-v0/corpus \
  --model openbmb/MiniCPM5-1B \
  --base-url http://127.0.0.1:8000/v1 \
  --max-tokens 2048 \
  --output results/base-minicpm5/localdocs-v0-closed-corpus.json
```

## Results

```json
{
  "num_tasks": 12,
  "final_answer_found": 0.08333333333333333,
  "recall": 0.08333333333333333,
  "precision": 0.027777777777777776,
  "f1": 0.041666666666666664
}
```

## Interpretation

This generated set is much harsher than the tiny hand-written smoke set. The base model frequently identifies the right document in its reasoning, but fails to emit the final XML-like document tags before the output limit, or emits hallucinated path variants such as `base-smoke/...` instead of the actual `base_smoke/...` IDs.

Measured failure mode:

1. **Output discipline failure** — long reasoning, final tags missing/truncated.
2. **Document ID fidelity failure** — path hallucinations / normalization errors.
3. **Selection precision failure** — when tags are emitted, extra plausible docs appear.

This strongly supports starting data prep with SFT traces for exact output format, document-id copying, and concise final ranking before attempting RL.
