# SFT lap 1 — localdocs format adapter

Date: 2026-07-06

## Step status

- Step 1: inspected harness and training environment.
- Step 2: built deterministic SFT data for exact document-ID/output discipline.
- Step 3: added minimal LoRA SFT trainer and tests.
- Step 4: ran first fine-tuning lap.
- Step 5: evaluated the adapter locally with PEFT.

## Training data

- Source tasks: `data/generated/localdocs-train-v0/tasks.jsonl`
- SFT examples: `data/sft/localdocs-format-lap1/train.jsonl`
- Example count: 51
- Objective: closed-corpus selection output discipline — emit exact `<Document id="...">` tags using grounded positive document IDs.

## Adapter

- Base model: `openbmb/MiniCPM5-1B`
- Adapter: `models/minicpm5-1b-localdocs-format-lap1-lora-6144/`
- LoRA target modules: `q_proj,v_proj`
- LoRA rank / alpha: `8 / 16`
- Max sequence length: `6144`
- Epochs: `1`
- Optimizer steps: `7`
- Mean loss: `1.7797670528000475`
- Final loss: `1.43502938747406`
- Train runtime: `64.31506109237671` seconds

## Eval

Eval file: `results/sft-lap1/localdocs-v0-peft-6144-eval-no-thinking.json`

Generation used the MiniCPM chat template with `enable_thinking=False` so the model starts after the empty thinking block and is pushed directly into final document tags.

```json
{
  "num_tasks": 12,
  "final_answer_found": 0.25,
  "recall": 0.25,
  "precision": 0.125,
  "f1": 0.1488095238095238
}
```

## Baseline comparison

Previous base result on the same `localdocs-v0` dataset:

```json
{
  "num_tasks": 12,
  "final_answer_found": 0.08333333333333333,
  "recall": 0.08333333333333333,
  "precision": 0.027777777777777776,
  "f1": 0.041666666666666664
}
```

Lap 1 improved recall/final-answer-found from `0.083` to `0.25`, precision from `0.028` to `0.125`, and F1 from `0.042` to `0.149`.

## Interpretation

The adapter learned to emit valid document tags much more reliably, but it still over-returns plausible documents and sometimes copies generic tool/corpus IDs. The next lap should train with:

1. explicit `enable_thinking=False` prompt formatting during SFT tokenization;
2. stronger negative examples where only one document is allowed;
3. shorter per-document prompt rendering so the query and all document IDs remain inside the training window without left truncation;
4. a precision-focused validation set separate from the small in-sample sanity set.
