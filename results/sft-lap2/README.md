# SFT lap 2 — thinking-enabled localdocs adapter

Date: 2026-07-06

## Goal

Respond to the lap-1 finding that disabling thinking improves tag emission but weakens the intended retrieval-reasoning behavior. Lap 2 trains the model to keep a short thinking trace and still terminate with exact `<Document id="...">` tags.

## Data

- Source tasks: `data/generated/localdocs-train-v0/tasks.jsonl`
- SFT examples: `data/sft/localdocs-thinking-lap2/train.jsonl`
- Example count: 51
- Tokenized length: roughly 5.4k–5.5k tokens/example, no examples over 8k.
- Target shape: short `<think>...</think>` reasoning followed by exact supporting document tags.

## Best adapter

- Base model: `openbmb/MiniCPM5-1B`
- Adapter: `models/minicpm5-1b-localdocs-thinking-lap2-aligned-lora-8192-e8/`
- LoRA target modules: `q_proj,v_proj`
- LoRA rank / alpha: `8 / 16`
- Max length: `8192`
- Epochs: `8`
- Optimizer steps: `56`
- Train runtime: `521.7142577171326` seconds
- Mean loss: `0.4959040422987777`
- Final loss: `0.02178681455552578`

Important implementation detail: the trainer uses `--enable-thinking` so prompt masking is aligned with MiniCPM's generation prompt, which already prefixes `<think>\n`. Without this alignment, the adapter trains against a slightly different assistant-prefix shape and performs worse.

## Eval

Best eval file: `results/sft-lap2/localdocs-v0-peft-thinking-aligned-e8-eval.json`

```json
{
  "num_tasks": 12,
  "final_answer_found": 0.4166666666666667,
  "recall": 0.4166666666666667,
  "precision": 0.25277777777777777,
  "f1": 0.2916666666666667
}
```

## Progression on `localdocs-v0`

| Checkpoint | Thinking | Final answer found | Recall | Precision | F1 |
|---|---|---:|---:|---:|---:|
| Base MiniCPM5-1B | default | 0.083 | 0.083 | 0.028 | 0.042 |
| SFT lap 1 | disabled at eval | 0.250 | 0.250 | 0.125 | 0.149 |
| SFT lap 2 aligned e8 | enabled | 0.417 | 0.417 | 0.253 | 0.292 |

## Interpretation

Thinking-enabled SFT works, but only after aligning the training prompt mask to the thinking generation prompt and training longer than the one-epoch format lap. The model now often reaches the correct positive document while still over-returning distractors. The next improvement should focus on precision and termination:

1. add explicit negative/contrastive traces where distractors are named in thinking and excluded from the final tags;
2. add a stricter parser/metric for malformed repeated `>` output;
3. add held-out localdocs and then start implementing Context-1 benchmark adapters rather than overfitting this tiny in-repo set.
