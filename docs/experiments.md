# Experiment log

This log records model/dataset/harness checkpoints. Every row should correspond to a committed repo state and, once training starts, a model artifact or adapter path.

| Date | Checkpoint | Commit | Model | Dataset | Eval suite | Final answer found | Recall | Precision | F1 | Trajectory recall | Notes |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 2026-07-06 | scaffold-v0 | `6bd092a` | MiniCPM5-1B base | none | speed only | — | — | — | — | — | vLLM BF16 on RTX 5060 Ti: ~196.6 single-request output tok/s; ~6,414.5 batched output tok/s. |
| 2026-07-06 | data-pipeline-v0 | `c3672ec` | MiniCPM5-1B base | seed JSONL schema | schema/unit tests | — | — | — | — | — | Context-1 data pipeline distilled into local schema + validation target. |
| 2026-07-06 | base-closed-corpus-v0 | `7e071c9` | MiniCPM5-1B base | `data/base_smoke/tasks.jsonl` | closed-corpus selection | 1.00 | 1.00 | 0.722 | 0.762 | n/a | First base checkpoint. Recall is good on tiny corpus; output precision/discipline is the immediate failure mode. |
| 2026-07-06 | localdocs-v0 | `2dde61b` | MiniCPM5-1B base | `data/generated/localdocs-v0/tasks.jsonl` | generated closed-corpus selection | 0.083 | 0.083 | 0.028 | 0.042 | n/a | First generated dataset. Base model often identifies targets in reasoning but fails final tag emission/document-id fidelity. |
| 2026-07-06 | sft-lap1-localdocs-format | pending | MiniCPM5-1B + LoRA r8 q/v | `data/sft/localdocs-format-lap1/train.jsonl` | local PEFT eval on `localdocs-v0`, thinking disabled | 0.250 | 0.250 | 0.125 | 0.149 | n/a | First SFT lap: 51 examples, 1 epoch, 7 optimizer steps, max length 6144. Improved tag emission but over-returns plausible docs. |

## Decision rule

Each experiment should decide the next experiment from measured failure mode:

- low trajectory recall → improve search strategy, corpus indexing, or parallel rollouts;
- high trajectory recall but low output recall → train final ranking/selection and pruning;
- low precision but good final-answer-found → shift reward/curriculum toward precision;
- poor tool formatting → more SFT on well-formed traces before RL;
- high latency → reduce turns, improve parallel calls, quantize, or use smaller context.
