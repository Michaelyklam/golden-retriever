# Experiment log

This log records model/dataset/harness checkpoints. Every row should correspond to a committed repo state and, once training starts, a model artifact or adapter path.

| Date | Checkpoint | Commit | Model | Dataset | Eval suite | Final answer found | Recall | Precision | F1 | Trajectory recall | Notes |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 2026-07-06 | scaffold-v0 | `6bd092a` | MiniCPM5-1B base | none | speed only | — | — | — | — | — | vLLM BF16 on RTX 5060 Ti: ~196.6 single-request output tok/s; ~6,414.5 batched output tok/s. |
| 2026-07-06 | data-pipeline-v0 | `c3672ec` | MiniCPM5-1B base | seed JSONL schema | schema/unit tests | — | — | — | — | — | Context-1 data pipeline distilled into local schema + validation target. |
| 2026-07-06 | base-closed-corpus-v0 | `7e071c9` | MiniCPM5-1B base | `data/base_smoke/tasks.jsonl` | closed-corpus selection | 1.00 | 1.00 | 0.722 | 0.762 | n/a | First base checkpoint. Recall is good on tiny corpus; output precision/discipline is the immediate failure mode. |
| 2026-07-06 | localdocs-v0 | `2dde61b` | MiniCPM5-1B base | `data/generated/localdocs-v0/tasks.jsonl` | generated closed-corpus selection | 0.083 | 0.083 | 0.028 | 0.042 | n/a | First generated dataset. Base model often identifies targets in reasoning but fails final tag emission/document-id fidelity. |
| 2026-07-06 | sft-lap1-localdocs-format | `4099920` | MiniCPM5-1B + LoRA r8 q/v | `data/sft/localdocs-format-lap1/train.jsonl` | local PEFT eval on `localdocs-v0`, thinking disabled | 0.250 | 0.250 | 0.125 | 0.149 | n/a | First SFT lap: 51 examples, 1 epoch, 7 optimizer steps, max length 6144. Improved tag emission but over-returns plausible docs. |
| 2026-07-06 | sft-lap2-thinking-aligned-e8 | `15d313d` | MiniCPM5-1B + LoRA r8 q/v | `data/sft/localdocs-thinking-lap2/train.jsonl` | local PEFT eval on `localdocs-v0`, thinking enabled | 0.417 | 0.417 | 0.253 | 0.292 | n/a | Thinking-enabled lap with prompt-mask alignment, 8 epochs, 56 optimizer steps, max length 8192. Best localdocs result so far; still over-returns distractors. |
| 2026-07-06 | sft-lap3-synthdomains | `5b2905b` | MiniCPM5-1B + LoRA r16 all linear | `data/sft/synthdomains-v1-thinking/train.jsonl` | held-out `synthdomains-v1` task-candidate eval | 1.000 | 1.000 | 1.000 | 1.000 | n/a | 640 synthetic-domain examples over finance/legal/patent/web, 3 epochs, 240 optimizer steps, max length 2048. Strong on matching held-out synthetic task-candidate format, but regressed localdocs task-candidate F1 to 0.111 and full-corpus F1 to 0.131. |
| 2026-07-06 | sft-lap4-mixed-replay | `5b2905b` | MiniCPM5-1B + LoRA r16 all linear | `data/sft/mixed-synthdomains-localdocs-thinking-lap4/train.jsonl` | mixed proxy eval: `synthdomains-v1` held-out + `localdocs-v0` task-candidates | 1.000 | 1.000 | 1.000 | 1.000 | n/a | Mixed replay lap: 640 synthdomains + 51 localdocs task-candidate examples repeated 8x, 3 epochs, 393 optimizer steps, max length 4096. Preserved synthdomains held-out 160/160 and restored localdocs task-candidate 12/12. Full-corpus localdocs remains poor at F1 0.083 because Lap 4 targets candidate-selection/tool-loop style, not whole-corpus stuffing; still needs true Context-1 tool-loop/public-suite adapters. |

## Decision rule

Each experiment should decide the next experiment from measured failure mode:

- low trajectory recall → improve search strategy, corpus indexing, or parallel rollouts;
- high trajectory recall but low output recall → train final ranking/selection and pruning;
- low precision but good final-answer-found → shift reward/curriculum toward precision;
- poor tool formatting → more SFT on well-formed traces before RL;
- high latency → reduce turns, improve parallel calls, quantize, or use smaller context.
