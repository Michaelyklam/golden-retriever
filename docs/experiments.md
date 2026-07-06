# Experiment log

This log records model/dataset/harness checkpoints. Every row should correspond to a committed repo state and, once training starts, a model artifact or adapter path.

| Date | Checkpoint | Commit | Model | Dataset | Eval suite | Final answer found | Recall | Precision | F1 | Trajectory recall | Notes |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 2026-07-06 | scaffold-v0 | `6bd092a` | MiniCPM5-1B base | none | speed only | — | — | — | — | — | vLLM BF16 on RTX 5060 Ti: ~196.6 single-request output tok/s; ~6,414.5 batched output tok/s. |
| 2026-07-06 | data-pipeline-v0 | `c3672ec` | MiniCPM5-1B base | seed JSONL schema | schema/unit tests | — | — | — | — | — | Context-1 data pipeline distilled into local schema + validation target. |
| 2026-07-06 | base-closed-corpus-v0 | `7bf1837` | MiniCPM5-1B base | `data/base_smoke/tasks.jsonl` | closed-corpus selection | 1.00 | 1.00 | 0.722 | 0.762 | n/a | First base checkpoint. Recall is good on tiny corpus; output precision/discipline is the immediate failure mode. |

## Decision rule

Each experiment should decide the next experiment from measured failure mode:

- low trajectory recall → improve search strategy, corpus indexing, or parallel rollouts;
- high trajectory recall but low output recall → train final ranking/selection and pruning;
- low precision but good final-answer-found → shift reward/curriculum toward precision;
- poor tool formatting → more SFT on well-formed traces before RL;
- high latency → reduce turns, improve parallel calls, quantize, or use smaller context.
