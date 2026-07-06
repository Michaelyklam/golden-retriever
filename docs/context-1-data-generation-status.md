# Context-1 data generation status

Michael's chosen path: **do not use Anthropic API keys and do not use OpenAI API keys** for the teacher/model-generation layer. Use the **OpenAI Codex subscription/OAuth flow via Codex CLI** instead.

Chroma's public `context-1-data-gen` repository remains the recipe/reference for domain structure and task-generation logic, but the upstream implementation assumes API-key based Anthropic/OpenAI/Chroma Cloud services. `golden-retriever` will adapt the recipe into a local/Codex-backed pipeline rather than requiring those keys.

## Provider modes

Run the status helper:

```bash
PYTHONPATH=src python3 -m golden_retriever.context1_data_gen --provider codex-cli
```

This is now the default. It checks:

- whether the `codex` CLI is installed;
- whether Codex OAuth/subscription auth exists under `~/.codex/auth.json` or Hermes auth exists under `~/.hermes/auth.json`;
- whether any remaining non-LLM domain prerequisites are required by the selected materializer.

For comparison only, the upstream Chroma requirements can still be printed with:

```bash
PYTHONPATH=src python3 -m golden_retriever.context1_data_gen --provider upstream
```

## Upstream Chroma credential assumptions, not our default

| Domain | Upstream required environment |
|---|---|
| web | `ANTHROPIC_API_KEY`, `SERPER_API_KEY`, `JINA_API_KEY`, `OPENAI_API_KEY`, `CHROMA_API_KEY`, `CHROMA_DATABASE` |
| SEC / finance | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `CHROMA_API_KEY`, `CHROMA_DATABASE`, `BASETEN_API_KEY` |
| patents / legal | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `CHROMA_API_KEY`, `CHROMA_DATABASE`, `USPTO_API_KEY`, `SEARCH_API_KEY`, `DATALAB_API_KEY` |
| Epstein email | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `CHROMA_API_KEY`, `CHROMA_DATABASE` |

Those are documented so we understand Chroma's original pipeline, but they are **not** the target path for this project.

## Current local status

As of 2026-07-06:

- Codex CLI has been installed with `npm install -g @openai/codex`.
- Codex/Hermes auth files exist locally.
- `golden-retriever` status now reports `codex-cli` as the LLM provider path.
- The Codex path reports no Anthropic/OpenAI API-key requirement.

## What is already prepared without API-key teacher generation

The benchmark setup can materialize public benchmark corpora/manifests from Hugging Face:

```bash
PYTHONPATH=src python3 -m golden_retriever.benchmark_prepare --suite all --output-root data/benchmarks/materialized
```

This produces:

- HotpotQA distractor validation: 7,405 tasks, embedded corpus;
- LongSeal: 252 scoreable tasks, fixed corpus;
- FRAMES: 824 URL-positive tasks;
- Seal-0: 111 URL-positive tasks.

FRAMES and Seal-0 still need static-page/Wikipedia materialization to run through the local document-ID harness.

## Bootstrap SFT corpus now running

The keyless first-stage training-data pipeline is operational:

```bash
PYTHONPATH=src python3 -m golden_retriever.context1_bootstrap \
  --benchmark-root data/benchmarks/materialized \
  --output-root data/sft/context1-bootstrap-v1 \
  --max-chars-per-doc 1200 \
  --train-fraction 0.9
```

Current output:

| Split | Examples |
|---|---:|
| train | 6,891 |
| eval | 766 |
| total | 7,657 |

Source suites:

- LongSeal: 252 examples;
- HotpotQA: 7,405 examples.

This first corpus is `gold_label_bootstrap`: it uses official benchmark labels to create supervised evidence-selection examples with thinking targets, without Anthropic/OpenAI API keys. Codex CLI teacher trajectories are the next augmentation layer.

## Next implementation change

Replace the upstream API-key teacher calls with a Codex-backed teacher rollout layer:

```text
Benchmark/Context-1 task → Pi-compatible tool loop → Codex CLI teacher action proposal → trajectory JSONL → SFT data
```

This keeps the Context-1 recipe shape while avoiding Anthropic/OpenAI API keys.
