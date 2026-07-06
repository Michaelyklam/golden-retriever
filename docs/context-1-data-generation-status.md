# Context-1 data generation status

Chroma's public `context-1-data-gen` repository is set up as the recipe for the first full fine-tuning corpus, but actually generating the corpus requires external API credentials and a ChromaDB Cloud database.

## Required credentials by domain

| Domain | Required environment |
|---|---|
| web | `ANTHROPIC_API_KEY`, `SERPER_API_KEY`, `JINA_API_KEY`, `OPENAI_API_KEY`, `CHROMA_API_KEY`, `CHROMA_DATABASE` |
| SEC / finance | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `CHROMA_API_KEY`, `CHROMA_DATABASE`, `BASETEN_API_KEY` |
| patents / legal | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `CHROMA_API_KEY`, `CHROMA_DATABASE`, `USPTO_API_KEY`, `SEARCH_API_KEY`, `DATALAB_API_KEY` |
| Epstein email | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `CHROMA_API_KEY`, `CHROMA_DATABASE` |

Run:

```bash
PYTHONPATH=src python3 -m golden_retriever.context1_data_gen
```

The command prints missing credentials plus the exact upstream dry-run commands. No paid API calls are made by the status helper.

## Current status

On 2026-07-06, none of the required data-generation credentials were present in the local shell, project `.env`, upstream repo `.env`, or Hermes `.env`. That blocks true Context-1 training-data generation for now.

## What is already prepared without credentials

The benchmark setup can still materialize public benchmark corpora/manifests from Hugging Face:

```bash
PYTHONPATH=src python3 -m golden_retriever.benchmark_prepare --suite all --output-root data/benchmarks/materialized
```

This produces:

- HotpotQA distractor validation: 7,405 tasks, embedded corpus;
- LongSeal: 252 scoreable tasks, fixed corpus;
- FRAMES: 824 URL-positive tasks;
- Seal-0: 111 URL-positive tasks.

FRAMES and Seal-0 still need browsing/Wikipedia/static-page materialization to run through the local document-ID harness.
