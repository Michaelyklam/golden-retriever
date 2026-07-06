# Context-1 public benchmark setup report

Date: 2026-07-06

## Benchmark setup status

Prepared with:

```bash
PYTHONPATH=src python3 -m golden_retriever.benchmark_prepare --suite all --output-root data/benchmarks/materialized
```

Materialized locally, but ignored by git because the corpora are large:

| Suite | Local status | Tasks | Docs / corpus | Run status |
|---|---:|---:|---:|---|
| HotpotQA distractor validation | prepared | 7,405 | 73,693 docs | ready for local candidate-eval; full model run not launched because the long-running background command was blocked by the execution guard |
| LongSeal | prepared | 252 scoreable | 6,529 docs | smoke run completed; full model run not launched because the long-running background command was blocked by the execution guard |
| FRAMES | task manifest prepared | 824 | URL-positive only | needs Wikipedia/static-page materialization before local document-ID eval |
| Seal-0 | task manifest prepared | 111 | URL-positive only | needs web browsing/scraping/static-page materialization before local document-ID eval |
| BrowseComp+ | discovered on Hugging Face | n/a | multi-GB parquet corpus/query set | not downloaded yet; requires a dedicated long-running setup job |

## Accessible sources discovered

| Suite | Source |
|---|---|
| Seal-0 / LongSeal | `vtllms/sealqa` on Hugging Face |
| FRAMES | `google/frames-benchmark` on Hugging Face |
| HotpotQA | `hotpotqa/hotpot_qa` on Hugging Face |
| BrowseComp+ | `Tevatron/browsecomp-plus`, `Tevatron/browsecomp-plus-corpus` on Hugging Face |

## Current measured model scores

These are **smoke** scores on the first 10 LongSeal tasks, not final full-suite scores.

| Model | Suite | Tasks | Final answer found | Recall | Precision | F1 |
|---|---|---:|---:|---:|---:|---:|
| MiniCPM5-1B base | LongSeal smoke | 10 | 0.200 | 0.250 | 0.300 | 0.267 |
| Lap 4 mixed replay adapter | LongSeal smoke | 10 | 0.500 | 1.000 | 0.383 | 0.550 |
| Context-1 (1x), reported by Chroma | LongSeal full | n/a | n/a | n/a | n/a | 0.650 |
| Context-1 (4x), reported by Chroma | LongSeal full | n/a | n/a | n/a | n/a | 0.790 |

Interpretation: the Lap 4 adapter is directionally better than base on the LongSeal smoke slice, but this is not parity evidence until full LongSeal runs complete and FRAMES/Seal-0/HotpotQA/BrowseComp+ are runnable through equivalent harnesses.

## Data-generation blocker

The actual Chroma `context-1-data-gen` corpus generation cannot run in this environment yet because none of the required credentials are present:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `SERPER_API_KEY`
- `JINA_API_KEY`
- `CHROMA_API_KEY`
- `CHROMA_DATABASE`
- `BASETEN_API_KEY`
- `USPTO_API_KEY`
- `SEARCH_API_KEY`
- `DATALAB_API_KEY`

The helper below now reports per-domain missing credentials and exact upstream dry-run commands:

```bash
PYTHONPATH=src python3 -m golden_retriever.context1_data_gen
```

No paid calls are made by the helper.

## Execution guard blocker

I attempted to launch a long-running full benchmark evaluation job for:

1. base MiniCPM5-1B on full LongSeal;
2. Lap 4 on full LongSeal;
3. base MiniCPM5-1B on full HotpotQA;
4. Lap 4 on full HotpotQA.

The terminal safety guard blocked that background command before it started, reporting that it could not proceed without user response. I did not retry that same long-running command.

## Next safe slices

1. Add Wikipedia/static-page materialization for FRAMES.
2. Add web/static-page materialization for Seal-0, or mark it unavailable without browsing credentials.
3. Add BrowseComp+ downloader with explicit disk/runtime budget and shard-level resumability.
4. Run full LongSeal/HotpotQA in a foreground or approved long-running job.
5. Once credentials exist, run `context-1-data-gen` to produce the first true full fine-tuning corpus.
