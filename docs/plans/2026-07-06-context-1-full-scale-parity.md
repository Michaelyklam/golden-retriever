# Context-1 Full-Scale Parity Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Replace toy local proxy experiments with a reproducible Context-1-style full-scale training/evaluation pipeline using Chroma's technical report and public `context-1-data-gen` repo as the recipe, then report scores on comparable benchmark suites before making any parity claims.

**Architecture:** Build a compatible observe → infer → act retrieval-agent harness with the Context-1 tool surface (`search_corpus`, `grep_corpus`, `read_document`, `prune_chunks`), convert public Chroma data-generation outputs into `golden-retriever` datasets, generate/collect SFT trajectories from stronger teacher rollouts, then train MiniCPM adapters and evaluate against full benchmark adapters. Keep benchmark adapters separate from training data adapters so reported scores are reproducible and not contaminated by training traces.

**Tech Stack:** Python 3.12, `pytest`, Pydantic schemas, local BM25/regex search first, optional dense embeddings/ChromaDB later, Hugging Face Transformers/PEFT LoRA for MiniCPM, optional TRL/Atropos-style RLVR later, external `chroma-core/context-1-data-gen` pinned by commit.

---

## Non-negotiable scope decisions

1. Stop optimizing on the tiny `localdocs-v0` / `synthdomains-v1` proxy scores. Keep them only as smoke/regression tests.
2. Do not claim Context-1 parity until we have comparable public-suite numbers for at least LongSeal, Seal-0, FRAMES, HotpotQA, and a generated held-out multi-domain suite.
3. Treat the Chroma technical report as the benchmark recipe. Treat `context-1-data-gen` as the public data recipe. Record every deviation from Chroma's harness because their full harness/eval code is not public yet.
4. Keep model weights out of GitHub. Commit code, configs, docs, metrics, generated task manifests, lightweight model metadata, and updated graphs.
5. Build benchmark adapters before another training lap. We need hard comparable numbers, not vibes.

## Source references

- `docs/context-1-source-inventory.md`
- Chroma Context-1 report: https://www.trychroma.com/research/context-1
- Chroma data generation repo: https://github.com/chroma-core/context-1-data-gen
- Context-1 model card: https://huggingface.co/chromadb/context-1

---

## Phase 0: Repository hygiene and source pinning

### Task 0.1: Add external-source manifest

**Objective:** Track upstream Context-1 sources and pinned commits without vendoring large unrelated files.

**Files:**
- Create: `external/context-1-data-gen.manifest.json`
- Modify: `.gitignore` if external scratch clones are introduced
- Test: none

**Implementation:**

Create a manifest like:

```json
{
  "name": "chroma-core/context-1-data-gen",
  "url": "https://github.com/chroma-core/context-1-data-gen",
  "pinned_commit": "<fill from git ls-remote>",
  "retrieved_at": "2026-07-06",
  "notes": "Used as recipe/reference; not vendored wholesale."
}
```

**Verification:**

Run:

```bash
git ls-remote https://github.com/chroma-core/context-1-data-gen.git HEAD
python3 -m json.tool external/context-1-data-gen.manifest.json >/dev/null
```

Expected: manifest parses and pinned commit matches upstream HEAD at capture time.

### Task 0.2: Freeze current toy experiments as smoke tests

**Objective:** Make clear that toy scores are no longer the main target while preserving regression coverage.

**Files:**
- Modify: `docs/experiments.md`
- Modify: `README.md`

**Implementation:**

Add a short note: `localdocs-v0` and `synthdomains-v1` are now smoke/regression suites only; comparable Context-1 scoring lives in the benchmark matrix.

**Verification:**

Run:

```bash
grep -n "smoke" README.md docs/experiments.md
```

Expected: both docs explicitly label toy suites as smoke/regression only.

---

## Phase 1: Comparable benchmark harness before training

### Task 1.1: Define benchmark suite schema

**Objective:** Add typed schema for public/full-scale benchmark manifests independent from training data.

**Files:**
- Modify: `src/golden_retriever/dataset.py`
- Create: `tests/test_benchmark_schema.py`

**Implementation sketch:**

Add Pydantic models:

```python
class BenchmarkDocument(BaseModel):
    doc_id: str
    text: str
    url: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class BenchmarkTask(BaseModel):
    task_id: str
    suite: Literal["generated", "browsecomp_plus", "seal0", "longseal", "frames", "hotpotqa", "hle"]
    question: str
    answer: str | None = None
    positive_doc_ids: list[str] = Field(default_factory=list)
    positive_urls: list[str] = Field(default_factory=list)
    corpus_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

**Test:**

```python
def test_benchmark_task_round_trip():
    task = BenchmarkTask(
        task_id="longseal-0001",
        suite="longseal",
        question="Who holds the record?",
        answer="...",
        positive_doc_ids=["doc-1"],
    )
    assert task.suite == "longseal"
    assert task.positive_doc_ids == ["doc-1"]
```

**Verification:**

Run:

```bash
pytest tests/test_benchmark_schema.py -q
```

Expected: new schema tests pass.

### Task 1.2: Implement benchmark scoring module

**Objective:** Score outputs in the same family as Context-1: final-answer-found, recall, precision, F1, trajectory recall, and URL/doc normalization.

**Files:**
- Create: `src/golden_retriever/benchmark_eval.py`
- Create: `tests/test_benchmark_eval.py`

**Implementation notes:**

- Reuse existing document-tag parser from `model_eval.py` if possible.
- Add URL canonicalization for Seal/FRAMES: lowercase host, strip fragments, normalize trailing slashes.
- Support fact-level matching where multiple chunks can satisfy one positive fact.
- Track `unknown_doc_ids` separately from wrong-but-valid IDs.

**Verification:**

Run:

```bash
pytest tests/test_benchmark_eval.py tests/test_model_eval.py -q
```

Expected: scoring handles perfect, partial, empty, malformed, and URL-normalized cases.

### Task 1.3: Add public benchmark result table generator

**Objective:** Generate a graph/table comparable to Chroma's public benchmark table.

**Files:**
- Modify: `src/golden_retriever/progress.py`
- Create: `docs/context-1-benchmark-matrix.md`
- Test: `tests/test_progress.py`

**Implementation:**

The table should include rows for:

- Context-1 (1x) reported target;
- Context-1 (4x) reported target;
- `gpt-oss-20b` reported target;
- our base MiniCPM baseline;
- our latest MiniCPM adapter.

Columns:

```text
BrowseComp+ | LongSeal | Seal0 | FRAMES | HotpotQA | generated aggregate
```

Use `n/a` until an adapter result exists. Do not mix toy proxy scores into this table.

**Verification:**

Run:

```bash
PYTHONPATH=src python3 -m golden_retriever.progress --experiments docs/experiments.md --readme README.md --svg assets/metrics-progression.svg
pytest tests/test_progress.py -q
```

Expected: README/graph still generate and matrix is present.

---

## Phase 2: Context-1-compatible tool harness

### Task 2.1: Define observe/infer/act trajectory models

**Objective:** Add serializable trajectory state for training rollouts and evaluation.

**Files:**
- Create: `src/golden_retriever/agent_state.py`
- Create: `tests/test_agent_state.py`

**Implementation sketch:**

Models:

```python
class Observation(BaseModel):
    turn: int
    content: str
    token_usage: int | None = None

class ToolCall(BaseModel):
    name: Literal["search_corpus", "grep_corpus", "read_document", "prune_chunks"]
    arguments: dict[str, Any]

class AgentAction(BaseModel):
    turn: int
    reasoning: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_doc_ids: list[str] = Field(default_factory=list)

class AgentTrajectory(BaseModel):
    task_id: str
    observations: list[Observation] = Field(default_factory=list)
    actions: list[AgentAction] = Field(default_factory=list)
    encountered_doc_ids: set[str] = Field(default_factory=set)
    pruned_doc_ids: set[str] = Field(default_factory=set)
```

**Verification:**

Run:

```bash
pytest tests/test_agent_state.py -q
```

Expected: round-trip JSON serialization preserves encountered/pruned IDs.

### Task 2.2: Implement local search_corpus with BM25-first fallback

**Objective:** Provide a deterministic local `search_corpus(query)` tool before adding ChromaDB/dense embeddings.

**Files:**
- Modify: `src/golden_retriever/tools.py`
- Create: `tests/test_agent_tools.py`

**Implementation notes:**

- Use lightweight lexical scoring first to avoid new services.
- Return ranked chunks with doc IDs, scores, and snippets.
- Exclude previously encountered IDs.
- Keep interface compatible with future dense/RRF implementation.

**Verification:**

Run:

```bash
pytest tests/test_agent_tools.py tests/test_tools.py -q
```

Expected: search returns relevant docs, excludes seen IDs, respects result limit.

### Task 2.3: Implement grep/read/prune tools

**Objective:** Complete the four Context-1 tool contracts.

**Files:**
- Modify: `src/golden_retriever/tools.py`
- Modify: `tests/test_agent_tools.py`

**Implementation notes:**

- `grep_corpus(pattern)` returns up to 5 matching chunks.
- `read_document(doc_id)` returns full doc/chunks within token budget.
- `prune_chunks(chunk_ids)` updates active context but not full trajectory.

**Verification:**

Run:

```bash
pytest tests/test_agent_tools.py -q
```

Expected: all four tools match deterministic fixtures.

### Task 2.4: Add token budget manager

**Objective:** Match Context-1 soft/hard budget behavior.

**Files:**
- Create: `src/golden_retriever/token_budget.py`
- Create: `tests/test_token_budget.py`

**Implementation notes:**

- Track active context token estimate.
- Append `[Token usage: used/limit]` after turns.
- Soft threshold injects prune/conclude warning.
- Hard cutoff rejects non-prune tool calls.

**Verification:**

Run:

```bash
pytest tests/test_token_budget.py -q
```

Expected: soft and hard thresholds trigger at configured values.

### Task 2.5: Implement deterministic agent loop

**Objective:** Run observe → infer → act with pluggable inference backends.

**Files:**
- Modify: `src/golden_retriever/harness.py`
- Create: `src/golden_retriever/inference.py`
- Create: `tests/test_agent_loop.py`

**Implementation notes:**

- `InferenceBackend` interface: `infer(trajectory, tools) -> AgentAction`.
- `ScriptedBackend` for unit tests.
- Later backends: local PEFT model, OpenAI/Anthropic teacher.

**Verification:**

Run:

```bash
pytest tests/test_agent_loop.py -q
```

Expected: scripted backend can search, read, prune, and terminate with final docs.

---

## Phase 3: Data-generation bridge from Context-1 repo

### Task 3.1: Write converter for Chroma web-domain outputs

**Objective:** Convert `context-1-data-gen` web JSON outputs into `BenchmarkTask` / training task JSONL.

**Files:**
- Create: `src/golden_retriever/context1_convert.py`
- Create: `tests/fixtures/context1_web_sample.json`
- Create: `tests/test_context1_convert.py`

**Implementation notes:**

Input shape from Chroma README:

```json
{
  "tasks": [{
    "question": "...",
    "truth": "...",
    "supporting_items": [{"id": "https://...", "item_quotes": [], "contains_truth": true}],
    "items_and_contents": {"https://...": "page text..."},
    "distractors": []
  }]
}
```

Output should create:

- task manifest with positive URLs/doc IDs;
- static corpus files/chunks;
- quote-grounding metadata.

**Verification:**

Run:

```bash
pytest tests/test_context1_convert.py -q
```

Expected: sample Chroma JSON converts to deterministic task/corpus output.

### Task 3.2: Add SEC, patents, and email converters

**Objective:** Convert every public `context-1-data-gen` domain into common schema.

**Files:**
- Modify: `src/golden_retriever/context1_convert.py`
- Add fixtures under `tests/fixtures/context1_{sec,patents,epstein}_sample.json`
- Modify: `tests/test_context1_convert.py`

**Implementation notes:**

- SEC positives are chunk IDs from filing chunks.
- Patents positives are `positive_docids` from extracted rejection evals.
- Epstein positives are thread IDs/chunks.
- Preserve source domain and original IDs in metadata.

**Verification:**

Run:

```bash
pytest tests/test_context1_convert.py -q
```

Expected: all four domain fixture converters pass.

### Task 3.3: Add generation runner wrapper

**Objective:** Provide a wrapper that runs upstream data-gen commands with pinned config and records exact environment requirements.

**Files:**
- Create: `scripts/context1_generate.py`
- Create: `configs/context1-data-gen.example.yaml`
- Create: `docs/context-1-data-generation.md`

**Implementation notes:**

This wrapper should not require secrets for dry-run. It should print missing required env vars by domain and generate runnable commands:

```bash
python -m agentic_search_data_gen.domains.web --seeds ... --output ... --collection ... --extension-rounds 1
python -m agentic_search_data_gen.domains.sec -o ... -c ... --identity "Name email@example.com"
python -m agentic_search_data_gen.domains.patents --seeds ... --output ... --collection ...
python -m agentic_search_data_gen.domains.epstein -o ... -c ...
```

**Verification:**

Run:

```bash
python3 scripts/context1_generate.py --config configs/context1-data-gen.example.yaml --dry-run
```

Expected: prints commands and missing env vars without making network/API calls.

---

## Phase 4: Public benchmark adapters

### Task 4.1: Implement LongSeal fixed-corpus adapter first

**Objective:** Get the first genuinely comparable public benchmark running.

**Files:**
- Create: `src/golden_retriever/benchmarks/longseal.py`
- Create: `tests/test_longseal_adapter.py`
- Create: `docs/benchmarks/longseal.md`

**Implementation notes:**

- Convert each LongSeal question + document set into a fixed local corpus.
- Chunk documents into 512-token chunks as the report describes.
- Score positive document/chunk retrieval.
- Record known label-noise caveat from the report.

**Verification:**

Run:

```bash
pytest tests/test_longseal_adapter.py -q
```

Expected: fixture task produces expected positive IDs and chunking.

### Task 4.2: Implement HotpotQA adapter

**Objective:** Add a simpler saturation benchmark as a sanity check.

**Files:**
- Create: `src/golden_retriever/benchmarks/hotpotqa.py`
- Create: `tests/test_hotpotqa_adapter.py`
- Create: `docs/benchmarks/hotpotqa.md`

**Verification:**

Run:

```bash
pytest tests/test_hotpotqa_adapter.py -q
```

Expected: known fixture maps supporting facts/docs correctly.

### Task 4.3: Implement FRAMES adapter

**Objective:** Add Wikipedia-positive-URL recall benchmark.

**Files:**
- Create: `src/golden_retriever/benchmarks/frames.py`
- Create: `tests/test_frames_adapter.py`
- Create: `docs/benchmarks/frames.md`

**Implementation notes:**

- Prefer static Wikipedia snapshots if available.
- If live Wikipedia/API is used, record timestamp, normalized URLs, and filtered inaccessible positives.
- Match report behavior: Serper search with `site:wikipedia.org` and Wikipedia API page content where possible.

**Verification:**

Run:

```bash
pytest tests/test_frames_adapter.py -q
```

Expected: URL normalization and positive coverage filtering are deterministic.

### Task 4.4: Implement Seal-0 adapter

**Objective:** Add curated URL-positive benchmark.

**Files:**
- Create: `src/golden_retriever/benchmarks/seal.py`
- Create: `tests/test_seal_adapter.py`
- Create: `docs/benchmarks/seal.md`

**Verification:**

Run:

```bash
pytest tests/test_seal_adapter.py -q
```

Expected: fixture positive URLs score correctly despite URL normalization edge cases.

### Task 4.5: Implement BrowseComp+ adapter or document access blocker

**Objective:** Decide whether BrowseComp+ is accessible/reproducible; implement if possible, otherwise explicitly block it.

**Files:**
- Create: `src/golden_retriever/benchmarks/browsecomp_plus.py` if dataset is accessible
- Or create/update: `docs/benchmarks/browsecomp-plus.md` with access blocker

**Verification:**

Run:

```bash
pytest tests/test_browsecomp_plus_adapter.py -q
```

Expected if implemented: fixture scoring works. If blocked: docs explain missing dataset/source and exact next action.

---

## Phase 5: Baseline comparable evaluations

### Task 5.1: Run MiniCPM base model on public adapters

**Objective:** Establish comparable baseline before any new training.

**Files:**
- Create result files under `results/context1-public-baselines/`
- Modify: `docs/experiments.md`
- Modify: `docs/context-1-benchmark-matrix.md`

**Command template:**

```bash
PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/michael/minicpm5-1b-bench/.venv/bin/python -m golden_retriever.benchmark_eval \
  --suite longseal \
  --model openbmb/MiniCPM5-1B \
  --output results/context1-public-baselines/minicpm5-longseal.json
```

**Verification:**

Run:

```bash
python3 -m json.tool results/context1-public-baselines/minicpm5-longseal.json >/dev/null
```

Expected: result JSON includes score fields and model metadata.

### Task 5.2: Run latest Lap 4 adapter on public adapters

**Objective:** Measure current local adapter against real public benchmarks.

**Files:**
- Create: `results/context1-public-lap4/*.json`
- Modify: `docs/experiments.md`
- Modify: `docs/context-1-benchmark-matrix.md`

**Verification:**

Run same commands as baseline with `--peft-adapter models/minicpm5-1b-mixed-synthdomains-localdocs-thinking-lap4-r16-all-lora-4096-e3`.

Expected: result JSONs exist for every implemented public adapter.

---

## Phase 6: Full-scale SFT data and teacher trajectories

### Task 6.1: Generate or import multi-domain Context-1-style tasks

**Objective:** Build a real training/held-out dataset, not a toy proxy.

**Files:**
- Generated data under `data/context1-generated/`
- SFT traces under `data/sft/context1-trajectories-v1/`
- Manifest: `data/context1-generated/manifest.json`

**Target scale:**

Start with a pragmatic smaller full-scale target if API cost is constrained, but keep shape compatible:

| Domain | Train target | Held-out target |
|---|---:|---:|
| web | 1,000 | 200 |
| SEC/finance | 1,000 | 200 |
| patents/legal | 1,000 | 200 |
| email | 1,000 | 200 |

Chroma used over 8,000 synthetic generated tasks; this staged target gets us to the same order of magnitude without blocking on a single giant run.

**Verification:**

Run:

```bash
python3 scripts/summarize_context1_data.py data/context1-generated/manifest.json
```

Expected: per-domain counts, verification pass rate, and held-out split counts are printed.

### Task 6.2: Roll out teacher trajectories through our harness

**Objective:** Generate SFT traces that teach tool use, query decomposition, pruning, and final evidence selection.

**Files:**
- Create: `src/golden_retriever/teacher_rollout.py`
- Create: `tests/test_teacher_rollout.py`
- Generated traces under `data/sft/context1-trajectories-v1/raw/`

**Implementation notes:**

- Backend can be Anthropic/OpenAI/local teacher later.
- Store full trajectory: prompt, tool calls, tool observations, prunes, final docs.
- Store encountered docs so trajectory recall can be computed.

**Verification:**

Run fixture rollout with `ScriptedBackend`:

```bash
pytest tests/test_teacher_rollout.py -q
```

Expected: trajectory has search/read/prune/final events and encountered-doc IDs.

### Task 6.3: Filter SFT trajectories like Context-1

**Objective:** Implement the report's filtering policy.

**Files:**
- Create: `src/golden_retriever/trajectory_filter.py`
- Create: `tests/test_trajectory_filter.py`

**Rules:**

- Keep full trajectories when `trajectory_recall >= 0.50` and `output_recall >= 0.40`.
- Sample lower-recall well-formed trajectories at diminishing probability.
- Cap zero-recall trajectories at 5%, deduped by query.
- Exclude high trajectory recall / low output recall traces.
- Discard malformed outputs/tool calls.

**Verification:**

Run:

```bash
pytest tests/test_trajectory_filter.py -q
```

Expected: every rule has an explicit test.

### Task 6.4: Convert trajectories to chat SFT format

**Objective:** Train on full observe/reason/act traces, not only final document tags.

**Files:**
- Modify: `src/golden_retriever/sft_data.py`
- Modify: `tests/test_sft_data.py`

**Implementation notes:**

- Preserve tool-call messages and observations in chat format.
- Keep final evidence selection as supervised target.
- Align thinking-enabled prompt masking with generation-time chat-template settings.

**Verification:**

Run:

```bash
pytest tests/test_sft_data.py tests/test_train_lora_sft.py -q
```

Expected: token-label masking still aligns with `enable_thinking=True`.

---

## Phase 7: Full-scale training laps

### Task 7.1: Train Context-1-style SFT Lap 5

**Objective:** First real non-toy SFT adapter over full trajectories.

**Files:**
- Output adapter metadata under `models/minicpm5-1b-context1-sft-lap5-*`
- Weights excluded by `.gitignore`
- Metrics under `training_metrics.json`

**Command template:**

```bash
PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/michael/minicpm5-1b-bench/.venv/bin/python -m golden_retriever.train_lora_sft \
  --model openbmb/MiniCPM5-1B \
  --train-file data/sft/context1-trajectories-v1/train.jsonl \
  --output-dir models/minicpm5-1b-context1-sft-lap5-r16-8192-e3 \
  --epochs 3 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --max-length 8192 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --enable-thinking \
  --log-every 20
```

**Verification:**

Run held-out generated benchmark and at least one public adapter before another training lap.

### Task 7.2: Add RLVR/CISPO feasibility spike

**Objective:** Decide whether to implement RLVR locally or use an existing framework.

**Files:**
- Create: `docs/plans/context1-rlvr-spike.md`
- Optional: `src/golden_retriever/rl_reward.py`
- Tests: `tests/test_rl_reward.py`

**Implementation notes:**

Minimum reward function:

```text
reward = recall_heavy_f_score + trajectory_recall + final_answer_bonus - repeated_prune_penalty - turn_count_penalty
```

For local 16GB hardware, start with small LoRA groups and fewer rollouts than Chroma's 128×8 setup, then scale only if stable.

**Verification:**

Run:

```bash
pytest tests/test_rl_reward.py -q
```

Expected: reward function matches hand-computed examples.

---

## Phase 8: Comparable reporting and decision gates

### Task 8.1: Update experiment table and graph only from result JSONs

**Objective:** Prevent hand-edited benchmark claims.

**Files:**
- Modify: `src/golden_retriever/progress.py`
- Modify: `docs/experiments.md`
- Modify: `assets/metrics-progression.svg`

**Verification:**

Run:

```bash
PYTHONPATH=src python3 -m golden_retriever.progress --experiments docs/experiments.md --readme README.md --svg assets/metrics-progression.svg
pytest tests/test_progress.py -q
```

Expected: graph/table include full-suite scores and label missing adapters as `n/a`.

### Task 8.2: Define parity gate

**Objective:** Make pass/fail explicit.

**Files:**
- Modify: `docs/context-1-benchmark-matrix.md`

**Parity criteria:**

- **Local proxy parity:** irrelevant except as smoke/regression.
- **Public benchmark parity:** latest MiniCPM adapter matches or beats Context-1 (1x) on at least 4/5 public suite final-answer-found scores, with no suite below 90% of Context-1 (1x).
- **Generated benchmark parity:** latest MiniCPM adapter closes at least 80% of the gap between MiniCPM base and Context-1 on output recall/F1/final-answer-found over held-out generated tasks.
- **Operational parity:** tool-call validity >= 99%, parseability >= 99%, no catastrophic regression on older smoke suites.

**Verification:**

Run:

```bash
python3 scripts/check_context1_parity.py results/context1-public-*/ latest
```

Expected: machine-readable pass/fail with missing-suite blockers.

---

## Immediate next execution order

1. Phase 1 schema/scoring/matrix.
2. Phase 2 deterministic local agent harness.
3. Phase 4 LongSeal + HotpotQA adapters.
4. Baseline MiniCPM + Lap 4 evals on those adapters.
5. Add FRAMES and Seal-0 adapters.
6. Only then generate full SFT trajectories and train Lap 5.

This order gives us comparable numbers as early as possible and prevents another round of training against the wrong objective.
