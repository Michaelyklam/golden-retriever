# golden-retriever implementation roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a MiniCPM5-powered retrieval subagent and Pi-compatible tool harness inspired by Chroma Context-1.

**Architecture:** Start with deterministic retrieval tools over local corpora, then expose the same contracts to Pi, then add OpenAI-compatible model workers and trace capture for fine-tuning. Keep the retrieval worker separate from the final-answer coordinator.

**Tech Stack:** Python harness, Pi extension/package later, vLLM for MiniCPM5 serving, Chroma/BM25/regex retrieval backends, JSONL traces for training.

---

## Phase 1 — Concrete tool contracts

### Task 1: Harden filesystem corpus loading

**Objective:** Make corpus ingestion deterministic and safe for markdown/text/source documents.

**Files:**
- Modify: `src/golden_retriever/tools.py`
- Test: `tests/test_tools.py`

**Verification:** `pytest -q`

### Task 2: Add structured tool-call schemas

**Objective:** Define JSON schemas for `SearchTool`, `GrepTool`, `ReadDocument`, and `PruneChunksTool` so Pi/model calls can share one interface.

**Files:**
- Create: `src/golden_retriever/schemas.py`
- Test: `tests/test_schemas.py`

**Verification:** model-generated JSON validates before tool execution.

### Task 3: Add trace capture

**Objective:** Persist every prompt, model action, tool result, prune decision, and final ranked evidence set to JSONL.

**Files:**
- Create: `src/golden_retriever/traces.py`
- Modify: `src/golden_retriever/harness.py`
- Test: `tests/test_traces.py`

**Verification:** Run harness and inspect a replayable trace file.

## Phase 2 — Pi integration

### Task 4: Build a Pi package skeleton

**Objective:** Package the retrieval tools so Pi can call them as harness extensions.

**Files:**
- Create: `pi-package/README.md`
- Create: `pi-package/package.json`
- Create: `pi-package/src/index.ts`

**Verification:** `pi` can discover or load the local package in development mode.

### Task 5: Expose retrieval tools to Pi

**Objective:** Wire Pi tool calls to the Python retrieval backend or a native TypeScript equivalent.

**Files:**
- Modify: `pi-package/src/index.ts`
- Create: `pi-package/src/tools.ts`

**Verification:** A Pi session can search, grep, read, and prune over a test corpus.

## Phase 3 — MiniCPM5 worker loop

### Task 6: Add OpenAI-compatible worker client

**Objective:** Let the harness call a local vLLM-served MiniCPM5 endpoint for retrieval decisions.

**Files:**
- Create: `src/golden_retriever/model_client.py`
- Modify: `src/golden_retriever/harness.py`
- Test: `tests/test_model_client.py`

**Verification:** Mock endpoint test passes; manual vLLM smoke test emits tool calls.

### Task 7: Add parallel retrieval swarms

**Objective:** Run 1/4/8/16 MiniCPM retrieval workers over query facets and merge evidence.

**Files:**
- Create: `src/golden_retriever/swarm.py`
- Test: `tests/test_swarm.py`

**Verification:** Parallel workers produce distinct search strategies and deduped ranked chunks.

## Phase 4 — Training data and evaluation

### Task 8: Create synthetic task generator

**Objective:** Generate multi-hop retrieval tasks over local corpora with known answer evidence.

**Files:**
- Create: `src/golden_retriever/synthetic.py`
- Create: `docs/data-generation.md`

**Verification:** JSONL task set includes query, target docs, and required evidence chain.

### Task 9: Add retrieval metrics

**Objective:** Measure recall@k, MRR, evidence coverage, token cost, and latency.

**Files:**
- Create: `src/golden_retriever/eval.py`
- Test: `tests/test_eval.py`

**Verification:** `golden-retriever eval ...` produces metrics for baseline vs agentic runs.

### Task 10: Fine-tuning recipe

**Objective:** Document MiniCPM5 SFT/LoRA path using captured traces.

**Files:**
- Create: `docs/finetuning-minicpm5.md`

**Verification:** A small trace dataset can be converted into chat/tool-call SFT format.
