# Pi harness extension plan

`golden-retriever` uses the Pi harness as the base integration surface and extends it into a Context-1-style retrieval subagent harness. The goal is not to replace Pi; it is to give Pi a specialized, fast retrieval worker that can search, read, prune, and return evidence for Pi or another coordinator/answerer.

## Base assumption

Pi remains the orchestration/user-facing harness. `golden-retriever` adds a retrieval subagent layer with the same conceptual loop described in Chroma Context-1:

```text
observe → infer → act → observe ... → final ranked evidence
```

The retrieval worker should be model-agnostic:

- MiniCPM5-1B + LoRA locally for cheap parallel retrieval;
- Context-1/gpt-oss-style models for comparison if available;
- frontier teacher models for trajectory generation;
- Pi as coordinator/answerer and integration harness.

## Extension layers

### 1. Pi-compatible tool contracts

Expose the retrieval worker through stable tool contracts that can be called from Pi sessions:

| Tool | Purpose |
|---|---|
| `search_corpus(query)` | Hybrid lexical/dense search over a selected corpus. |
| `grep_corpus(pattern)` | Exact/regex search for names, dates, IDs, and quoted fragments. |
| `read_document(doc_id)` | Read fuller content for a promising document/chunk. |
| `prune_chunks(chunk_ids)` | Remove low-value evidence from the active context while preserving the full trajectory for scoring. |

### 2. Pi-compatible trajectory format

Every run should serialize:

- initial user/coordinator query;
- tool calls and arguments;
- tool observations;
- active context token usage;
- pruned vs retained chunks;
- final ranked evidence IDs;
- score metadata when ground truth exists.

This trajectory format is the bridge between Pi usage, benchmark evaluation, and SFT/RL training data.

### 3. Context-1-compatible benchmark adapters

Public/full-scale adapters feed the same Pi-compatible harness instead of one-off scripts. Priority order:

1. LongSeal fixed-corpus adapter;
2. HotpotQA sanity adapter;
3. FRAMES Wikipedia URL-recall adapter;
4. Seal-0 positive URL adapter;
5. BrowseComp+ or documented access blocker;
6. generated web/SEC/patents/email held-out suites from the Chroma data-generation recipe.

### 4. Training-data bridge

Teacher rollouts run through the same harness used for evaluation. We then filter trajectories with Context-1-style rules:

- keep high trajectory recall + high output recall;
- sample lower-recall but well-formed tool traces;
- cap zero-recall traces;
- discard malformed traces;
- exclude high-exploration / bad-final-selection traces.

### 5. Autonomy / narration convention

Implementation proceeds autonomously in committed slices. At the end of each slice, report:

- what changed;
- why it matters for Pi/Context-1 parity;
- verification command and result;
- commit SHA if pushed;
- next slice to execute.

No confirmation is needed between slices unless credentials, paid APIs, destructive data deletion, or external messages are involved.
