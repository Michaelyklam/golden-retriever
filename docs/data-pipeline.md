# Data pipeline: Context-1 → golden-retriever

This is the working data plan for training MiniCPM5-1B into a fast retrieval subagent.

Source notes are from Chroma's Context-1 technical report: https://www.trychroma.com/research/context-1

## What Context-1 trained

Context-1 is trained as a **retrieval subagent**, not an answer model. Given a query and tools, it should return a ranked set of document IDs/chunks that help a separate reasoning model answer the query.

Chroma's data generation targets two core search capabilities:

1. **Planning** — decompose a high-level, multi-constraint query into a sequence of searches.
2. **Evaluation** — decide which retrieved material is relevant, which is distracting, and when enough evidence has been found.

Their generated tasks use a shared structure across web, finance, legal, and email domains:

1. **Gather supporting documents** containing unique, verifiable facts.
2. **Generate obfuscated clues** from those facts plus a natural-language question and answer.
3. **Verify task validity**: supporting documents must actually support the clues and lead to the answer.
4. **Optionally collect distractors**: documents satisfying some criteria but leading to a wrong answer.
5. **Optionally chain recursively**: bridge the answer from one task into a new task to control hop count.

## Verification method to copy

The report emphasizes extraction-based verification rather than pure LLM judgment.

For each supporting document:

- extract verbatim `document_quotes` from the source document;
- extract matching `clue_quotes` from the generated clues;
- normalize both;
- confirm `document_quotes` occur literally in the document;
- confirm at least one document contains the final answer.

For distractors:

- check whether the distractor contains the answer in any form;
- reject distractors that accidentally leak the final answer.

This makes human spot-checking cheaper: reviewers only inspect quote/clue alignment, not full documents.

## Dataset schema

The first golden-retriever dataset format is JSONL. Each line is one `RetrievalTask`.

```json
{
  "task_id": "web-000001",
  "domain": "web",
  "difficulty": 2,
  "hop_count": 2,
  "question": "On what date was this building formally inaugurated?",
  "answer": "September 20, 1878",
  "clues": [
    "A sacred structure in a western European capital...",
    "The community gained official recognition during the early 1830s..."
  ],
  "supporting_documents": [
    {
      "doc_id": "web/grande-synagogue-bruxelles.md",
      "role": "positive",
      "document_quotes": ["Inaugurated on 20 September 1878"],
      "clue_quotes": ["formally inaugurated"]
    }
  ],
  "distractor_documents": [],
  "metadata": {
    "source": "generated",
    "generator_model": "todo",
    "verified_by": "extraction-check-v0"
  }
}
```

## Evaluation metrics

We will match Context-1's separation of retrieval quality from downstream answer quality.

Output-level metrics:

- **Final answer found** — any final output document/chunk contains the final answer.
- **Recall** — fraction of positive documents returned.
- **Precision** — fraction of returned documents that are positive.
- **F1** — harmonic mean of recall and precision.

Trajectory-level metric:

- **Trajectory recall** — fraction of target documents encountered at any point, even if later pruned.

Behavioral metrics to track during experiments:

- tool calls per turn;
- turns per trajectory;
- prune accuracy;
- token usage;
- latency and output tok/s;
- 1x vs 4x/8x/16x parallel rollout fusion.

## SFT trajectory filtering policy

Context-1 keeps both strong and weak trajectories, but controls the mix.

Initial golden-retriever policy:

| Rollout type | Keep policy |
|---|---|
| `trajectory_recall >= 0.50` and `output_recall >= 0.40` | keep full trajectory |
| low recall but well-formed tool use | sample with diminishing probability |
| zero recall | cap at 5%, dedupe by query |
| high trajectory recall but low output recall | exclude, because it reinforces poor final selection |
| malformed tool calls/output | discard |

## RL direction after SFT

Context-1 uses RLVR after SFT. We should not jump there until we have:

1. validated task schema;
2. deterministic harness traces;
3. baseline MiniCPM5 eval results;
4. enough SFT traces to teach tool formatting and pruning behavior.

When ready, copy the reward shape:

- outcome: recall-heavy F-beta/F-score favoring recall over precision;
- process: trajectory recall;
- binary bonus: final answer found;
- penalties: repeated prune streaks and excessive turn counts;
- curriculum: start easier / recall-heavy, then shift harder / more precision-aware.

## First build targets

1. Implement typed dataset schemas.
2. Implement JSONL validation and quote-grounding checks.
3. Create a tiny hand-written seed dataset over repo docs to exercise the format.
4. Add eval functions for final-answer-found, recall, precision, F1, and trajectory recall.
5. Then generate a real seed corpus/domain. The best first domain is probably **repo/documentation retrieval** because we can deterministically control corpus, positives, distractors, and answer strings before adding web browsing noise.
