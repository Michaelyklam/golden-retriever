# Context-1 notes

Source: https://www.trychroma.com/research/context-1

## Core idea

Context-1 is a retrieval subagent model. It does not answer the user's question directly. It retrieves and ranks evidence for another reasoning model.

The agent loop is useful because many real queries require multi-hop retrieval: the result of one search informs what to search next. Instead of assuming a single vector/keyword query is enough, the retrieval model decomposes the problem, searches iteratively, reads deeper where necessary, and prunes low-value context.

## Tool surface from the published prompt

The published prompt names four tools:

- `SearchTool`: hybrid semantic and keyword search.
- `GrepTool`: text pattern matching.
- `ReadDocument`: read specific document snippets that look promising but incomplete.
- `PruneChunksTool`: remove irrelevant chunks to free up context space.

## Process requirements from the prompt

The retrieval subagent should:

1. Break the query into key concepts and explicit information needs.
2. Build a search strategy for each concept.
3. Search from several distinct, non-overlapping angles.
4. Use multiple parallel tool calls when possible.
5. After every search round, decide:
   - What do I know?
   - What should I search for next?
   - What should I prune?
   - Do I have enough information?
6. Avoid getting stuck on a single search strategy.
7. Return relevant documents/chunks for another model, not a final answer.

## golden-retriever interpretation

`golden-retriever` will treat this as a two-layer system:

- **Retrieval worker:** MiniCPM5-1B, tool-using, fast, parallelizable. Produces ranked evidence.
- **Coordinator / answerer:** Pi or a stronger frontier model. Assigns retrieval facets, merges results, and writes the final answer.

The first milestone is a deterministic local harness that exposes these tools over a filesystem corpus. The second milestone is a Pi package/extension that exposes the same tool contracts inside Pi sessions. The third milestone is trace generation + fine-tuning for MiniCPM5-1B.
