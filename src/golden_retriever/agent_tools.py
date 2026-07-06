from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from golden_retriever.agent_state import AgentTrajectory, Observation, ToolCall
from golden_retriever.tools import Chunk, LocalCorpus, RetrievalState


class ToolResult(BaseModel):
    """Structured result from one Context-1-compatible tool invocation."""

    tool_name: str
    chunks: list[Chunk] = Field(default_factory=list)
    removed_doc_ids: list[str] = Field(default_factory=list)
    message: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Context1ToolRunner:
    """Deterministic Context-1 tool surface backed by the local corpus.

    This is the first Pi-harness extension layer: it turns a validated tool call
    into corpus operations, updates retrieval scratch state, and appends a
    serializable observation to the shared trajectory.
    """

    def __init__(self, corpus: LocalCorpus, trajectory: AgentTrajectory | None = None) -> None:
        self.corpus = corpus
        self.trajectory = trajectory or AgentTrajectory(task_id="local")
        self.state = RetrievalState()

    def run(self, call: ToolCall) -> ToolResult:
        if call.name == "search_corpus":
            result = self._search_corpus(call)
        elif call.name == "grep_corpus":
            result = self._grep_corpus(call)
        elif call.name == "read_document":
            result = self._read_document(call)
        elif call.name == "prune_chunks":
            result = self._prune_chunks(call)
        else:  # pragma: no cover - ToolCall validation should make this unreachable.
            raise ValueError(f"Unsupported Context-1 tool: {call.name}")
        self._observe(result)
        return result

    def _search_corpus(self, call: ToolCall) -> ToolResult:
        query = str(call.arguments.get("query", ""))
        limit = int(call.arguments.get("limit", 5))
        candidates = self.corpus.search(query, limit=max(limit * 4, limit))
        seen_doc_ids = set(self.trajectory.encountered_doc_ids)
        chunks = [chunk for chunk in candidates if chunk.doc_id not in seen_doc_ids][:limit]
        self.state.add(chunks)
        self.trajectory.mark_encountered({chunk.doc_id for chunk in chunks})
        return ToolResult(tool_name="search_corpus", chunks=chunks)

    def _grep_corpus(self, call: ToolCall) -> ToolResult:
        pattern = str(call.arguments.get("pattern", ""))
        limit = int(call.arguments.get("limit", 5))
        chunks = self.corpus.grep(pattern, limit=limit)
        self.state.add(chunks)
        self.trajectory.mark_encountered({chunk.doc_id for chunk in chunks})
        return ToolResult(tool_name="grep_corpus", chunks=chunks)

    def _read_document(self, call: ToolCall) -> ToolResult:
        doc_id = str(call.arguments["doc_id"])
        start_line = int(call.arguments.get("start_line", 1))
        num_lines = int(call.arguments.get("num_lines", 80))
        chunk = self.corpus.read_document(doc_id, start_line=start_line, num_lines=num_lines)
        chunks = [chunk]
        self.state.add(chunks)
        self.trajectory.mark_encountered({chunk.doc_id})
        return ToolResult(tool_name="read_document", chunks=chunks)

    def _prune_chunks(self, call: ToolCall) -> ToolResult:
        # Context-1 names this argument chunk_ids. In the current local corpus
        # each chunk is keyed primarily by doc_id; benchmark adapters can later
        # pass doc_id-like chunk IDs without changing the tool contract.
        chunk_ids = [str(value) for value in call.arguments.get("chunk_ids", [])]
        drop = set(chunk_ids)
        removed = [chunk for chunk in self.state.chunks if chunk.doc_id in drop]
        self.state.chunks = [chunk for chunk in self.state.chunks if chunk.doc_id not in drop]
        self.trajectory.mark_pruned(drop)
        return ToolResult(tool_name="prune_chunks", removed_doc_ids=sorted({chunk.doc_id for chunk in removed}))

    def _observe(self, result: ToolResult) -> None:
        if result.tool_name == "prune_chunks":
            count = len(result.removed_doc_ids)
            noun = "doc" if count == 1 else "docs"
            message = f"{result.tool_name} removed {count} {noun}"
        else:
            count = len(result.chunks)
            noun = "chunk" if count == 1 else "chunks"
            message = f"{result.tool_name} returned {count} {noun}"
        result.message = message
        self.trajectory.observe(Observation(turn=len(self.trajectory.observations), content=message))
