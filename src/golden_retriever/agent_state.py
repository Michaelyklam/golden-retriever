from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Context1ToolName = Literal["search_corpus", "grep_corpus", "read_document", "prune_chunks"]


class Observation(BaseModel):
    """One observation emitted by the Pi-compatible retrieval harness."""

    turn: int
    content: str
    token_usage: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A Context-1-style tool call requested by the retrieval worker."""

    name: Context1ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentAction(BaseModel):
    """One model/action step in the observe → infer → act loop."""

    turn: int
    reasoning: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_doc_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTrajectory(BaseModel):
    """Serializable Pi-compatible retrieval-agent trajectory.

    The trajectory is intentionally separate from benchmark/training data. It is
    the shared artifact Pi integration, public benchmark adapters, and SFT/RL
    trace generation can all consume.
    """

    task_id: str
    observations: list[Observation] = Field(default_factory=list)
    actions: list[AgentAction] = Field(default_factory=list)
    encountered_doc_ids: set[str] = Field(default_factory=set)
    pruned_doc_ids: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def observe(self, observation: Observation) -> None:
        self.observations.append(observation)

    def act(self, action: AgentAction) -> None:
        self.actions.append(action)
        self.mark_encountered(action.final_doc_ids)

    def mark_encountered(self, doc_ids: list[str] | tuple[str, ...] | set[str]) -> None:
        self.encountered_doc_ids.update(doc_ids)

    def mark_pruned(self, doc_ids: list[str] | tuple[str, ...] | set[str]) -> None:
        self.pruned_doc_ids.update(doc_ids)

    @property
    def active_doc_ids(self) -> list[str]:
        return sorted(self.encountered_doc_ids - self.pruned_doc_ids)

    @property
    def final_doc_ids(self) -> list[str]:
        final_ids: list[str] = []
        seen: set[str] = set()
        for action in self.actions:
            for doc_id in action.final_doc_ids:
                if doc_id not in seen:
                    seen.add(doc_id)
                    final_ids.append(doc_id)
        return final_ids
