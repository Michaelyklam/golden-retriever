from __future__ import annotations

from pathlib import Path
from typing import Protocol

from golden_retriever.agent_state import AgentAction, AgentTrajectory, Observation
from golden_retriever.agent_tools import Context1ToolRunner
from golden_retriever.tools import LocalCorpus


class InferenceBackend(Protocol):
    """Minimal interface for Pi/model backends that choose the next action."""

    def infer(self, trajectory: AgentTrajectory) -> AgentAction:
        """Return the next action for the current trajectory."""
        ...


class ScriptedBackend:
    """Deterministic backend for unit tests and fixture rollouts."""

    def __init__(self, actions: list[AgentAction]) -> None:
        self.actions = list(actions)
        self.index = 0

    def infer(self, trajectory: AgentTrajectory) -> AgentAction:
        if self.index >= len(self.actions):
            return AgentAction(turn=len(trajectory.actions) + 1, reasoning="No scripted action remaining.")
        action = self.actions[self.index]
        self.index += 1
        return action


def run_agent_loop(
    *,
    corpus_path: str | Path,
    task_id: str,
    query: str,
    backend: InferenceBackend,
    max_turns: int = 8,
) -> AgentTrajectory:
    """Run a deterministic observe → infer → act retrieval loop.

    The loop is deliberately backend-agnostic: Pi, a local PEFT model, a
    teacher API, or a scripted test backend can all supply `AgentAction`s. Tool
    execution stays inside the Context-1-compatible local runner.
    """

    trajectory = AgentTrajectory(task_id=task_id, metadata={"query": query})
    trajectory.observe(Observation(turn=0, content=f"User query: {query}"))
    runner = Context1ToolRunner(LocalCorpus(corpus_path), trajectory)

    for turn in range(1, max_turns + 1):
        proposed = backend.infer(trajectory)
        action = proposed.model_copy(update={"turn": turn})
        trajectory.act(action)
        for tool_call in action.tool_calls:
            runner.run(tool_call)
        if action.final_doc_ids:
            trajectory.metadata["stop_reason"] = "final"
            return trajectory

    trajectory.metadata["stop_reason"] = "max_turns"
    return trajectory
