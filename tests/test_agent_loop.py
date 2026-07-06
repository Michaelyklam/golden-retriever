from pathlib import Path

from golden_retriever.agent_state import AgentAction, ToolCall
from golden_retriever.inference import ScriptedBackend, run_agent_loop


def test_scripted_agent_loop_searches_then_returns_final_docs(tmp_path: Path):
    (tmp_path / "answer.md").write_text("Context-1 style agents search before returning evidence.\n", encoding="utf-8")
    backend = ScriptedBackend(
        actions=[
            AgentAction(
                turn=1,
                reasoning="Find evidence first.",
                tool_calls=[ToolCall(name="search_corpus", arguments={"query": "Context-1 evidence", "limit": 3})],
            ),
            AgentAction(turn=2, reasoning="Return the supporting document.", final_doc_ids=["answer.md"]),
        ]
    )

    trajectory = run_agent_loop(
        corpus_path=tmp_path,
        task_id="loop-1",
        query="What should the retrieval agent do?",
        backend=backend,
        max_turns=4,
    )

    assert trajectory.final_doc_ids == ["answer.md"]
    assert trajectory.encountered_doc_ids == {"answer.md"}
    assert [action.turn for action in trajectory.actions] == [1, 2]
    assert trajectory.observations[0].content == "User query: What should the retrieval agent do?"
    assert any("search_corpus returned" in observation.content for observation in trajectory.observations)


def test_agent_loop_stops_at_max_turns_without_final_docs(tmp_path: Path):
    (tmp_path / "answer.md").write_text("loop budget evidence\n", encoding="utf-8")
    backend = ScriptedBackend(
        actions=[
            AgentAction(tool_calls=[ToolCall(name="search_corpus", arguments={"query": "evidence"})], turn=99),
            AgentAction(tool_calls=[ToolCall(name="grep_corpus", arguments={"pattern": "evidence"})], turn=99),
        ]
    )

    trajectory = run_agent_loop(
        corpus_path=tmp_path,
        task_id="loop-2",
        query="Find evidence",
        backend=backend,
        max_turns=1,
    )

    assert len(trajectory.actions) == 1
    assert trajectory.final_doc_ids == []
    assert trajectory.metadata["stop_reason"] == "max_turns"
