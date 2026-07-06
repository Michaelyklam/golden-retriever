from golden_retriever.agent_state import (
    AgentAction,
    AgentTrajectory,
    Observation,
    ToolCall,
)


def test_agent_trajectory_round_trip_preserves_pi_harness_state():
    trajectory = AgentTrajectory(task_id="longseal-0001")
    trajectory.observe(Observation(turn=0, content="Initial Pi coordinator query", token_usage=42))
    trajectory.act(
        AgentAction(
            turn=1,
            reasoning="Search for the award record holder.",
            tool_calls=[ToolCall(name="search_corpus", arguments={"query": "Grammy album of the year most wins"})],
        )
    )
    trajectory.mark_encountered(["doc-a", "doc-b"])
    trajectory.mark_pruned(["doc-b"])
    trajectory.act(AgentAction(turn=2, final_doc_ids=["doc-a"]))

    decoded = AgentTrajectory.model_validate_json(trajectory.model_dump_json())

    assert decoded.task_id == "longseal-0001"
    assert decoded.observations[0].token_usage == 42
    assert decoded.actions[0].tool_calls[0].name == "search_corpus"
    assert decoded.encountered_doc_ids == {"doc-a", "doc-b"}
    assert decoded.pruned_doc_ids == {"doc-b"}
    assert decoded.final_doc_ids == ["doc-a"]


def test_tool_call_accepts_only_context1_tool_names():
    valid = ToolCall(name="prune_chunks", arguments={"chunk_ids": ["doc-b"]})
    assert valid.name == "prune_chunks"

    try:
        ToolCall(name="answer_question", arguments={})
    except Exception as exc:
        assert "answer_question" in str(exc)
    else:
        raise AssertionError("non-Context-1 tool call should fail validation")


def test_active_doc_ids_exclude_pruned_docs():
    trajectory = AgentTrajectory(task_id="frames-0001")
    trajectory.mark_encountered(["wiki-a", "wiki-b", "wiki-c"])
    trajectory.mark_pruned(["wiki-b"])

    assert trajectory.active_doc_ids == ["wiki-a", "wiki-c"]
