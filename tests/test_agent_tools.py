from pathlib import Path

from golden_retriever.agent_state import AgentTrajectory, ToolCall
from golden_retriever.agent_tools import Context1ToolRunner
from golden_retriever.tools import LocalCorpus


def test_search_corpus_updates_trajectory_and_excludes_seen_docs(tmp_path: Path):
    (tmp_path / "alpha.md").write_text("alpha retrieval target\n", encoding="utf-8")
    (tmp_path / "beta.md").write_text("beta retrieval target\n", encoding="utf-8")
    corpus = LocalCorpus(tmp_path)
    trajectory = AgentTrajectory(task_id="task-1")
    runner = Context1ToolRunner(corpus, trajectory)

    first = runner.run(ToolCall(name="search_corpus", arguments={"query": "retrieval target", "limit": 2}))
    assert first.tool_name == "search_corpus"
    assert {chunk.doc_id for chunk in first.chunks} == {"alpha.md", "beta.md"}
    assert trajectory.encountered_doc_ids == {"alpha.md", "beta.md"}

    second = runner.run(ToolCall(name="search_corpus", arguments={"query": "retrieval target", "limit": 2}))
    assert second.chunks == []


def test_grep_and_read_document_emit_observations(tmp_path: Path):
    (tmp_path / "notes.md").write_text("one\nneedle quote\nthree\n", encoding="utf-8")
    trajectory = AgentTrajectory(task_id="task-2")
    runner = Context1ToolRunner(LocalCorpus(tmp_path), trajectory)

    grep_result = runner.run(ToolCall(name="grep_corpus", arguments={"pattern": "needle", "limit": 1}))
    read_result = runner.run(ToolCall(name="read_document", arguments={"doc_id": "notes.md", "start_line": 2, "num_lines": 1}))

    assert grep_result.chunks[0].doc_id == "notes.md"
    assert read_result.chunks[0].text == "needle quote"
    assert len(trajectory.observations) == 2
    assert "grep_corpus returned 1 chunk" in trajectory.observations[0].content
    assert "read_document returned 1 chunk" in trajectory.observations[1].content


def test_prune_chunks_marks_docs_pruned_and_removes_state_chunks(tmp_path: Path):
    (tmp_path / "keep.md").write_text("useful evidence\n", encoding="utf-8")
    (tmp_path / "drop.md").write_text("irrelevant evidence\n", encoding="utf-8")
    trajectory = AgentTrajectory(task_id="task-3")
    runner = Context1ToolRunner(LocalCorpus(tmp_path), trajectory)
    runner.run(ToolCall(name="search_corpus", arguments={"query": "evidence", "limit": 5}))

    result = runner.run(ToolCall(name="prune_chunks", arguments={"chunk_ids": ["drop.md"]}))

    assert result.removed_doc_ids == ["drop.md"]
    assert trajectory.pruned_doc_ids == {"drop.md"}
    assert trajectory.active_doc_ids == ["keep.md"]
    assert all(chunk.doc_id != "drop.md" for chunk in runner.state.chunks)
