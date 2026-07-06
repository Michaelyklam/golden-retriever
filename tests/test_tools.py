from pathlib import Path

from golden_retriever.harness import run_local_retrieval
from golden_retriever.tools import LocalCorpus, RetrievalState


def test_search_and_grep_find_chunks(tmp_path: Path):
    (tmp_path / "a.md").write_text("# Alpha\nSearchTool finds hybrid retrieval clues.\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Beta\nGrepTool finds exact patterns.\n", encoding="utf-8")

    corpus = LocalCorpus(tmp_path)

    search_hits = corpus.search("hybrid retrieval")
    assert search_hits
    assert search_hits[0].doc_id == "a.md"
    assert search_hits[0].source_tool == "SearchTool"

    grep_hits = corpus.grep("exact patterns")
    assert grep_hits
    assert grep_hits[0].doc_id == "b.md"
    assert grep_hits[0].source_tool == "GrepTool"


def test_read_document(tmp_path: Path):
    (tmp_path / "doc.md").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    corpus = LocalCorpus(tmp_path)

    chunk = corpus.read_document("doc.md", start_line=2, num_lines=2)

    assert chunk.text == "two\nthree"
    assert chunk.start_line == 2
    assert chunk.end_line == 3


def test_prune_chunks(tmp_path: Path):
    (tmp_path / "keep.md").write_text("useful evidence\n", encoding="utf-8")
    (tmp_path / "drop.md").write_text("irrelevant tangent\n", encoding="utf-8")
    corpus = LocalCorpus(tmp_path)
    state = RetrievalState()
    state.add(corpus.search("evidence tangent", limit=10))

    removed = state.prune(keep_doc_ids={"keep.md"})

    assert removed
    assert all(chunk.doc_id == "keep.md" for chunk in state.chunks)


def test_run_local_retrieval(tmp_path: Path):
    (tmp_path / "context.md").write_text("Context-1 uses SearchTool and PruneChunksTool for agentic retrieval.\n", encoding="utf-8")

    result = run_local_retrieval(tmp_path, "What tools support agentic retrieval?")

    assert result["query"]
    assert "Available Tools" in result["prompt"]
    assert result["chunks"]
    assert result["trajectory"]["observations"]
    assert result["trajectory"]["encountered_doc_ids"]
