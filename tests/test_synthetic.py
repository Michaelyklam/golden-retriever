from pathlib import Path

from golden_retriever.dataset import load_jsonl, validate_dataset
from golden_retriever.synthetic import extract_fact_candidates, generate_tasks, select_distractors, write_tasks_jsonl


def test_extract_fact_candidates_uses_substantive_sentences(tmp_path: Path):
    (tmp_path / "alpha.md").write_text(
        "# Alpha\n\nGrepTool performs regex text pattern matching over a corpus for exact phrase retrieval.\n\nTiny.\n",
        encoding="utf-8",
    )

    candidates = extract_fact_candidates(tmp_path, min_chars=40)

    assert len(candidates) == 1
    assert candidates[0].doc_id == "alpha.md"
    assert candidates[0].quote == "GrepTool performs regex text pattern matching over a corpus for exact phrase retrieval."


def test_extract_fact_candidates_does_not_join_across_headings(tmp_path: Path):
    (tmp_path / "note.md").write_text(
        "Source: https://example.com\n\n# Core idea\n\nContext-1 is a retrieval subagent model with enough detail for extraction.\n",
        encoding="utf-8",
    )

    candidates = extract_fact_candidates(tmp_path, min_chars=40)

    assert [c.quote for c in candidates] == ["Context-1 is a retrieval subagent model with enough detail for extraction."]


def test_select_distractors_prefers_token_overlap(tmp_path: Path):
    (tmp_path / "positive.md").write_text("GrepTool performs regex matching over documents.\n", encoding="utf-8")
    (tmp_path / "near.md").write_text("SearchTool performs keyword matching over documents.\n", encoding="utf-8")
    (tmp_path / "far.md").write_text("Banana bread uses ripe fruit and flour.\n", encoding="utf-8")
    candidates = extract_fact_candidates(tmp_path, min_chars=20)
    positive = next(c for c in candidates if c.doc_id == "positive.md")

    distractors = select_distractors(positive, candidates, count=1)

    assert [d.doc_id for d in distractors] == ["near.md"]


def test_extract_fact_candidates_can_take_multiple_per_document(tmp_path: Path):
    (tmp_path / "multi.md").write_text(
        "First substantive sentence has enough length to become one candidate.\n"
        "Second substantive sentence also has enough length to become another candidate.\n",
        encoding="utf-8",
    )

    candidates = extract_fact_candidates(tmp_path, min_chars=40, max_candidates_per_doc=2)

    assert [c.quote for c in candidates] == [
        "First substantive sentence has enough length to become one candidate.",
        "Second substantive sentence also has enough length to become another candidate.",
    ]


def test_generate_tasks_write_and_validate(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "grep.md").write_text(
        "GrepTool performs regex text pattern matching over a corpus for exact phrase retrieval.\n",
        encoding="utf-8",
    )
    (corpus / "search.md").write_text(
        "SearchTool performs hybrid keyword and semantic matching over a corpus for broad retrieval.\n",
        encoding="utf-8",
    )
    output = tmp_path / "tasks.jsonl"

    tasks = generate_tasks(corpus, limit=2, distractor_count=1, domain="unit")
    write_tasks_jsonl(tasks, output)

    loaded = load_jsonl(output)
    assert len(loaded) == 2
    assert loaded[0].task_id == "unit-000001"
    assert loaded[0].supporting_documents[0].document_quotes
    assert validate_dataset(output, corpus) == []
