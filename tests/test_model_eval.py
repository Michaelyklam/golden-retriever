from pathlib import Path

from golden_retriever.dataset import RetrievalTask
from golden_retriever.model_eval import evaluate_task_from_text, parse_document_ids, render_corpus_prompt


def test_parse_document_ids_from_context1_output():
    output = (
        '<Document id="context-1-notes.md"><Justification>Contains the tool list.</Justification></Document>\n'
        '<Document id=roadmap.md><Justification>Secondary planning context.</Justification></Document>'
    )

    assert parse_document_ids(output) == ["context-1-notes.md", "roadmap.md"]


def test_render_corpus_prompt_contains_doc_ids(tmp_path: Path):
    (tmp_path / "doc.md").write_text("SearchTool performs hybrid search.\n", encoding="utf-8")
    task = RetrievalTask(
        task_id="docs-1",
        domain="docs",
        difficulty=1,
        hop_count=1,
        question="Which tool performs hybrid search?",
        answer="SearchTool",
        clues=["hybrid search"],
        supporting_documents=[
            {
                "doc_id": "doc.md",
                "role": "positive",
                "document_quotes": ["SearchTool performs hybrid search"],
                "clue_quotes": ["hybrid search"],
            }
        ],
    )

    prompt = render_corpus_prompt(task, tmp_path)

    assert '<Document id="doc.md">' in prompt
    assert "Which tool performs hybrid search?" in prompt
    assert "SearchTool performs hybrid search" in prompt


def test_evaluate_task_from_text_filters_unknown_document_ids(tmp_path: Path):
    (tmp_path / "positive.md").write_text("The answer is GrepTool.\n", encoding="utf-8")
    task = RetrievalTask(
        task_id="docs-filter",
        domain="docs",
        difficulty=1,
        hop_count=1,
        question="Which tool performs text pattern matching?",
        answer="GrepTool",
        clues=["text pattern matching"],
        supporting_documents=[
            {
                "doc_id": "positive.md",
                "role": "positive",
                "document_quotes": ["GrepTool"],
                "clue_quotes": ["text pattern matching"],
            }
        ],
    )

    result = evaluate_task_from_text(
        task,
        tmp_path,
        '<Document id="document_id"><Justification>format placeholder</Justification></Document>'
        '<Document id="positive.md"><Justification>real output</Justification></Document>'
        '<Document id="missing.md"><Justification>hallucinated id</Justification></Document>',
    )

    assert result["returned_doc_ids"] == ["positive.md"]
    assert result["unknown_doc_ids"] == ["document_id", "missing.md"]


def test_evaluate_task_from_text_scores_model_output(tmp_path: Path):
    (tmp_path / "positive.md").write_text("The answer is GrepTool.\n", encoding="utf-8")
    (tmp_path / "negative.md").write_text("The answer is SearchTool.\n", encoding="utf-8")
    task = RetrievalTask(
        task_id="docs-2",
        domain="docs",
        difficulty=1,
        hop_count=1,
        question="Which tool performs text pattern matching?",
        answer="GrepTool",
        clues=["text pattern matching"],
        supporting_documents=[
            {
                "doc_id": "positive.md",
                "role": "positive",
                "document_quotes": ["GrepTool"],
                "clue_quotes": ["text pattern matching"],
            }
        ],
    )

    result = evaluate_task_from_text(
        task,
        tmp_path,
        '<Document id="positive.md"><Justification>It contains the answer.</Justification></Document>',
    )

    assert result["returned_doc_ids"] == ["positive.md"]
    assert result["final_answer_found"] is True
    assert result["recall"] == 1.0
    assert result["precision"] == 1.0
    assert result["f1"] == 1.0
