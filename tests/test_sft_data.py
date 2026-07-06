from pathlib import Path

from golden_retriever.dataset import EvidenceDocument, RetrievalTask
from golden_retriever.sft_data import build_sft_examples, render_gold_completion, write_sft_jsonl


def _task() -> RetrievalTask:
    return RetrievalTask(
        task_id="unit-000001",
        domain="unit",
        difficulty=1,
        hop_count=1,
        question="Which document contains the exact claim?",
        answer="SearchTool performs hybrid keyword",
        clues=["Find evidence for SearchTool."],
        supporting_documents=[
            EvidenceDocument(
                doc_id="docs/search.md",
                role="positive",
                document_quotes=["SearchTool performs hybrid keyword and semantic matching."],
                clue_quotes=["Find evidence for SearchTool."],
            )
        ],
        distractor_documents=[
            EvidenceDocument(
                doc_id="docs/grep.md",
                role="distractor",
                document_quotes=["GrepTool performs regex matching."],
                clue_quotes=["hard distractor"],
            )
        ],
    )


def test_render_gold_completion_uses_exact_doc_id_and_quote():
    completion = render_gold_completion(_task())

    assert completion == (
        '<Document id="docs/search.md"><Justification>'
        'Contains grounded quote: SearchTool performs hybrid keyword and semantic matching.'
        '</Justification></Document>'
    )


def test_render_gold_completion_can_include_short_thinking():
    completion = render_gold_completion(_task(), include_thinking=True)

    assert completion.startswith("<think>\n")
    assert "The query asks for evidence matching" in completion
    assert "docs/search.md" in completion
    assert "docs/grep.md" in completion
    assert "</think>\n\n<Document id=\"docs/search.md\">" in completion


def test_build_sft_examples_can_include_thinking_targets(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "search.md").write_text("SearchTool performs hybrid keyword and semantic matching.\n", encoding="utf-8")

    examples = build_sft_examples([_task()], corpus, include_thinking=True)

    assert examples[0]["messages"][2]["content"].startswith("<think>")


def test_build_sft_examples_can_use_task_candidate_prompt_scope(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "docs").mkdir()
    (corpus / "docs" / "search.md").write_text("SearchTool performs hybrid keyword and semantic matching.\n", encoding="utf-8")
    (corpus / "docs" / "grep.md").write_text("GrepTool performs regex matching.\n", encoding="utf-8")
    (corpus / "unused.md").write_text("Unused document text should not appear.\n", encoding="utf-8")

    examples = build_sft_examples([_task()], corpus, prompt_scope="task-candidates")

    prompt = examples[0]["messages"][1]["content"]
    assert '<Document id="docs/search.md">' in prompt
    assert '<Document id="docs/grep.md">' in prompt
    assert "unused.md" not in prompt


def test_build_sft_examples_can_cap_document_chars(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "docs").mkdir()
    (corpus / "docs" / "search.md").write_text("A" * 200, encoding="utf-8")
    (corpus / "docs" / "grep.md").write_text("B" * 200, encoding="utf-8")

    examples = build_sft_examples([_task()], corpus, prompt_scope="task-candidates", max_chars_per_doc=20)

    prompt = examples[0]["messages"][1]["content"]
    assert "A" * 20 in prompt
    assert "A" * 21 not in prompt


def test_build_sft_examples_matches_eval_prompt_shape(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "search.md").write_text("SearchTool performs hybrid keyword and semantic matching.\n", encoding="utf-8")

    examples = build_sft_examples([_task()], corpus)

    assert len(examples) == 1
    messages = examples[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert "Return only ranked document tags" in messages[0]["content"]
    assert "Query: Which document contains the exact claim?" in messages[1]["content"]
    assert messages[2]["content"].startswith('<Document id="docs/search.md">')


def test_write_sft_jsonl_round_trips(tmp_path: Path):
    path = tmp_path / "sft.jsonl"

    write_sft_jsonl([{"messages": [{"role": "assistant", "content": "ok"}]}], path)

    assert path.read_text(encoding="utf-8") == '{"messages":[{"role":"assistant","content":"ok"}]}\n'
