from pathlib import Path

from golden_retriever.dataset import RetrievalTask, validate_dataset, validate_quote_grounding
from golden_retriever.eval import f1, final_answer_found, precision, recall, trajectory_recall


def test_retrieval_task_roles_validate():
    task = RetrievalTask(
        task_id="docs-1",
        domain="docs",
        difficulty=1,
        hop_count=1,
        question="What tool searches text patterns?",
        answer="GrepTool",
        clues=["Text pattern matching"],
        supporting_documents=[
            {
                "doc_id": "context-1-notes.md",
                "role": "positive",
                "document_quotes": ["GrepTool"],
                "clue_quotes": ["Text pattern matching"],
            }
        ],
    )

    assert task.supporting_documents[0].role == "positive"


def test_quote_grounding(tmp_path: Path):
    (tmp_path / "doc.md").write_text("SearchTool performs hybrid semantic and keyword search.\n", encoding="utf-8")
    task = RetrievalTask(
        task_id="docs-2",
        domain="docs",
        difficulty=1,
        hop_count=1,
        question="Which tool performs hybrid search?",
        answer="SearchTool",
        clues=["hybrid semantic and keyword search"],
        supporting_documents=[
            {
                "doc_id": "doc.md",
                "role": "positive",
                "document_quotes": ["hybrid semantic and keyword search"],
                "clue_quotes": ["hybrid search"],
            }
        ],
    )

    assert validate_quote_grounding(task, tmp_path) == []


def test_dataset_jsonl_validation(tmp_path: Path):
    (tmp_path / "doc.md").write_text("PruneChunksTool removes irrelevant chunks.\n", encoding="utf-8")
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(
        '{"task_id":"docs-3","domain":"docs","difficulty":1,"hop_count":1,'
        '"question":"What removes irrelevant chunks?","answer":"PruneChunksTool",'
        '"clues":["removes irrelevant chunks"],'
        '"supporting_documents":[{"doc_id":"doc.md","role":"positive",'
        '"document_quotes":["removes irrelevant chunks"],"clue_quotes":["removes irrelevant chunks"]}]}'
        "\n",
        encoding="utf-8",
    )

    assert validate_dataset(dataset, tmp_path) == []


def test_eval_metrics():
    returned = ["a", "b", "c"]
    positives = ["b", "c", "d"]

    assert recall(returned, positives) == 2 / 3
    assert precision(returned, positives) == 2 / 3
    assert f1(returned, positives) == 2 / 3
    assert trajectory_recall(["x", "b", "d"], positives) == 2 / 3
    assert final_answer_found(["noise", "The answer is GrepTool."], "GrepTool")
