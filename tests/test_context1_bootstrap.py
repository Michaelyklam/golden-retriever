import json
from pathlib import Path

from golden_retriever.context1_bootstrap import build_bootstrap_corpus
from golden_retriever.dataset import EvidenceDocument, RetrievalTask


def _write_task(path: Path, task: RetrievalTask) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(task.model_dump_json() + "\n", encoding="utf-8")


def test_build_bootstrap_corpus_combines_suites_and_writes_split(tmp_path: Path):
    corpus = tmp_path / "bench" / "hotpotqa" / "corpus"
    (corpus / "doc.md").parent.mkdir(parents=True, exist_ok=True)
    (corpus / "doc.md").write_text("The answer is here." * 20, encoding="utf-8")
    task = RetrievalTask(
        task_id="hp-1",
        domain="hotpotqa",
        difficulty=1,
        hop_count=1,
        question="Where is the answer?",
        answer="here",
        clues=["answer"],
        supporting_documents=[EvidenceDocument(doc_id="doc.md", role="positive", document_quotes=["answer"], clue_quotes=[])],
    )
    _write_task(tmp_path / "bench" / "hotpotqa" / "tasks.jsonl", task)

    report = build_bootstrap_corpus(
        benchmark_root=tmp_path / "bench",
        output_root=tmp_path / "out",
        suites=["hotpotqa"],
        train_fraction=1.0,
        max_chars_per_doc=10,
        include_thinking=True,
    )

    assert report["examples_total"] == 1
    assert report["examples_train"] == 1
    assert Path(report["train_path"]).exists()
    row = json.loads(Path(report["train_path"]).read_text().splitlines()[0])
    assert row["task_id"] == "hp-1"
    assert row["messages"][2]["content"].startswith("<think>")
    assert "The answer" in row["messages"][1]["content"]
