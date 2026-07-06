import pytest
from pydantic import ValidationError

from golden_retriever.dataset import BenchmarkDocument, BenchmarkTask


def test_benchmark_document_round_trip_with_metadata():
    doc = BenchmarkDocument(
        doc_id="wiki:Bronte_Tower:0001",
        text="Charlotte Bronte published Jane Eyre in 1847.",
        url="https://en.wikipedia.org/wiki/Jane_Eyre#Plot",
        title="Jane Eyre",
        metadata={"source_suite": "frames"},
    )

    encoded = doc.model_dump()
    decoded = BenchmarkDocument.model_validate(encoded)

    assert decoded.doc_id == "wiki:Bronte_Tower:0001"
    assert decoded.metadata["source_suite"] == "frames"


def test_benchmark_task_round_trip_for_longseal():
    task = BenchmarkTask(
        task_id="longseal-0001",
        suite="longseal",
        question="Who holds the all-time record at the Grammys for album of the year wins?",
        answer="...",
        positive_doc_ids=["doc-1"],
        corpus_path="benchmarks/longseal/longseal-0001/corpus.jsonl",
        metadata={"source": "fixture"},
    )

    decoded = BenchmarkTask.model_validate_json(task.model_dump_json())

    assert decoded.suite == "longseal"
    assert decoded.positive_doc_ids == ["doc-1"]
    assert decoded.positive_urls == []
    assert decoded.metadata["source"] == "fixture"


def test_benchmark_task_requires_some_positive_target_for_scored_suites():
    with pytest.raises(ValidationError, match="positive_doc_ids or positive_urls"):
        BenchmarkTask(
            task_id="seal0-0001",
            suite="seal0",
            question="Find supporting URLs.",
        )


def test_benchmark_task_allows_hle_without_positive_documents():
    task = BenchmarkTask(
        task_id="hle-0001",
        suite="hle",
        question="Which answer choice is correct?",
        answer="A",
    )

    assert task.suite == "hle"
    assert task.positive_doc_ids == []
