from golden_retriever.benchmark_eval import (
    BenchmarkScore,
    canonicalize_url,
    score_benchmark_output,
)
from golden_retriever.dataset import BenchmarkTask


def test_canonicalize_url_normalizes_case_fragment_and_trailing_slash():
    assert canonicalize_url("HTTPS://Example.COM/path/?b=2&a=1#section") == "https://example.com/path?a=1&b=2"
    assert canonicalize_url("https://example.com/path/") == "https://example.com/path"


def test_score_benchmark_output_doc_ids_tracks_unknowns_and_metrics():
    task = BenchmarkTask(
        task_id="longseal-1",
        suite="longseal",
        question="Find the record holder.",
        positive_doc_ids=["doc-a", "doc-b"],
    )

    score = score_benchmark_output(
        task,
        returned_doc_ids=["doc-a", "doc-x"],
        valid_doc_ids={"doc-a", "doc-b", "doc-c"},
        encountered_doc_ids=["doc-b", "doc-c"],
    )

    assert isinstance(score, BenchmarkScore)
    assert score.recall == 0.5
    assert score.precision == 0.5
    assert score.f1 == 0.5
    assert score.trajectory_recall == 1.0
    assert score.unknown_doc_ids == ["doc-x"]


def test_score_benchmark_output_urls_are_canonicalized():
    task = BenchmarkTask(
        task_id="seal0-1",
        suite="seal0",
        question="Find supporting URLs.",
        positive_urls=["https://Example.com/supporting/page#proof"],
    )

    score = score_benchmark_output(
        task,
        returned_urls=["https://example.com/supporting/page/"],
    )

    assert score.recall == 1.0
    assert score.precision == 1.0
    assert score.f1 == 1.0


def test_score_benchmark_output_final_answer_found_uses_texts():
    task = BenchmarkTask(
        task_id="frames-1",
        suite="frames",
        question="What date?",
        answer="September 20, 1878",
        positive_urls=["https://en.wikipedia.org/wiki/example"],
    )

    score = score_benchmark_output(
        task,
        returned_urls=[],
        output_texts=["The retrieved passage says it was inaugurated on September 20, 1878."],
    )

    assert score.final_answer_found is True
    assert score.recall == 0.0
