from golden_retriever.benchmark_prepare import (
    hotpot_row_to_task,
    longseal_row_to_task,
    parse_frames_row,
)


def test_hotpot_row_to_task_maps_supporting_titles_to_doc_ids():
    row = {
        "id": "hp1",
        "question": "Were both people American?",
        "answer": "yes",
        "supporting_facts": {"title": ["Person A", "Person B"], "sent_id": [0, 0]},
        "context": {
            "title": ["Person A", "Person B", "Distractor"],
            "sentences": [["A was American."], ["B was American."], ["Other text."]],
        },
    }

    task, docs = hotpot_row_to_task(row)

    assert task.task_id == "hotpotqa-hp1"
    assert [d.doc_id for d in task.supporting_documents] == ["hotpotqa/hp1/person-a.md", "hotpotqa/hp1/person-b.md"]
    assert [d.doc_id for d in task.distractor_documents] == ["hotpotqa/hp1/distractor.md"]
    assert docs["hotpotqa/hp1/person-a.md"].startswith("# Person A")


def test_longseal_row_to_task_uses_gold_url_positives_and_docs():
    row = {
        "question": "Who holds the record?",
        "answer": "Serban Ghenea",
        "urls": ["https://example.com/gold"],
        "golds": [{"title": "Gold", "text": "Serban Ghenea has five wins.", "url": "https://example.com/gold"}],
        "30_docs": [
            {"title": "Gold", "text": "Serban Ghenea has five wins.", "url": "https://example.com/gold"},
            {"title": "Noise", "text": "Distractor", "url": "https://example.com/noise"},
        ],
    }

    task, docs = longseal_row_to_task(row, 7)

    assert task.task_id == "longseal-000007"
    assert len(task.supporting_documents) == 1
    assert "/gold-" in task.supporting_documents[0].doc_id
    assert len(task.distractor_documents) == 1
    assert "Serban Ghenea" in next(iter(docs.values()))


def test_parse_frames_row_collects_wiki_links():
    row = {
        "Prompt": "Find the name",
        "Answer": "Jane Ballou",
        "wikipedia_link_1": "https://en.wikipedia.org/wiki/James_Buchanan",
        "wikipedia_link_2": "",
        "wiki_links": "['https://en.wikipedia.org/wiki/Harriet_Lane']",
    }

    task = parse_frames_row(row, 3)

    assert task.task_id == "frames-000003"
    assert task.question == "Find the name"
    assert task.answer == "Jane Ballou"
    assert task.positive_urls == [
        "https://en.wikipedia.org/wiki/James_Buchanan",
        "https://en.wikipedia.org/wiki/Harriet_Lane",
    ]
