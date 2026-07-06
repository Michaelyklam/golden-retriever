from pathlib import Path

from golden_retriever.progress import ExperimentPoint, parse_experiment_points, render_svg, update_readme_section


def test_parse_experiment_points_reads_numeric_metrics():
    markdown = """
| Date | Checkpoint | Commit | Model | Dataset | Eval suite | Final answer found | Recall | Precision | F1 | Trajectory recall | Notes |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 2026-07-06 | base | `abc` | MiniCPM5 | `data/x.jsonl` | localdocs-v0 | 0.083 | 0.083 | 0.028 | 0.042 | n/a | Base. |
| 2026-07-06 | lap1 | `def` | MiniCPM5 + LoRA | `data/y.jsonl` | localdocs-v0 | 0.250 | 0.250 | 0.125 | 0.149 | n/a | Lap. |
"""

    points = parse_experiment_points(markdown)

    assert points == [
        ExperimentPoint("base", "localdocs-v0", 0.083, 0.083, 0.028, 0.042),
        ExperimentPoint("lap1", "localdocs-v0", 0.25, 0.25, 0.125, 0.149),
    ]


def test_render_svg_contains_metric_labels_and_points():
    svg = render_svg([
        ExperimentPoint("base", "localdocs-v0", 0.083, 0.083, 0.028, 0.042),
        ExperimentPoint("lap1", "localdocs-v0", 0.25, 0.25, 0.125, 0.149),
    ])

    assert "Model Progression" in svg
    assert "Recall" in svg
    assert "Precision" in svg
    assert "F1" in svg
    assert "base" in svg
    assert "lap1" in svg
    assert svg.startswith("<svg")


def test_update_readme_section_replaces_existing_block():
    original = "# Title\n\n<!-- metrics-progress:start -->\nold\n<!-- metrics-progress:end -->\n\n## Next\n"
    block = "![Progress](assets/metrics-progression.svg)"

    updated = update_readme_section(original, block)

    assert "old" not in updated
    assert block in updated
    assert updated.count("<!-- metrics-progress:start -->") == 1


def test_update_readme_section_inserts_after_status_when_missing():
    original = "# Title\n\n## Current status\n\ntext\n\n## Quick start\n"

    updated = update_readme_section(original, "graph")

    assert "<!-- metrics-progress:start -->\ngraph\n<!-- metrics-progress:end -->" in updated
    assert updated.index("## Current status") < updated.index("graph") < updated.index("## Quick start")
