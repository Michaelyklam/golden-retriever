from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentPoint:
    checkpoint: str
    eval_suite: str
    final_answer_found: float
    recall: float
    precision: float
    f1: float


NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
START = "<!-- metrics-progress:start -->"
END = "<!-- metrics-progress:end -->"


def _split_row(line: str) -> list[str]:
    return [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]


def _is_number(value: str) -> bool:
    return bool(NUMERIC_RE.match(value.strip()))


def parse_experiment_points(markdown: str) -> list[ExperimentPoint]:
    points: list[ExperimentPoint] = []
    for line in markdown.splitlines():
        if not line.startswith("|") or "---" in line or "Checkpoint" in line:
            continue
        cells = _split_row(line)
        if len(cells) < 10:
            continue
        checkpoint = cells[1]
        eval_suite = cells[5]
        final_answer_found, recall, precision, f1_value = cells[6], cells[7], cells[8], cells[9]
        if not all(_is_number(value) for value in [final_answer_found, recall, precision, f1_value]):
            continue
        points.append(
            ExperimentPoint(
                checkpoint=checkpoint,
                eval_suite=eval_suite,
                final_answer_found=float(final_answer_found),
                recall=float(recall),
                precision=float(precision),
                f1=float(f1_value),
            )
        )
    return points


def _polyline(points: list[tuple[float, float]], color: str) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{coords}" />'


def render_svg(points: list[ExperimentPoint], width: int = 860, height: int = 360) -> str:
    margin_left, margin_top, margin_right, margin_bottom = 70, 42, 28, 82
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    if not points:
        points = [ExperimentPoint("none", "none", 0.0, 0.0, 0.0, 0.0)]

    def x_at(index: int) -> float:
        if len(points) == 1:
            return margin_left + plot_w / 2
        return margin_left + plot_w * index / (len(points) - 1)

    def y_at(value: float) -> float:
        return margin_top + plot_h * (1 - max(0.0, min(1.0, value)))

    recall_points = [(x_at(i), y_at(p.recall)) for i, p in enumerate(points)]
    precision_points = [(x_at(i), y_at(p.precision)) for i, p in enumerate(points)]
    f1_points = [(x_at(i), y_at(p.f1)) for i, p in enumerate(points)]

    rows: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020" />',
        '<text x="24" y="28" fill="#f8fafc" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700">Model Progression</text>',
        '<text x="24" y="50" fill="#94a3b8" font-family="Inter,Arial,sans-serif" font-size="12">Closed-corpus retrieval metrics; target is Context-1 parity on matching benchmark suites as they are added.</text>',
    ]
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_at(tick)
        rows.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width-margin_right}" y2="{y:.1f}" stroke="#1e293b" />')
        rows.append(f'<text x="18" y="{y+4:.1f}" fill="#94a3b8" font-family="Inter,Arial,sans-serif" font-size="11">{tick:.2f}</text>')
    rows.extend([
        _polyline(recall_points, "#38bdf8"),
        _polyline(precision_points, "#f97316"),
        _polyline(f1_points, "#22c55e"),
    ])
    for i, point in enumerate(points):
        x = x_at(i)
        label = html.escape(point.checkpoint.replace("sft-lap", "lap"))
        rows.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top+plot_h}" stroke="#172033" />')
        rows.append(f'<text x="{x:.1f}" y="{height-38}" fill="#cbd5e1" font-family="Inter,Arial,sans-serif" font-size="10" text-anchor="middle" transform="rotate(-18 {x:.1f},{height-38})">{label}</text>')
        for value, color in [(point.recall, "#38bdf8"), (point.precision, "#f97316"), (point.f1, "#22c55e")]:
            rows.append(f'<circle cx="{x:.1f}" cy="{y_at(value):.1f}" r="4" fill="{color}" />')
    legend = [("Recall", "#38bdf8"), ("Precision", "#f97316"), ("F1", "#22c55e")]
    for idx, (label, color) in enumerate(legend):
        x = width - 260 + idx * 85
        rows.append(f'<circle cx="{x}" cy="28" r="5" fill="{color}" />')
        rows.append(f'<text x="{x+10}" y="32" fill="#e2e8f0" font-family="Inter,Arial,sans-serif" font-size="12">{label}</text>')
    rows.append("</svg>")
    return "\n".join(rows) + "\n"


def render_readme_block(svg_path: str = "assets/metrics-progression.svg") -> str:
    return "\n".join(
        [
            "## Progression",
            "",
            f"![Model progression]({svg_path})",
            "",
            "Tracked in [`docs/experiments.md`](docs/experiments.md). The graph shows local proxy smoke/regression benchmarks only; Context-1 parity now tracks full public-suite/tool-loop adapters. See [`docs/context-1-source-inventory.md`](docs/context-1-source-inventory.md) and [`docs/plans/2026-07-06-context-1-full-scale-parity.md`](docs/plans/2026-07-06-context-1-full-scale-parity.md).",
            "",
            "### Context-1 parity benchmark matrix",
            "",
            "| Suite | Status | Notes |",
            "|---|---|---|",
            "| Generated legal / patent / web / finance retrieval | planned full-scale | Chroma recipe uses web, SEC/finance, patents/legal, and Epstein/email task generation with verification, distractors, and chained hops. Local `synthdomains-v1` is now only smoke/regression. |",
            "| Public LongSeal | planned first adapter | Fixed-corpus 512-token chunking makes it the best first comparable public suite. |",
            "| Public Seal-0 | planned | Requires positive URL dataset and browsing/scraping/static snapshot adapter. |",
            "| Public FRAMES | planned | Requires Wikipedia/Serper-style retrieval adapter or static Wikipedia snapshot, plus positive URL coverage filtering. |",
            "| Public HotpotQA | planned sanity suite | Simpler benchmark expected to saturate; useful for harness validation, not sufficient for parity. |",
            "| BrowseComp+ | access TBD | Needs reproducible static corpus/source access; live web BrowseComp is not comparable. |",
        ]
    )


def update_readme_section(readme: str, block: str) -> str:
    wrapped = f"{START}\n{block}\n{END}"
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if pattern.search(readme):
        return pattern.sub(wrapped, readme)
    marker = "\n## Quick start"
    if marker in readme:
        return readme.replace(marker, "\n" + wrapped + "\n" + marker, 1)
    return readme.rstrip() + "\n\n" + wrapped + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render README metrics progression graph.")
    parser.add_argument("--experiments", default="docs/experiments.md")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--svg", default="assets/metrics-progression.svg")
    args = parser.parse_args()

    experiments_path = Path(args.experiments)
    readme_path = Path(args.readme)
    svg_path = Path(args.svg)

    points = parse_experiment_points(experiments_path.read_text(encoding="utf-8"))
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(render_svg(points), encoding="utf-8")
    readme_path.write_text(update_readme_section(readme_path.read_text(encoding="utf-8"), render_readme_block(svg_path.as_posix())), encoding="utf-8")
    print(json.dumps({"points": len(points), "svg": svg_path.as_posix(), "readme": readme_path.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
