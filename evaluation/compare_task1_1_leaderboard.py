from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = [
    "ROUGE",
    "BLEU",
    "METEOR",
    "BERTScore",
    "FKGL",
    "DCRS",
    "CLI",
    "LENS",
    "AlignScore",
    "SummaC",
]
LOWER_IS_BETTER = {"FKGL", "DCRS", "CLI"}


def load_leaderboard(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["official_rank"] = int(row["official_rank"])
        for metric in METRICS:
            row[metric] = float(row[metric])
    return rows


def load_overall(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    scores = data.get("overall", data)
    return {metric: float(scores[metric]) for metric in METRICS}


def metric_rank(value: float, metric: str, leaderboard: list[dict[str, Any]]) -> int:
    if metric in LOWER_IS_BETTER:
        return 1 + sum(float(row[metric]) < value for row in leaderboard)
    return 1 + sum(float(row[metric]) > value for row in leaderboard)


def nearest_team(value: float, metric: str, leaderboard: list[dict[str, Any]]) -> dict[str, Any]:
    return min(leaderboard, key=lambda row: abs(float(row[metric]) - value))


def compare_variant(
    variant: str,
    scores: dict[str, float],
    leaderboard: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for metric in METRICS:
        nearest = nearest_team(scores[metric], metric, leaderboard)
        rows.append(
            {
                "variant": variant,
                "metric": metric,
                "value": scores[metric],
                "direction": "lower" if metric in LOWER_IS_BETTER else "higher",
                "metric_rank_among_17": metric_rank(scores[metric], metric, leaderboard),
                "nearest_team": nearest["team"],
                "nearest_team_score": nearest[metric],
                "difference_from_nearest": scores[metric] - float(nearest[metric]),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in ("value", "nearest_team_score", "difference_from_nearest"):
                formatted[key] = f"{float(row[key]):.3f}"
            writer.writerow(formatted)


def write_markdown(
    comparisons: dict[str, list[dict[str, Any]]],
    initial: dict[str, float],
    rewritten: dict[str, float],
    path: Path,
) -> None:
    lines = [
        "# BioLaySumm Task 1.1 Leaderboard Comparison",
        "",
        "> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.",
        "",
        "| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    initial_rows = {row["metric"]: row for row in comparisons["initial"]}
    rewritten_rows = {row["metric"]: row for row in comparisons["rewritten"]}
    for metric in METRICS:
        lines.append(
            f"| {metric} | {initial[metric]:.3f} | {initial_rows[metric]['metric_rank_among_17']}/17 | "
            f"{rewritten[metric]:.3f} | {rewritten_rows[metric]['metric_rank_among_17']}/17 | "
            f"{rewritten[metric] - initial[metric]:+.3f} |"
        )
    lines.extend(
        [
            "",
            "Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Compare experiment metrics with the BioLaySumm Task 1.1 leaderboard.")
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--leaderboard", type=Path, default=root / "evaluation" / "task1_1_leaderboard.csv")
    args = parser.parse_args()

    leaderboard = load_leaderboard(args.leaderboard)
    initial = load_overall(args.metrics_dir / "initial_scores.json")
    rewritten = load_overall(args.metrics_dir / "rewritten_scores.json")
    comparisons = {
        "initial": compare_variant("initial", initial, leaderboard),
        "rewritten": compare_variant("rewritten", rewritten, leaderboard),
    }
    flat_rows = comparisons["initial"] + comparisons["rewritten"]

    write_csv(flat_rows, args.metrics_dir / "leaderboard_comparison.csv")
    (args.metrics_dir / "leaderboard_comparison.json").write_text(
        json.dumps(
            {
                "caution": "20-article validation pilot versus published official-test leaderboard; positions are descriptive only.",
                "comparisons": comparisons,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(comparisons, initial, rewritten, args.metrics_dir / "leaderboard_comparison.md")
    print(f"Wrote leaderboard comparison to {args.metrics_dir}")


if __name__ == "__main__":
    main()
