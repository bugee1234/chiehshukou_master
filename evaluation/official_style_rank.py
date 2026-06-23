from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.compare_task1_1_leaderboard import METRICS, LOWER_IS_BETTER, load_leaderboard, load_overall

SCORES = list(METRICS)
RELEVANCE = ["ROUGE", "BLEU", "METEOR", "BERTScore"]
READABILITY = ["FKGL", "DCRS", "CLI", "LENS"]
FACTUALITY = ["AlignScore", "SummaC"]
LOWER_IS_BETTER_LIST = [m for m in METRICS if m in LOWER_IS_BETTER]

CAUTION = (
    "Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. "
    "Final ranks are computed by min-max normalizing across the 16 published teams plus "
    "our row(s), following evaluation/rank.py; not the competition's published official_rank."
)


def min_max_normalize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result_df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val == min_val:
            result_df[col] = 0.0
        else:
            result_df[col] = (df[col] - min_val) / (max_val - min_val)
    return result_df


def _leaderboard_rows(leaderboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in leaderboard:
        row = {
            "team": entry["team"],
            "variant": "",
            "published_official_rank": entry.get("official_rank"),
        }
        for metric in METRICS:
            row[metric] = float(entry[metric])
        rows.append(row)
    return rows


def build_rank_table(
    *,
    leaderboard: list[dict[str, Any]],
    generated: dict[str, float] | None = None,
    rewritten: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows = _leaderboard_rows(leaderboard)
    if generated is not None:
        rows.append(
            {
                "team": "thesis_laysumm (generated)",
                "variant": "generated",
                "published_official_rank": None,
                **{metric: float(generated[metric]) for metric in METRICS},
            }
        )
    if rewritten is not None:
        rows.append(
            {
                "team": "thesis_laysumm (rewritten)",
                "variant": "rewritten",
                "published_official_rank": None,
                **{metric: float(rewritten[metric]) for metric in METRICS},
            }
        )

    raw_df = pd.DataFrame(rows)
    for metric in METRICS:
        raw_df[f"raw_{metric}"] = raw_df[metric]

    normalized_df = min_max_normalize(raw_df, SCORES)
    for col in LOWER_IS_BETTER_LIST:
        normalized_df[col] = 1 - normalized_df[col]
    for metric in METRICS:
        normalized_df[f"norm_{metric}"] = normalized_df[metric]

    normalized_df["Relevance"] = normalized_df[RELEVANCE].mean(axis=1)
    normalized_df["Readability"] = normalized_df[READABILITY].mean(axis=1)
    normalized_df["Factuality"] = normalized_df[FACTUALITY].mean(axis=1)
    normalized_df["Final"] = normalized_df[["Relevance", "Readability", "Factuality"]].mean(axis=1)
    normalized_df = normalized_df.sort_values("Final", ascending=False).reset_index(drop=True)
    normalized_df["final_rank"] = normalized_df.index + 1
    return normalized_df


def _our_system_summary(table: pd.DataFrame, variant: str) -> dict[str, Any]:
    row = table[table["variant"] == variant].iloc[0]
    pool_size = len(table)
    return {
        "team": row["team"],
        "final_score": round(float(row["Final"]), 4),
        "final_rank_among_pool": int(row["final_rank"]),
        "pool_size": pool_size,
        "relevance": round(float(row["Relevance"]), 4),
        "readability": round(float(row["Readability"]), 4),
        "factuality": round(float(row["Factuality"]), 4),
        "raw_scores": {
            metric: round(float(row[f"raw_{metric}"]), 4) for metric in METRICS
        },
    }


def _table_records(table: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        records.append(
            {
                "team": row["team"],
                "variant": row["variant"] or None,
                "published_official_rank": (
                    int(row["published_official_rank"])
                    if pd.notna(row["published_official_rank"])
                    else None
                ),
                "final_rank": int(row["final_rank"]),
                "raw": {metric: round(float(row[f"raw_{metric}"]), 4) for metric in METRICS},
                "normalized": {
                    metric: round(float(row[f"norm_{metric}"]), 4) for metric in METRICS
                },
                "Relevance": round(float(row["Relevance"]), 4),
                "Readability": round(float(row["Readability"]), 4),
                "Factuality": round(float(row["Factuality"]), 4),
                "Final": round(float(row["Final"]), 4),
            }
        )
    return records


def write_official_style_rank(
    *,
    metrics_dir: Path,
    leaderboard_path: Path,
    include_generated: bool = True,
    include_rewritten: bool = True,
    caution: str | None = None,
) -> dict[str, Any]:
    leaderboard = load_leaderboard(leaderboard_path)
    generated = load_overall(metrics_dir / "generated_scores.json") if include_generated else None
    rewritten = load_overall(metrics_dir / "rewritten_scores.json") if include_rewritten else None

    table = build_rank_table(leaderboard=leaderboard, generated=generated, rewritten=rewritten)
    pool_size = len(table)
    our_systems: dict[str, Any] = {}
    if generated is not None:
        our_systems["generated"] = _our_system_summary(table, "generated")
    if rewritten is not None:
        our_systems["rewritten"] = _our_system_summary(table, "rewritten")

    payload: dict[str, Any] = {
        "caution": caution or CAUTION,
        "pool_size": pool_size,
        "our_systems": our_systems,
        "full_table": _table_records(table),
    }

    json_path = metrics_dir / "official_style_rank.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = metrics_dir / "official_style_rank.csv"
    csv_fields = [
        "final_rank",
        "team",
        "variant",
        "published_official_rank",
        *[f"raw_{m}" for m in METRICS],
        *[f"norm_{m}" for m in METRICS],
        "Relevance",
        "Readability",
        "Factuality",
        "Final",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for _, row in table.iterrows():
            out = {
                "final_rank": int(row["final_rank"]),
                "team": row["team"],
                "variant": row["variant"],
                "published_official_rank": (
                    int(row["published_official_rank"])
                    if pd.notna(row["published_official_rank"])
                    else ""
                ),
                "Relevance": f"{float(row['Relevance']):.4f}",
                "Readability": f"{float(row['Readability']):.4f}",
                "Factuality": f"{float(row['Factuality']):.4f}",
                "Final": f"{float(row['Final']):.4f}",
            }
            for metric in METRICS:
                out[f"raw_{metric}"] = f"{float(row[f'raw_{metric}']):.3f}"
                out[f"norm_{metric}"] = f"{float(row[f'norm_{metric}']):.4f}"
            writer.writerow(out)

    return payload


def append_official_rank_to_markdown(
    *,
    metrics_dir: Path,
    payload: dict[str, Any],
) -> None:
    md_path = metrics_dir / "leaderboard_comparison.md"
    if not md_path.exists():
        return

    lines = [
        "",
        "## Official-style Final rank",
        "",
        f"> {CAUTION}",
        "",
        "| Variant | Final rank | Final score | Relevance | Readability | Factuality |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, summary in payload["our_systems"].items():
        lines.append(
            f"| {variant} | {summary['final_rank_among_pool']}/{summary['pool_size']} | "
            f"{summary['final_score']:.4f} | {summary['relevance']:.4f} | "
            f"{summary['readability']:.4f} | {summary['factuality']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.",
            "",
        ]
    )
    existing = md_path.read_text(encoding="utf-8")
    marker = "## Official-style Final rank"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip()
    md_path.write_text(existing + "\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Compute official-style Final rank for thesis_laysumm scores.")
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--leaderboard", type=Path, default=root / "evaluation" / "task1_1_leaderboard.csv")
    args = parser.parse_args()

    payload = write_official_style_rank(metrics_dir=args.metrics_dir, leaderboard_path=args.leaderboard)
    for variant, summary in payload["our_systems"].items():
        print(
            f"{variant}: Final rank {summary['final_rank_among_pool']}/{summary['pool_size']} "
            f"(score={summary['final_score']:.4f})"
        )
    print(f"Wrote {args.metrics_dir / 'official_style_rank.json'}")


if __name__ == "__main__":
    main()
