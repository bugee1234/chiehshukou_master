from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results" / "laysumm_pipeline" / "runs" / "pilot_n20_v10_factuality_chase"
OUT_DIR = ROOT / "results" / "laysumm_pipeline" / "analysis_v10_three_model"
MODELS = [
    "gpt41_mini",
    "gemini3_flash_preview_minimal",
    "gemini25_flash_non_thinking",
]


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for model in MODELS:
        official = load_json(RUN_ROOT / model / "official_style_rank.json")["our_systems"]["rewritten"]
        sent_meta = load_json(RUN_ROOT / model / "sentence_alignscore_rewritten_metadata.json")
        low_rate = sent_meta["low_sentence_count"] / sent_meta["sentence_count"]
        summary_rows.append(
            {
                "model": model,
                "final_score": official["final_score"],
                "relevance": official["relevance"],
                "readability": official["readability"],
                "factuality": official["factuality"],
                "overall_alignscore": official["raw_scores"]["AlignScore"],
                "overall_summac": official["raw_scores"]["SummaC"],
                "sentence_count": sent_meta["sentence_count"],
                "sentence_mean_alignscore": sent_meta["mean_sentence_alignscore"],
                "sentence_median_alignscore": sent_meta["median_sentence_alignscore"],
                "low_sentence_count": sent_meta["low_sentence_count"],
                "low_sentence_rate": round(low_rate, 4),
                "min_sentence_alignscore": sent_meta["min_sentence_alignscore"],
            }
        )
    write_csv(OUT_DIR / "v10_three_model_summary.csv", summary_rows, list(summary_rows[0]))

    with (OUT_DIR / "sentence_alignscore_all_models.csv").open("w", encoding="utf-8-sig", newline="") as fout:
        writer = None
        for model in MODELS:
            with (RUN_ROOT / model / "sentence_alignscore_rewritten.csv").open(
                encoding="utf-8-sig", newline=""
            ) as f:
                reader = csv.DictReader(f)
                if writer is None:
                    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)

    article_rows: list[dict] = []
    for model in MODELS:
        with (RUN_ROOT / model / "sentence_alignscore_rewritten_articles.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                row["model_key"] = model
                article_rows.append(row)

    article_fields = [
        "article_id",
        "source_dataset",
        "original_index",
        "model_key",
        "sentence_count",
        "mean_sentence_alignscore",
        "min_sentence_alignscore",
        "low_sentence_count",
    ]
    write_csv(
        OUT_DIR / "article_alignscore_all_models.csv",
        [{key: row.get(key) for key in article_fields} for row in article_rows],
        article_fields,
    )

    by_article: dict[str, list[dict]] = {}
    for row in article_rows:
        by_article.setdefault(row["article_id"], []).append(row)

    spread_rows: list[dict] = []
    for article_id, rows in by_article.items():
        if len(rows) != len(MODELS):
            continue
        means = [float(row["mean_sentence_alignscore"]) for row in rows]
        lows = [int(row["low_sentence_count"]) for row in rows]
        sorted_rows = sorted(rows, key=lambda row: row["model_key"])
        spread_rows.append(
            {
                "article_id": article_id,
                "source_dataset": rows[0].get("source_dataset"),
                "original_index": rows[0].get("original_index"),
                "mean_alignscore_range": round(max(means) - min(means), 6),
                "low_sentence_count_range": max(lows) - min(lows),
                "means_by_model": "; ".join(
                    f"{row['model_key']}={row['mean_sentence_alignscore']}" for row in sorted_rows
                ),
                "low_by_model": "; ".join(f"{row['model_key']}={row['low_sentence_count']}" for row in sorted_rows),
            }
        )
    spread_rows.sort(key=lambda row: (row["low_sentence_count_range"], row["mean_alignscore_range"]), reverse=True)
    write_csv(OUT_DIR / "article_cross_model_spread.csv", spread_rows, list(spread_rows[0]))

    all_runs: list[dict] = []
    for path in (ROOT / "results" / "laysumm_pipeline" / "runs").rglob("official_style_rank.json"):
        data = load_json(path)
        run_name = path.parts[-3]
        model = path.parts[-2]
        for variant, scores in data.get("our_systems", {}).items():
            raw = scores.get("raw_scores", {})
            all_runs.append(
                {
                    "run": run_name,
                    "model": model,
                    "variant": variant,
                    "final_score": scores.get("final_score"),
                    "relevance": scores.get("relevance"),
                    "readability": scores.get("readability"),
                    "factuality": scores.get("factuality"),
                    "alignscore": raw.get("AlignScore"),
                    "summac": raw.get("SummaC"),
                }
            )
    all_runs.sort(key=lambda row: row["final_score"] or -1, reverse=True)
    write_csv(OUT_DIR / "all_runs_official_style_top.csv", all_runs, list(all_runs[0]))

    report = [
        "# V10 three-model analysis",
        "",
        "## Verdict",
        (
            "Among existing n=20 official-style pilot rows, v10 "
            "gemini25_flash_non_thinking rewritten is the highest final score: 0.6921. "
            "However, v10 is not uniformly best across all three models: GPT-4.1 mini "
            "rewritten is 0.6333 and Gemini 3 Flash preview rewritten is 0.6075."
        ),
        "",
        "## V10 rewritten summary",
        (
            "| model | final | relevance | readability | factuality | doc AlignScore | "
            "SummaC | sent mean AlignScore | low sent <0.65 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report.append(
            (
                f"| {row['model']} | {row['final_score']:.4f} | {row['relevance']:.4f} | "
                f"{row['readability']:.4f} | {row['factuality']:.4f} | "
                f"{row['overall_alignscore']:.4f} | {row['overall_summac']:.4f} | "
                f"{row['sentence_mean_alignscore']:.4f} | {row['low_sentence_count']}/"
                f"{row['sentence_count']} ({row['low_sentence_rate']:.1%}) |"
            )
        )

    report.extend(
        [
            "",
            "## Biggest cross-model article spread",
            "| article | dataset | mean AlignScore range | low-sentence range | means by model | low by model |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for row in spread_rows[:10]:
        report.append(
            (
                f"| {row['article_id']} | {row['source_dataset']} | "
                f"{row['mean_alignscore_range']:.4f} | {row['low_sentence_count_range']} | "
                f"{row['means_by_model']} | {row['low_by_model']} |"
            )
        )

    report.extend(
        [
            "",
            "## Files",
            "- v10_three_model_summary.csv",
            "- sentence_alignscore_all_models.csv",
            "- article_alignscore_all_models.csv",
            "- article_cross_model_spread.csv",
            "- all_runs_official_style_top.csv",
        ]
    )
    (OUT_DIR / "v10_analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(OUT_DIR)


if __name__ == "__main__":
    main()
