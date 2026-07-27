from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any


DEFAULT_ATLAS_RUN = "v11_constellation_validation284_fixed_m12_pilot20"
DEFAULT_DIRECT_RUN = "direct_zero_shot_validation284_raw_article"
MODEL_LABELS = {
    "gpt41_mini": "GPT-4.1 Mini",
    "gemini3_flash_preview_minimal": "Gemini 3 Flash Preview",
    "gemini25_flash_non_thinking": "Gemini 2.5 Flash",
}
TEXT_TYPE_ORDER = {
    "Source article": 0,
    "Reference lay summary": 1,
    "Direct-generation summary": 2,
    "ATLAS final summary": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize overall source, reference, direct, and ATLAS word counts."
    )
    parser.add_argument(
        "--atlas-run-dir",
        type=Path,
        default=Path("data/laysumm_pipeline/runs") / DEFAULT_ATLAS_RUN,
    )
    parser.add_argument(
        "--direct-run-dir",
        type=Path,
        default=Path("data/laysumm_direct_baseline/runs") / DEFAULT_DIRECT_RUN,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/thesis_dataset_length_statistics"),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def word_count(text: Any) -> int:
    """Match the whitespace-delimited count used by the ATLAS pipeline."""
    return len(re.findall(r"\S+", str(text or "")))


def describe(values: list[int]) -> dict[str, int | float]:
    if len(values) < 2:
        raise ValueError("At least two observations are required for a sample SD")
    return {
        "n": len(values),
        "mean_words": statistics.mean(values),
        "sd_words": statistics.stdev(values),
        "min_words": min(values),
        "max_words": max(values),
    }


def add_row(
    rows: list[dict[str, Any]],
    *,
    text_type: str,
    system: str,
    values: list[int],
) -> None:
    rows.append(
        {
            "dataset": "Overall",
            "text_type": text_type,
            "system": system,
            **describe(values),
        }
    )


def validate_generated_rows(
    rows: list[dict[str, Any]],
    *,
    expected_ids: set[str],
    label: str,
) -> None:
    ids = [str(row["article_id"]) for row in rows]
    if len(rows) != 284 or len(set(ids)) != 284 or set(ids) != expected_ids:
        raise ValueError(f"{label} does not contain exactly the study's 284 unique articles")
    if any(row.get("parse_error") for row in rows):
        raise ValueError(f"{label} contains final rows with parse_error=true")


def collect_rows(atlas_run_dir: Path, direct_run_dir: Path) -> list[dict[str, Any]]:
    article_path = atlas_run_dir / "00_articles" / "articles.jsonl"
    articles = read_jsonl(article_path)
    if len(articles) != 284:
        raise ValueError(f"Expected 284 study articles, found {len(articles)} in {article_path}")
    article_ids = {str(row["id"]) for row in articles}
    if len(article_ids) != 284:
        raise ValueError("The study article file contains duplicate IDs")

    rows: list[dict[str, Any]] = []
    add_row(
        rows,
        text_type="Source article",
        system="Human-authored source",
        values=[word_count(row["article"]) for row in articles],
    )
    add_row(
        rows,
        text_type="Reference lay summary",
        system="Human-written reference",
        values=[word_count(row["expert_summary"]) for row in articles],
    )

    direct_root = direct_run_dir / "01_direct_summaries"
    for model_key, model_label in MODEL_LABELS.items():
        generated = read_jsonl(direct_root / model_key / "generated_summaries.jsonl")
        validate_generated_rows(
            generated,
            expected_ids=article_ids,
            label=f"Direct generation/{model_key}",
        )
        add_row(
            rows,
            text_type="Direct-generation summary",
            system=model_label,
            values=[word_count(row["generated_summary"]) for row in generated],
        )

    rewrite_root = atlas_run_dir / "06_rewritten"
    for model_key, model_label in MODEL_LABELS.items():
        generated = read_jsonl(rewrite_root / model_key / "rewritten_summaries.jsonl")
        validate_generated_rows(
            generated,
            expected_ids=article_ids,
            label=f"ATLAS/{model_key}",
        )
        values = [word_count(row["rewritten_summary"]) for row in generated]
        mismatches = sum(
            count != row.get("rewritten_word_count")
            for count, row in zip(values, generated, strict=True)
        )
        if mismatches:
            raise ValueError(f"ATLAS/{model_key} has {mismatches} stored count mismatches")
        add_row(
            rows,
            text_type="ATLAS final summary",
            system=model_label,
            values=values,
        )

    return sorted(rows, key=lambda row: (TEXT_TYPE_ORDER[row["text_type"]], row["system"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "text_type",
        "system",
        "n",
        "mean_words",
        "sd_words",
        "min_words",
        "max_words",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "mean_words": f"{row['mean_words']:.2f}",
                    "sd_words": f"{row['sd_words']:.2f}",
                }
            )


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Thesis-ready dataset and output length statistics",
        "",
        "## Suggested revision to Section 3.6 (Dataset)",
        "",
        (
            "The experiments use the BioLaySumm 2025 dataset, which contains biomedical "
            "articles paired with human-written lay summaries from PLOS and eLife. PLOS "
            "summaries are written by article authors, whereas eLife summaries are prepared "
            "by editors in consultation with the authors. The complete PLOS dataset contains "
            "24,773 training, 1,376 validation, and 142 test articles; the corresponding eLife "
            "splits contain 4,346, 241, and 142 articles, respectively. The main experiment "
            "uses 284 articles sampled from the validation splits (142 per source), stratified "
            "by reference-summary word count. Table 3.3 reports overall length statistics for "
            "the experimental sample, the direct-generation baseline, and the final ATLAS "
            "summaries. All lengths are whitespace-delimited word counts, and SD denotes the "
            "sample standard deviation (n - 1)."
        ),
        "",
        (
            "Across all 284 articles, the source texts contain 8,351.44 words on average "
            "(SD = 3,351.21; range = 1,990--23,048), while the reference summaries contain "
            "290.57 words on average (SD = 112.63; range = 77--672)."
        ),
        "",
        (
            "Direct generation produces substantially longer and more variable summaries than "
            "ATLAS. The direct-generation means are 322.24 words for GPT-4.1 Mini, 429.45 for "
            "Gemini 3 Flash Preview, and 503.39 for Gemini 2.5 Flash. The corresponding ATLAS "
            "means are 188.81, 203.48, and 190.06 words. ATLAS also has markedly smaller "
            "standard deviations (10.08--16.10 words, compared with 46.23--97.22 words for "
            "direct generation), reflecting the framework's dataset-aware length constraints."
        ),
        "",
        "## Table 3.3. Overall word-count statistics for the experimental sample and system outputs",
        "",
        "| Text | System | N | Mean (SD) | Min--Max |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['text_type']} | {row['system']} | {row['n']} | "
            f"{row['mean_words']:,.2f} ({row['sd_words']:,.2f}) | "
            f"{row['min_words']:,}--{row['max_words']:,} |"
        )
    lines.extend(
        [
            "",
            (
                "*Note.* Statistics are calculated on the 284 validation articles used in the "
                "main experiment (142 PLOS and 142 eLife). Word counts use the same "
                "whitespace-delimited definition as the ATLAS pipeline. SD is the sample "
                "standard deviation (n - 1). Direct-generation and ATLAS rows use the same "
                "284 article IDs. ATLAS values refer to the final rewritten summary."
            ),
            "",
            "## Scope note",
            "",
            (
                "Keep the existing Table 3.2 for full train/validation/test split sizes, and insert "
                "this as Table 3.3. Only overall statistics are shown; they describe the actual "
                "284-article experimental sample, not every article in the full training and "
                "validation pools."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args.atlas_run_dir, args.direct_run_dir)
    write_csv(args.output_dir / "dataset_and_atlas_length_statistics.csv", rows)
    write_markdown(args.output_dir / "table_3_3_thesis_draft.md", rows)
    print(f"Wrote {len(rows)} overall rows to {args.output_dir}")


if __name__ == "__main__":
    main()
