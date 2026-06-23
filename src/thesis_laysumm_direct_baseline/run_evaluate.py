from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.thesis_laysumm.paths import run_results_dir as pipeline_results_dir
from src.thesis_laysumm_direct_baseline.paths import (
    inputs_path,
    references_path,
    run_results_dir,
    summaries_path,
)
from src.utils import load_json, load_jsonl, save_json


METRICS = ["ROUGE", "BLEU", "METEOR", "BERTScore", "FKGL", "DCRS", "CLI", "LENS", "AlignScore", "SummaC"]
DEFAULT_LEADERBOARD = ROOT_DIR / "evaluation" / "task1_1_leaderboard.csv"


def _require_eval_environment() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("Run direct-baseline evaluation with .venv-eval") from exc


def _write_scores_txt(scores: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{key}: {value}" for key, value in scores.items()) + "\n", encoding="utf-8")


def _aligned_rows(run_name: str, model_key: str) -> list[dict[str, Any]]:
    inputs = {str(row["id"]): row for row in load_jsonl(inputs_path(run_name))}
    refs = {str(row["id"]): row for row in load_jsonl(references_path(run_name))}
    summaries = load_jsonl(summaries_path(run_name, model_key))
    summary_ids = {str(row["article_id"]) for row in summaries}
    if summary_ids != set(inputs) or summary_ids != set(refs):
        raise ValueError("Direct summaries, inputs, and references are not aligned")
    rows = []
    for summary in summaries:
        aid = str(summary["article_id"])
        rows.append(
            {
                "article_id": aid,
                "source_dataset": str(inputs[aid]["source_dataset"]),
                "original_index": int(inputs[aid]["original_index"]),
                "prediction": str(summary["generated_summary"]).strip(),
                "document": str(refs[aid]["document"]),
                "reference": str(refs[aid]["expert_summary"]),
            }
        )
    return sorted(rows, key=lambda row: (row["source_dataset"], row["original_index"]))


def _evaluate(run_name: str, model_key: str) -> dict[str, Any]:
    from evaluation.evaluation_final import evaluate_all

    rows = _aligned_rows(run_name, model_key)
    source_scores: dict[str, dict[str, float]] = {}
    for source in ("PLOS", "eLife"):
        source_rows = [row for row in rows if row["source_dataset"] == source]
        if not source_rows:
            raise ValueError(f"No {source} rows in direct-baseline run")
        preds = [row["prediction"] for row in source_rows]
        refs = [{"document": row["document"], "reference": row["reference"]} for row in source_rows]
        docs = [row["document"] for row in source_rows]
        source_scores[source] = evaluate_all(preds, refs, "lay_summ", summac_docs=docs)
    overall = {
        metric: float(np.mean([source_scores["PLOS"][metric], source_scores["eLife"][metric]]))
        for metric in source_scores["PLOS"]
    }
    return {
        "variant": "direct_generated",
        "summac_document": "original",
        "article_count": len(rows),
        "PLOS": source_scores["PLOS"],
        "eLife": source_scores["eLife"],
        "overall": overall,
        "per_article": rows,
    }


def _compare_pipeline(*, direct: dict[str, Any], pipeline_run: str, model_key: str, out_dir: Path) -> None:
    pipeline_dir = pipeline_results_dir(pipeline_run) / model_key
    variants: dict[str, dict[str, Any]] = {}
    for name in ("generated", "rewritten"):
        path = pipeline_dir / f"{name}_scores.json"
        if path.exists():
            variants[name] = load_json(path)
    if not variants:
        raise FileNotFoundError(f"No pipeline scores found under {pipeline_dir}")

    payload: dict[str, Any] = {
        "caution": "All differences are direct baseline minus pipeline variant on the same validation articles.",
        "pipeline_run": pipeline_run,
        "model_key": model_key,
        "direct": direct["overall"],
        "pipeline": {},
    }
    csv_rows: list[dict[str, Any]] = []
    for variant, scores in variants.items():
        if int(scores.get("article_count", -1)) != int(direct["article_count"]):
            raise ValueError(
                f"Article-count mismatch: direct={direct['article_count']} "
                f"pipeline_{variant}={scores.get('article_count')}"
            )
        delta = {metric: float(direct["overall"][metric]) - float(scores["overall"][metric]) for metric in METRICS}
        payload["pipeline"][variant] = {"scores": scores["overall"], "direct_minus_pipeline": delta}
        for metric in METRICS:
            csv_rows.append(
                {
                    "pipeline_variant": variant,
                    "metric": metric,
                    "direct": direct["overall"][metric],
                    "pipeline": scores["overall"][metric],
                    "direct_minus_pipeline": delta[metric],
                }
            )
    save_json(payload, out_dir / "direct_vs_pipeline.json")
    with (out_dir / "direct_vs_pipeline.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def _write_leaderboard_files(*, scores_path: Path, leaderboard_path: Path) -> None:
    from evaluation.compare_task1_1_leaderboard import (
        METRICS as LEADERBOARD_METRICS,
        compare_variant,
        load_leaderboard,
        load_overall,
        write_csv,
    )
    from evaluation.official_style_rank import write_official_style_rank

    scores = load_overall(scores_path)
    score_payload = load_json(scores_path)
    article_count = int(score_payload.get("article_count", 0))
    leaderboard = load_leaderboard(leaderboard_path)
    rows = compare_variant("direct_generated", scores, leaderboard)
    out_dir = scores_path.parent
    caution = (
        f"Direct zero-shot validation sample ({article_count} articles) versus the published "
        "official-test leaderboard; metric positions and normalized Final rank are descriptive only."
    )
    write_csv(rows, out_dir / "leaderboard_comparison.csv")
    save_json({"caution": caution, "comparisons": {"direct_generated": rows}}, out_dir / "leaderboard_comparison.json")
    by_metric = {row["metric"]: row for row in rows}
    lines = [
        "# Direct Zero-shot BioLaySumm Leaderboard Comparison",
        "",
        f"> {caution}",
        "",
        "| Metric | Direct score | Descriptive position | Nearest published team |",
        "|---|---:|---:|---|",
    ]
    for metric in LEADERBOARD_METRICS:
        row = by_metric[metric]
        lines.append(
            f"| {metric} | {scores[metric]:.3f} | {row['metric_rank_among_17']}/17 | "
            f"{row['nearest_team']} ({row['nearest_team_score']:.3f}) |"
        )
    lines.extend(["", "These are not official competition ranks.", ""])
    (out_dir / "leaderboard_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    write_official_style_rank(
        metrics_dir=out_dir,
        leaderboard_path=leaderboard_path,
        include_generated=True,
        include_rewritten=False,
        caution=caution,
    )


def run_evaluation(
    *,
    run_name: str,
    model_key: str,
    pipeline_run: str | None,
    save_per_article: bool,
    leaderboard_path: Path,
) -> None:
    _require_eval_environment()
    result = _evaluate(run_name, model_key)
    out_dir = run_results_dir(run_name) / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json({key: value for key, value in result.items() if key != "per_article"}, out_dir / "generated_scores.json")
    _write_scores_txt(result["overall"], out_dir / "generated_scores.txt")
    if save_per_article:
        with (out_dir / "generated_per_article.jsonl").open("w", encoding="utf-8") as handle:
            for row in result["per_article"]:
                handle.write(
                    json.dumps(
                        {
                            "article_id": row["article_id"],
                            "source_dataset": row["source_dataset"],
                            "original_index": row["original_index"],
                            "prediction": row["prediction"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    if pipeline_run:
        _compare_pipeline(direct=result, pipeline_run=pipeline_run, model_key=model_key, out_dir=out_dir)
    _write_leaderboard_files(scores_path=out_dir / "generated_scores.json", leaderboard_path=leaderboard_path)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "model_key": model_key,
            "design": "direct_zero_shot_full_article",
            "article_count": result["article_count"],
            "pipeline_comparison_run": pipeline_run,
        },
        out_dir / "evaluation_metadata.json",
    )
    print(f"[direct evaluate] {model_key} articles={result['article_count']} -> {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate direct zero-shot summaries with official BioLaySumm metrics.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--compare-pipeline-run")
    parser.add_argument("--save-per-article", action="store_true")
    parser.add_argument("--leaderboard", type=Path, default=DEFAULT_LEADERBOARD)
    parser.add_argument(
        "--leaderboard-only",
        action="store_true",
        help="Create leaderboard files from existing generated_scores.json without rerunning metrics.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.leaderboard_only:
        scores_path = run_results_dir(args.run_name) / args.model_key / "generated_scores.json"
        if not scores_path.exists():
            raise FileNotFoundError(f"Missing existing direct scores: {scores_path}")
        _write_leaderboard_files(scores_path=scores_path, leaderboard_path=args.leaderboard)
        print(f"[direct leaderboard] {args.model_key} -> {scores_path.parent}")
        return
    run_evaluation(
        run_name=args.run_name,
        model_key=args.model_key,
        pipeline_run=args.compare_pipeline_run,
        save_per_article=args.save_per_article,
        leaderboard_path=args.leaderboard,
    )


if __name__ == "__main__":
    main()
