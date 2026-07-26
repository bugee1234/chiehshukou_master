from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.thesis_laysumm.paths import run_data_dir, run_results_dir

DEFAULT_MODEL_KEY = "gemini3_flash_preview_minimal"
VARIANTS = {
    "generated": {
        "summary_key": "generated_summary",
        "summary_path": lambda run_name, model_key: run_data_dir(run_name)
        / "04_summaries"
        / model_key
        / "generated_summaries.jsonl",
    },
    "rewritten": {
        "summary_key": "rewritten_summary",
        "summary_path": lambda run_name, model_key: run_data_dir(run_name)
        / "06_rewritten"
        / model_key
        / "rewritten_summaries.jsonl",
    },
}


def load_jsonl(path: str | Path) -> list[Any]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_json(data: Any, path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_articles(run_name: str) -> list[dict[str, Any]]:
    path = run_data_dir(run_name) / "00_articles" / "articles.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing articles file: {path}")
    return load_jsonl(path)


def _require_eval_environment() -> None:
    """Official metrics need torch/CUDA; use the project's evaluation venv, not the local-LLM venv."""
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Evaluation dependencies are not available in the active Python environment.\n"
            "Use the evaluation venv on this machine:\n"
            "  deactivate\n"
            "  .\\.venv\\Scripts\\Activate.ps1\n"
            "  $env:NLTK_DATA=\"$PWD\\.nltk_data\"\n"
            "  $env:HF_HOME=\"C:\\hf_cache\"\n"
            "  .\\.venv\\Scripts\\python.exe -m src.thesis_laysumm.run_evaluate ...\n"
            "See evaluation/LOCAL_TEST_MACHINE_SETUP.md for details."
        ) from exc


def _results_dir(run_name: str, model_key: str) -> Path:
    path = run_results_dir(run_name) / model_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_summary_rows(run_name: str, model_key: str, variant: str) -> list[dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}")
    path = VARIANTS[variant]["summary_path"](run_name, model_key)
    if not path.exists():
        raise FileNotFoundError(f"Missing {variant} summaries: {path}")
    rows = load_jsonl(path)
    bad = [
        str(r.get("article_id"))
        for r in rows
        if VARIANTS[variant]["summary_key"] not in r
        or not str(r.get(VARIANTS[variant]["summary_key"], "")).strip()
        or r.get("parse_error")
    ]
    if bad:
        raise ValueError(f"{variant} summaries missing/empty/parse_error for articles: {bad}")
    return rows


def _build_aligned_pairs(
    *,
    articles: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    summary_key: str,
) -> list[dict[str, Any]]:
    articles_by_id = {str(a["id"]): a for a in articles}
    pairs: list[dict[str, Any]] = []
    for summary in sorted(
        summaries,
        key=lambda row: (str(row.get("source_dataset")), int(row["original_index"])),
    ):
        article_id = str(summary["article_id"])
        article = articles_by_id.get(article_id)
        if article is None:
            raise ValueError(f"Summary article_id not found in articles.jsonl: {article_id}")
        prediction = str(summary[summary_key]).strip()
        if not prediction:
            raise ValueError(f"Empty prediction for {article_id}")
        pairs.append(
            {
                "article_id": article_id,
                "source_dataset": str(summary.get("source_dataset")),
                "original_index": int(summary["original_index"]),
                "prediction": prediction,
                "document": str(article["document"]),
                "reference": str(article["expert_summary"]),
            }
        )
    return pairs


def _split_by_source(pairs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"PLOS": [], "eLife": []}
    for pair in pairs:
        source = str(pair["source_dataset"])
        if source not in out:
            raise ValueError(f"Unexpected source_dataset: {source!r}")
        out[source].append(pair)
    for source, rows in out.items():
        indices = [int(row["original_index"]) for row in rows]
        if indices != sorted(indices):
            raise ValueError(f"{source} pairs are not sorted by original_index")
    return out


def _refs_dicts(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    refs = [{"document": row["document"], "reference": row["reference"]} for row in rows]
    empty = sum(1 for row in refs if not str(row["reference"]).strip())
    if empty:
        raise ValueError(f"{empty}/{len(refs)} references are empty")
    return refs


def _summac_docs(rows: list[dict[str, Any]], mode: str) -> list[str]:
    if mode == "original":
        return [row["document"] for row in rows]
    if mode == "abstract":
        return [str(row["document"]).split("\n", 1)[0] for row in rows]
    if mode == "expert-summary":
        return [row["reference"] for row in rows]
    raise ValueError(f"Unknown summac_document mode: {mode!r}")


def _write_scores_txt(scores: dict[str, float], path: Path) -> None:
    path.write_text("\n".join(f"{key}: {value}" for key, value in scores.items()) + "\n", encoding="utf-8")


def evaluate_variant(
    *,
    run_name: str,
    model_key: str,
    variant: str,
    summac_document: str,
) -> dict[str, Any]:
    from evaluation.evaluation_final import evaluate_all

    articles = load_articles(run_name)
    summaries = _load_summary_rows(run_name, model_key, variant)
    summary_key = VARIANTS[variant]["summary_key"]
    pairs = _build_aligned_pairs(articles=articles, summaries=summaries, summary_key=summary_key)
    by_source = _split_by_source(pairs)

    source_scores: dict[str, dict[str, float]] = {}
    for source in ("PLOS", "eLife"):
        rows = by_source[source]
        preds = [row["prediction"] for row in rows]
        refs = _refs_dicts(rows)
        summac_docs = _summac_docs(rows, summac_document)
        source_scores[source] = evaluate_all(preds, refs, "lay_summ", summac_docs=summac_docs)

    overall = {
        key: float(np.mean([source_scores["PLOS"][key], source_scores["eLife"][key]]))
        for key in source_scores["PLOS"]
    }
    return {
        "variant": variant,
        "summac_document": summac_document,
        "article_count": len(pairs),
        "PLOS": source_scores["PLOS"],
        "eLife": source_scores["eLife"],
        "overall": overall,
        "per_article": pairs,
    }


def compare_with_leaderboard(*, metrics_dir: Path, leaderboard_path: Path) -> None:
    from evaluation.compare_task1_1_leaderboard import (
        compare_variant,
        load_leaderboard,
        load_overall,
        write_csv,
        write_markdown,
    )
    from evaluation.official_style_rank import append_official_rank_to_markdown, write_official_style_rank

    leaderboard = load_leaderboard(leaderboard_path)
    generated = load_overall(metrics_dir / "generated_scores.json")
    rewritten = load_overall(metrics_dir / "rewritten_scores.json")
    comparisons = {
        "initial": compare_variant("initial", generated, leaderboard),
        "rewritten": compare_variant("rewritten", rewritten, leaderboard),
    }
    flat_rows = comparisons["initial"] + comparisons["rewritten"]
    write_csv(flat_rows, metrics_dir / "leaderboard_comparison.csv")
    (metrics_dir / "leaderboard_comparison.json").write_text(
        json.dumps(
            {
                "caution": (
                    "Pilot validation sample versus published official-test leaderboard; "
                    "positions are descriptive only."
                ),
                "comparisons": comparisons,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_markdown(comparisons, generated, rewritten, metrics_dir / "leaderboard_comparison.md")
    rank_payload = write_official_style_rank(
        metrics_dir=metrics_dir,
        leaderboard_path=leaderboard_path,
        include_generated=True,
        include_rewritten=True,
    )
    append_official_rank_to_markdown(metrics_dir=metrics_dir, payload=rank_payload)


def run_evaluation(
    *,
    run_name: str,
    model_key: str,
    variant: str,
    summac_document: str,
    compare_leaderboard: bool,
    leaderboard_path: Path,
    save_per_article: bool,
) -> dict[str, Any]:
    _require_eval_environment()
    out_dir = _results_dir(run_name, model_key)
    variants = ["generated", "rewritten"] if variant == "both" else [variant]
    results: dict[str, Any] = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "model_key": model_key,
        "summac_document": summac_document,
        "variants": {},
    }

    for name in variants:
        result = evaluate_variant(
            run_name=run_name,
            model_key=model_key,
            variant=name,
            summac_document=summac_document,
        )
        score_path = out_dir / f"{name}_scores.json"
        save_json(
            {k: v for k, v in result.items() if k != "per_article"},
            score_path,
        )
        _write_scores_txt(result["overall"], out_dir / f"{name}_scores.txt")
        if save_per_article:
            per_article_path = out_dir / f"{name}_per_article.jsonl"
            per_article_path.write_text(
                "".join(
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
                    for row in result["per_article"]
                ),
                encoding="utf-8",
            )
        results["variants"][name] = result["overall"]
        print(f"[evaluate] {name} overall -> {score_path}")

    if "generated" in results["variants"] and "rewritten" in results["variants"]:
        delta = {
            key: results["variants"]["rewritten"][key] - results["variants"]["generated"][key]
            for key in results["variants"]["generated"]
        }
        save_json(delta, out_dir / "rewritten_minus_generated.json")
        _write_scores_txt(delta, out_dir / "rewritten_minus_generated.txt")
        print(f"[evaluate] delta -> {out_dir / 'rewritten_minus_generated.json'}")

    save_json(results, out_dir / "evaluation_metadata.json")

    if compare_leaderboard:
        if variant != "both":
            raise ValueError("--compare-leaderboard requires --variant both")
        compare_with_leaderboard(metrics_dir=out_dir, leaderboard_path=leaderboard_path)
        print(f"[evaluate] leaderboard comparison -> {out_dir / 'leaderboard_comparison.csv'}")
        print(f"[evaluate] official-style Final rank -> {out_dir / 'official_style_rank.json'}")

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Thesis lay-summary pipeline — Phase C official BioLaySumm evaluation."
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--model-key", type=str, default=DEFAULT_MODEL_KEY)
    parser.add_argument(
        "--variant",
        choices=["generated", "rewritten", "both"],
        default="both",
        help="Evaluate initial generated summaries, rewritten summaries, or both.",
    )
    parser.add_argument(
        "--summac-document",
        choices=["original", "abstract", "expert-summary"],
        default="original",
        help="Document passed to SummaC only.",
    )
    parser.add_argument(
        "--compare-leaderboard",
        action="store_true",
        help="Write leaderboard comparison (requires --variant both).",
    )
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=ROOT_DIR / "evaluation" / "task1_1_leaderboard.csv",
    )
    parser.add_argument(
        "--save-per-article",
        action="store_true",
        help="Also write per-article prediction lists for debugging.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_evaluation(
        run_name=args.run_name,
        model_key=args.model_key,
        variant=args.variant,
        summac_document=args.summac_document,
        compare_leaderboard=args.compare_leaderboard,
        leaderboard_path=args.leaderboard,
        save_per_article=args.save_per_article,
    )


if __name__ == "__main__":
    main()
