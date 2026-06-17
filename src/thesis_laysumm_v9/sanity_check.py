from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiment_2.run_three_model_exp2 import MODEL_CONFIGS
from src.thesis_laysumm.paths import run_data_dir, run_results_dir
from src.thesis_laysumm_v9.common import find_mojibake, readability_stats


EXPECTED_MODELS = ["gpt41_mini", "gemini3_flash_preview_minimal", "gemini25_flash_non_thinking"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_diagnostics() -> None:
    path = ROOT_DIR / "results" / "laysumm_pipeline" / "analysis_sentence_diagnostics_v7_v8" / "sentence_readability_overlap.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing sentence diagnostics: {path}")
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError(f"Sentence diagnostics is empty: {path}")
    required = {
        "run",
        "model",
        "variant",
        "article_id",
        "source_dataset",
        "sentence",
        "risk_score",
        "risk_tags",
        "word_count",
        "long_word_ratio",
        "hard_word_ratio",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Sentence diagnostics missing columns: {sorted(missing)}")
    print(
        "[V9 sanity] sentence diagnostics ok "
        f"rows={len(rows)} runs={dict(Counter(row['run'] for row in rows))}"
    )


def check_models() -> None:
    missing = [key for key in EXPECTED_MODELS if key not in MODEL_CONFIGS]
    if missing:
        raise ValueError(f"Missing expected model configs: {missing}")
    for key in EXPECTED_MODELS:
        cfg = MODEL_CONFIGS[key]
        print(f"[V9 sanity] model ok {key}: provider={cfg['provider']} model={cfg['model']}")


def check_run_outputs(run_name: str, model_key: str, *, require_eval: bool) -> None:
    data_dir = run_data_dir(run_name)
    articles_path = data_dir / "00_articles" / "articles.jsonl"
    if not articles_path.exists():
        raise FileNotFoundError(f"Missing articles: {articles_path}")
    articles = _read_jsonl(articles_path)
    article_ids = {str(row["id"]) for row in articles}
    print(f"[V9 sanity] articles ok count={len(articles)}")

    summary_paths = [
        ("generated", data_dir / "04_summaries" / model_key / "generated_summaries.jsonl", "generated_summary"),
        ("rewritten", data_dir / "06_rewritten" / model_key / "rewritten_summaries.jsonl", "rewritten_summary"),
    ]
    for label, path, key in summary_paths:
        if not path.exists():
            print(f"[V9 sanity] skip missing {label}: {path}")
            continue
        rows = _read_jsonl(path)
        ids = {str(row["article_id"]) for row in rows}
        if ids != article_ids:
            raise ValueError(f"{label} article ids mismatch: rows={len(ids)} articles={len(article_ids)}")
        mojibake_hits = []
        high_risk = 0
        for row in rows:
            text = str(row.get(key, ""))
            hits = find_mojibake(text)
            if hits:
                mojibake_hits.append({"article_id": row.get("article_id"), "hits": hits})
            stats = readability_stats(text)
            high_risk += int(stats.get("high_risk_sentence_count", 0))
            if int(stats.get("max_sentence_words", 0)) > 40:
                raise ValueError(f"{label} very long sentence: {row.get('article_id')} stats={stats}")
        if mojibake_hits:
            raise ValueError(f"{label} mojibake-like text found: {mojibake_hits[:5]}")
        print(f"[V9 sanity] {label} ok rows={len(rows)} high_risk_sentence_count={high_risk}")

    if require_eval:
        results_dir = run_results_dir(run_name) / model_key
        for name in ("generated_scores.json", "rewritten_scores.json", "leaderboard_comparison.json", "official_style_rank.json"):
            path = results_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Missing eval output: {path}")
        print(f"[V9 sanity] eval outputs ok {results_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V9 lightweight sanity checks.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--require-eval", action="store_true")
    parser.add_argument("--skip-diagnostics", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    check_models()
    if not args.skip_diagnostics:
        check_diagnostics()
    if args.run_name or args.model_key:
        if not args.run_name or not args.model_key:
            raise ValueError("--run-name and --model-key must be provided together")
        check_run_outputs(args.run_name, args.model_key, require_eval=args.require_eval)


if __name__ == "__main__":
    main()
