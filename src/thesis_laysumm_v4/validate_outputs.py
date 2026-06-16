from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.thesis_laysumm.paths import run_data_dir, run_results_dir
from src.thesis_laysumm_v4.common import length_policy, safe_word_count
from src.utils import load_json, load_jsonl


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def _article_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("id") or row.get("article_id")) for row in rows}


def validate_run(*, run_name: str, model_key: str, mode: str, require_eval: bool) -> None:
    data_dir = run_data_dir(run_name)
    results_dir = run_results_dir(run_name) / model_key

    articles = load_jsonl(_require(data_dir / "00_articles" / "articles.jsonl"))
    if not articles:
        raise ValueError("articles.jsonl is empty")
    articles_by_id = {str(row["id"]): row for row in articles}

    evidence_rows = load_jsonl(
        _require(data_dir / "02_5_evidence_table" / mode / model_key / "evidence_table.jsonl")
    )
    if _article_ids(evidence_rows) != set(articles_by_id):
        raise ValueError("Evidence article IDs do not match articles.jsonl")
    for row in evidence_rows:
        if row.get("parse_error"):
            raise ValueError(f"Evidence parse_error: {row.get('article_id')}")
        inner = row.get("evidence_rows") or []
        if not inner:
            raise ValueError(f"No evidence rows for {row.get('article_id')}")
        for erow in inner:
            for key in ("evidence_row_id", "core_keep_af", "allowed_evidence_spans", "story_role"):
                if not erow.get(key):
                    raise ValueError(f"Evidence row missing {key}: {row.get('article_id')} {erow}")
            if "lay_context" not in erow:
                raise ValueError(f"Evidence row missing lay_context key: {row.get('article_id')} {erow}")

    questions = load_jsonl(_require(data_dir / "03_questions_v4" / mode / model_key / "questions_1t3f_nota.jsonl"))
    if not questions:
        raise ValueError("questions_1t3f_nota.jsonl is empty")

    generated = load_jsonl(_require(data_dir / "04_summaries" / model_key / "generated_summaries.jsonl"))
    rewritten = load_jsonl(_require(data_dir / "06_rewritten" / model_key / "rewritten_summaries.jsonl"))
    for label, rows, key in (
        ("generated", generated, "generated_summary"),
        ("rewritten", rewritten, "rewritten_summary"),
    ):
        if _article_ids(rows) != set(articles_by_id):
            raise ValueError(f"{label} article IDs do not match articles.jsonl")
        for row in rows:
            aid = str(row["article_id"])
            if row.get("parse_error"):
                raise ValueError(f"{label} parse_error: {aid}")
            text = str(row.get(key, "")).strip()
            if not text:
                raise ValueError(f"{label} summary is empty: {aid}")
            policy = length_policy(
                mode=mode,
                expert_word_count=int(articles_by_id[aid].get("expert_summary_word_count") or 0),
            )
            wc = safe_word_count(text)
            if label == "rewritten" and wc < int(policy["min_word_count"]):
                raise ValueError(f"{label} summary below min length: {aid} wc={wc} policy={policy}")
            if wc > int(policy["max_word_count"]) + 20:
                raise ValueError(f"{label} summary far above max length: {aid} wc={wc} policy={policy}")

    answers = load_jsonl(_require(data_dir / "05_module3_answers_v4" / mode / model_key / "module3_answers.jsonl"))
    if len(answers) != len(questions):
        raise ValueError(f"Answer/question count mismatch: answers={len(answers)} questions={len(questions)}")
    if any(row.get("parse_error") for row in answers):
        bad = [row.get("question_id") for row in answers if row.get("parse_error")]
        raise ValueError(f"Answer parse_error rows: {bad[:10]}")

    if require_eval:
        for name in ("generated_scores.json", "rewritten_scores.json", "leaderboard_comparison.json", "official_style_rank.json"):
            _require(results_dir / name)
        generated_scores = load_json(results_dir / "generated_scores.json")
        rewritten_scores = load_json(results_dir / "rewritten_scores.json")
        for label, payload in (("generated", generated_scores), ("rewritten", rewritten_scores)):
            if int(payload.get("article_count") or 0) != len(articles):
                raise ValueError(f"{label} evaluated article_count does not match articles")
            for metric in ("ROUGE", "BLEU", "METEOR", "BERTScore", "FKGL", "DCRS", "CLI", "LENS", "AlignScore", "SummaC"):
                if metric not in (payload.get("overall") or {}):
                    raise ValueError(f"{label} scores missing metric {metric}")

    print(
        f"[v4 validate] ok run={run_name} model={model_key} "
        f"articles={len(articles)} evidence_articles={len(evidence_rows)} questions={len(questions)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate v4 lay-summary pipeline outputs.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--mode", choices=["balanced", "factuality_chase"], default="factuality_chase")
    parser.add_argument("--require-eval", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_run(
        run_name=args.run_name,
        model_key=args.model_key,
        mode=args.mode,
        require_eval=args.require_eval,
    )


if __name__ == "__main__":
    main()
