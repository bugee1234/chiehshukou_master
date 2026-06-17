from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.thesis_laysumm.paths import run_data_dir, run_results_dir
from src.thesis_laysumm_v9.common import find_mojibake, readability_proxy_score
from src.utils import load_json, load_jsonl


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def _ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(r.get("id") or r.get("article_id")) for r in rows}


def inspect_model(*, run_name: str, model_key: str, mode: str) -> dict[str, Any]:
    data_dir = run_data_dir(run_name)
    results_dir = run_results_dir(run_name) / model_key
    articles = load_jsonl(_require(data_dir / "00_articles" / "articles.jsonl"))
    generated = load_jsonl(_require(data_dir / "04_summaries" / model_key / "generated_summaries.jsonl"))
    rewritten = load_jsonl(_require(data_dir / "06_rewritten" / model_key / "rewritten_summaries.jsonl"))
    questions = load_jsonl(_require(data_dir / "03_questions_v9" / mode / model_key / "questions_1t3f_nota.jsonl"))
    answers = load_jsonl(_require(data_dir / "05_module3_answers_v9" / mode / model_key / "module3_answers.jsonl"))
    wrong = load_jsonl(_require(data_dir / "05_module3_answers_v9" / mode / model_key / "wrong_answers.jsonl"))

    article_ids = _ids(articles)
    if _ids(generated) != article_ids:
        raise ValueError(f"{model_key}: generated IDs do not match articles")
    if _ids(rewritten) != article_ids:
        raise ValueError(f"{model_key}: rewritten IDs do not match articles")
    if len(answers) != len(questions):
        raise ValueError(f"{model_key}: answer/question mismatch")

    mojibake_rows = []
    improved = 0
    changed = 0
    for row in rewritten:
        text = str(row.get("rewritten_summary", ""))
        hits = find_mojibake(text)
        if hits:
            mojibake_rows.append({"article_id": row.get("article_id"), "hits": hits})
        gen = str(row.get("generated_summary", ""))
        if gen != text:
            changed += 1
        before = row.get("generated_readability_proxy_score")
        after = row.get("rewritten_readability_proxy_score")
        if before is None:
            before = readability_proxy_score(gen)
        if after is None:
            after = readability_proxy_score(text)
        if float(after) <= float(before):
            improved += 1

    generated_scores = load_json(results_dir / "generated_scores.json") if (results_dir / "generated_scores.json").exists() else None
    rewritten_scores = load_json(results_dir / "rewritten_scores.json") if (results_dir / "rewritten_scores.json").exists() else None

    return {
        "model_key": model_key,
        "articles": len(articles),
        "questions": len(questions),
        "answers": len(answers),
        "wrong_answers": len(wrong),
        "generated_parse_errors": sum(1 for r in generated if r.get("parse_error")),
        "rewritten_parse_errors": sum(1 for r in rewritten if r.get("parse_error")),
        "rewritten_changed": changed,
        "readability_proxy_improved": f"{improved}/{len(rewritten)}",
        "rewrite_invoked": sum(1 for r in rewritten if r.get("rewrite_invoked")),
        "selected_variant_counts": dict(Counter(str(r.get("selected_variant")) for r in rewritten)),
        "below_min_length": sum(1 for r in rewritten if r.get("below_min_length")),
        "mojibake_rows": mojibake_rows,
        "generated_overall": (generated_scores or {}).get("overall"),
        "rewritten_overall": (rewritten_scores or {}).get("overall"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a completed V9 run.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=["balanced", "factuality_chase"], default="factuality_chase")
    parser.add_argument("--model-key", action="append", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = {
        "run_name": args.run_name,
        "mode": args.mode,
        "models": [inspect_model(run_name=args.run_name, model_key=m, mode=args.mode) for m in args.model_key],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

