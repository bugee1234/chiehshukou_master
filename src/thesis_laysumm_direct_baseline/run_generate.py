from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiment_2.run_three_model_exp2 import MODEL_CONFIGS, ProviderClient, UsageTracker
from src.thesis_laysumm.llm_utils import ParallelExecutionError, run_parallel, sha256_text
from src.thesis_laysumm_direct_baseline.paths import (
    PROMPT_PATH,
    inputs_path,
    run_data_dir,
    summaries_path,
)
from src.thesis_laysumm_v11.common import safe_word_count
from src.utils import load_jsonl, save_json, save_jsonl


MAX_ITEM_ATTEMPTS = 2
ALLOWED_PROMPT_FIELDS = ("article",)


def build_prompt(article: dict[str, Any], template: str) -> str:
    return template.replace("{article}", str(article.get("article") or ""))


def truncate_words(text: str, max_words: int | None) -> tuple[str, bool]:
    clean = str(text or "").strip()
    if not max_words:
        return clean, False
    tokens = re.findall(r"\S+", clean)
    if len(tokens) <= max_words:
        return clean, False
    truncated = " ".join(tokens[:max_words]).rstrip()
    if truncated and truncated[-1] not in ".!?":
        truncated += "."
    return truncated, True


def _usage_tracker(run_name: str, resume: bool) -> UsageTracker:
    path = run_data_dir(run_name) / "api_usage_calls.csv"
    return UsageTracker.from_csv(path) if resume and path.exists() else UsageTracker(rows=[])


def _save_usage(run_name: str, usage: UsageTracker) -> None:
    usage.save(run_data_dir(run_name))


def _sorted(rows: list[dict[str, Any]], order: dict[str, int]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: order[str(row["article_id"])])


def generate(
    *,
    run_name: str,
    model_key: str,
    max_workers: int,
    resume: bool,
    prompt_path: Path,
    max_summary_words: int | None,
) -> list[dict[str, Any]]:
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}. Valid: {sorted(MODEL_CONFIGS)}")
    in_path = inputs_path(run_name)
    if not in_path.exists():
        raise FileNotFoundError(f"Prepare the direct baseline run first: {in_path}")
    articles = load_jsonl(in_path)
    order = {str(row["id"]): index for index, row in enumerate(articles)}
    if len(order) != len(articles):
        raise ValueError("Direct-baseline inputs contain duplicate IDs")

    template = prompt_path.read_text(encoding="utf-8")
    prompt_hash = sha256_text(template)
    out_path = summaries_path(run_name, model_key)
    existing = load_jsonl(out_path) if resume and out_path.exists() else []
    existing_ids = [str(row.get("article_id")) for row in existing]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError(f"Existing direct summaries contain duplicate IDs: {out_path}")
    if set(existing_ids) - set(order):
        raise ValueError("Existing direct summaries contain IDs outside this run")
    if any(row.get("parse_error") or not str(row.get("generated_summary") or "").strip() for row in existing):
        raise ValueError("Existing direct summaries contain empty/parse_error rows; clean them before resume")

    done = set(existing_ids)
    todo = [article for article in articles if str(article["id"]) not in done]
    usage = _usage_tracker(run_name, resume=resume)
    client = ProviderClient(model_key, usage)

    def worker(article: dict[str, Any]) -> dict[str, Any]:
        article_id = str(article["id"])
        prompt = build_prompt(article, template)
        raw = ""
        last_error = "empty model response"
        for attempt in range(1, MAX_ITEM_ATTEMPTS + 1):
            raw = client.chat(
                stage=f"direct_zero_shot.generate:{model_key}",
                item_id=article_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format=None,
            )
            if raw.strip():
                break
            last_error = f"empty model response on attempt {attempt}"
        if not raw.strip():
            raise RuntimeError(
                f"direct generation failed after {MAX_ITEM_ATTEMPTS} attempts "
                f"for article_id={article_id}: {last_error}"
            )
        raw_summary = raw.strip()
        summary, truncated = truncate_words(raw_summary, max_summary_words)
        return {
            "article_id": article_id,
            "source_dataset": article.get("source_dataset"),
            "original_index": article.get("original_index"),
            "model_key": model_key,
            "provider": MODEL_CONFIGS[model_key]["provider"],
            "model": MODEL_CONFIGS[model_key]["model"],
            "baseline_design": "direct_zero_shot_full_article_length_controlled"
            if max_summary_words
            else "direct_zero_shot_full_article",
            "prompt_version": prompt_path.stem,
            "prompt_file": str(prompt_path),
            "prompt_sha256": prompt_hash,
            "visible_input_fields": list(ALLOWED_PROMPT_FIELDS),
            "temperature": 0.0,
            "generated_summary": summary,
            "word_count": safe_word_count(summary),
            "max_summary_words": max_summary_words,
            "truncated_by_word_limit": truncated,
            "raw_response": raw,
            "parse_error": False,
        }

    try:
        new_rows = run_parallel(
            todo,
            worker,
            max_workers=max_workers,
            desc=f"direct baseline | {model_key}",
        )
    except ParallelExecutionError as exc:
        checkpoint = _sorted(existing + exc.partial_results, order)
        save_jsonl(checkpoint, out_path)
        _save_usage(run_name, usage)
        print(f"[direct generate] checkpointed {len(checkpoint)} clean rows before failure -> {out_path}")
        raise

    combined = _sorted(existing + new_rows, order)
    save_jsonl(combined, out_path)
    _save_usage(run_name, usage)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "stage": "direct_zero_shot_generation",
            "model_key": model_key,
            "article_count": len(combined),
            "new_article_count": len(new_rows),
            "prompt_sha256": prompt_hash,
            "visible_input_fields": list(ALLOWED_PROMPT_FIELDS),
            "postprocessing": "strip_outer_whitespace_then_word_truncate"
            if max_summary_words
            else "strip_outer_whitespace_only",
            "max_summary_words": max_summary_words,
            "usage_summary": usage.summarize(),
        },
        out_path.parent / "generation_metadata.json",
    )
    print(f"[direct generate] {model_key}: summaries={len(combined)} new={len(new_rows)} -> {out_path}")
    return combined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate direct zero-shot full-article lay summaries.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--prompt-path", type=Path, default=PROMPT_PATH)
    parser.add_argument("--max-summary-words", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate(
        run_name=args.run_name,
        model_key=args.model_key,
        max_workers=args.max_workers,
        resume=not args.no_resume,
        prompt_path=args.prompt_path,
        max_summary_words=args.max_summary_words,
    )


if __name__ == "__main__":
    main()
