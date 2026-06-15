from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiment_2.run_three_model_exp2 import MODEL_CONFIGS, ProviderClient
from src.thesis_laysumm.llm_utils import (
    chunk_words,
    load_articles,
    norm_text,
    run_parallel,
    save_usage,
    sha256_text,
    usage_tracker,
)
from src.thesis_laysumm.paths import run_data_dir
from src.utils import load_jsonl, save_json, save_jsonl

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_MODEL_KEY = "gemini3_flash_preview_minimal"
DEFAULT_CHUNK_WORDS = 1200
DEFAULT_OVERLAP_WORDS = 120


def _module1_dir(run_name: str, model_key: str) -> Path:
    path = run_data_dir(run_name) / "01_module1" / model_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prompt_text() -> str:
    return (PROMPTS_DIR / "select_summary_candidate_af.txt").read_text(encoding="utf-8")


def extract_candidate_keep_af(
    *,
    run_name: str,
    model_key: str,
    chunk_words_n: int,
    overlap_words: int,
    max_workers: int,
    resume: bool,
    limit_articles: int | None = None,
) -> list[dict[str, Any]]:
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_key {model_key!r}; expected one of {sorted(MODEL_CONFIGS)}")

    articles = load_articles(run_name)
    if limit_articles is not None:
        articles = articles[:limit_articles]
    out_dir = _module1_dir(run_name, model_key)
    out_af = out_dir / "candidate_keep_af.jsonl"
    out_failed = out_dir / "failed_chunks.jsonl"
    prompt = _prompt_text()

    usage = usage_tracker(run_name, resume=resume)
    client = ProviderClient(model_key, usage)
    existing = load_jsonl(out_af) if resume and out_af.exists() else []
    done_articles = {str(r.get("article_id")) for r in existing}
    existing_failed = load_jsonl(out_failed) if resume and out_failed.exists() else []

    def worker(art: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        article_id = str(art["id"])
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        count = 0
        for chunk in chunk_words(str(art["article"]), chunk_words_n, overlap_words):
            raw = ""
            try:
                raw = client.chat(
                    stage=f"module1.candidate_af:{model_key}",
                    item_id=f"{article_id}:chunk{chunk['chunk_idx']}",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt.replace("{source_text}", chunk["chunk_text"]),
                        }
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                obj = json.loads(raw)
                facts = obj.get("candidate_atomic_facts", obj.get("keep_atomic_facts", []))
                if not isinstance(facts, list):
                    facts = []
                for fact_obj in facts:
                    if not isinstance(fact_obj, dict):
                        continue
                    fact = str(fact_obj.get("fact", "")).strip()
                    key = norm_text(fact)
                    if not fact or key in seen:
                        continue
                    seen.add(key)
                    count += 1
                    rows.append(
                        {
                            "af_id": f"{article_id}_cand_{count:04d}",
                            "model_key": model_key,
                            "article_id": article_id,
                            "source_dataset": art["source_dataset"],
                            "original_index": art["original_index"],
                            "fact": fact,
                            "source_span": str(fact_obj.get("source_span", "")).strip(),
                            "selection_reasoning": str(
                                fact_obj.get("reasoning", fact_obj.get("keep_reasoning", ""))
                            ).strip(),
                            "article_chunk_idx": chunk["chunk_idx"],
                            "chunk_start_word": chunk["start_word"],
                            "chunk_end_word": chunk["end_word"],
                        }
                    )
            except Exception as exc:
                failed.append(
                    {
                        "article_id": article_id,
                        "chunk_idx": chunk["chunk_idx"],
                        "error": str(exc),
                        "raw_preview": raw[:300],
                    }
                )
        return article_id, rows, failed

    todo = [a for a in articles if str(a["id"]) not in done_articles]
    new_rows: list[dict[str, Any]] = []
    new_failed: list[dict[str, Any]] = []
    for _, rows, failed in run_parallel(
        todo,
        worker,
        max_workers=max_workers,
        desc=f"module1 | {model_key}",
    ):
        new_rows.extend(rows)
        new_failed.extend(failed)

    all_rows = existing + new_rows
    all_rows.sort(key=lambda r: (str(r["article_id"]), str(r["af_id"])))
    all_failed = existing_failed + new_failed
    save_jsonl(all_rows, out_af)
    save_jsonl(all_failed, out_failed)
    save_usage(run_name, usage)

    completed_articles = {str(r["article_id"]) for r in all_rows}
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "stage": "01_module1",
            "model_key": model_key,
            "provider": MODEL_CONFIGS[model_key]["provider"],
            "model": MODEL_CONFIGS[model_key]["model"],
            "chunk_words": chunk_words_n,
            "overlap_words": overlap_words,
            "max_workers": max_workers,
            "prompt_file": str(PROMPTS_DIR / "select_summary_candidate_af.txt"),
            "prompt_sha256": sha256_text(prompt),
            "articles_total": len(articles),
            "articles_limit": limit_articles,
            "articles_completed": len(completed_articles),
            "candidate_keep_af_total": len(all_rows),
            "candidate_keep_af_by_source": dict(Counter(str(r["source_dataset"]) for r in all_rows)),
            "candidate_keep_af_by_article": dict(Counter(str(r["article_id"]) for r in all_rows)),
            "failed_chunk_count": len(all_failed),
            "usage_summary": usage.summarize(),
        },
        out_dir / "candidate_keep_af_metadata.json",
    )

    print(
        f"[module1] {model_key} articles={len(completed_articles)}/{len(articles)} "
        f"candidate_af={len(all_rows)} failed_chunks={len(all_failed)}"
    )
    print(f"[module1] output -> {out_af}")
    return all_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Thesis lay-summary pipeline — Module 1: select candidate summary atomic facts from articles."
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--model-key", type=str, default=DEFAULT_MODEL_KEY)
    parser.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    parser.add_argument("--overlap-words", type=int, default=DEFAULT_OVERLAP_WORDS)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit-articles", type=int, default=None, help="Process only the first N articles.")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    extract_candidate_keep_af(
        run_name=args.run_name,
        model_key=args.model_key,
        chunk_words_n=args.chunk_words,
        overlap_words=args.overlap_words,
        max_workers=args.max_workers,
        resume=not args.no_resume,
        limit_articles=args.limit_articles,
    )


if __name__ == "__main__":
    main()
