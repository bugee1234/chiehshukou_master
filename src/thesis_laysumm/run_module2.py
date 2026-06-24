from __future__ import annotations

import argparse
import json
import re
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
    ParallelExecutionError,
    load_articles,
    run_parallel,
    save_usage,
    sha256_text,
    usage_tracker,
)
from src.thesis_laysumm.paths import run_data_dir
from src.utils import load_jsonl, save_json, save_jsonl

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_MODEL_KEY = "gemini3_flash_preview_minimal"
DEFAULT_JUDGE_MODEL_KEY = "gemini3_flash_preview_minimal"
MAX_ITEM_ATTEMPTS = 2


def _loads_model_json(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # Some model responses include literal backslash-u text in reasoning fields
    # that is not a valid JSON unicode escape. Preserve it as text.
    text = re.sub(r"\\u(?![0-9a-fA-F]{4})", r"\\\\u", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = text.find("{")
        if start < 0:
            raise
        obj, _ = decoder.raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("model JSON response must be an object")
    return obj


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _module1_dir(run_name: str, model_key: str) -> Path:
    return run_data_dir(run_name) / "01_module1" / model_key


def _module2_dir(run_name: str, model_key: str) -> Path:
    path = run_data_dir(run_name) / "02_module2" / model_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _format_abstract_sentences(sentences: list[str]) -> str:
    lines = []
    for i, sentence in enumerate(sentences, start=1):
        text = str(sentence or "").strip()
        if text:
            lines.append(f"[S{i}] {text}")
    return "\n".join(lines) if lines else "[S1] (no abstract sentences parsed)"


def _is_final_keep(judgement: dict[str, Any]) -> bool:
    idx = judgement.get("abstract_sentence_idx")
    has_idx = idx is not None and str(idx).strip() not in {"", "null", "None"}
    return (
        _coerce_bool(judgement.get("aligned_with_abstract", False))
        and str(judgement.get("coverage_type", "")).strip() in {"exact", "paraphrase", "lay_generalization"}
        and str(judgement.get("confidence", "")).strip() in {"high", "medium"}
        and has_idx
        and bool(str(judgement.get("abstract_sentence", "") or "").strip())
        and not _coerce_bool(judgement.get("parse_error", False))
    )


def _build_judgement_row(
    *,
    af: dict[str, Any],
    model_key: str,
    judge_model_key: str,
    obj: dict[str, Any] | None,
    parse_error: bool = False,
    parse_error_message: str | None = None,
    raw_response: str | None = None,
) -> dict[str, Any]:
    obj = obj or {}
    coverage_type = str(obj.get("coverage_type", "absent")).strip()
    if coverage_type not in {
        "exact",
        "paraphrase",
        "lay_generalization",
        "topic_only",
        "absent",
        "contradicted",
    }:
        coverage_type = "absent"
    confidence = str(obj.get("confidence", "low")).strip()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    abstract_sentence = obj.get("abstract_sentence", None)
    abstract_sentence = None if abstract_sentence is None else str(abstract_sentence).strip()
    if abstract_sentence in {"", "null", "None"}:
        abstract_sentence = None

    abstract_sentence_idx = obj.get("abstract_sentence_idx", None)
    if abstract_sentence_idx in {"", "null", "None"}:
        abstract_sentence_idx = None

    row = {
        "af_id": af["af_id"],
        "model_key": model_key,
        "article_id": af["article_id"],
        "source_dataset": af["source_dataset"],
        "original_index": af["original_index"],
        "fact": af["fact"],
        "source_span": af.get("source_span", ""),
        "judge_model_key": judge_model_key,
        "judge_provider": MODEL_CONFIGS[judge_model_key]["provider"],
        "judge_model": MODEL_CONFIGS[judge_model_key]["model"],
        "aligned_with_abstract": False if parse_error else _coerce_bool(obj.get("aligned_with_abstract", False)),
        "abstract_sentence_idx": None if parse_error else abstract_sentence_idx,
        "abstract_sentence": None if parse_error else abstract_sentence,
        "coverage_type": "absent" if parse_error else coverage_type,
        "confidence": "low" if parse_error else confidence,
        "reasoning": str(obj.get("reasoning", "")).strip(),
        "parse_error": parse_error,
        "parse_error_message": parse_error_message,
        "raw_response": raw_response if parse_error else None,
    }
    row["final_keep"] = _is_final_keep(row)
    return row


def judge_candidate_af(
    *,
    run_name: str,
    model_key: str,
    judge_model_key: str,
    max_workers: int,
    resume: bool,
    limit_articles: int | None = None,
) -> list[dict[str, Any]]:
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_key {model_key!r}")
    if judge_model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown judge_model_key {judge_model_key!r}")

    articles = {str(r["id"]): r for r in load_articles(run_name)}
    candidate_path = _module1_dir(run_name, model_key) / "candidate_keep_af.jsonl"
    if not candidate_path.exists():
        raise FileNotFoundError(f"Missing Module 1 output: {candidate_path}")

    candidate_rows = load_jsonl(candidate_path)
    if limit_articles is not None:
        allowed_ids = {str(r["id"]) for r in load_articles(run_name)[:limit_articles]}
        candidate_rows = [r for r in candidate_rows if str(r["article_id"]) in allowed_ids]

    out_dir = _module2_dir(run_name, model_key)
    out_path = out_dir / "module2_judgements.jsonl"
    prompt = (PROMPTS_DIR / "judge_af_against_abstract.txt").read_text(encoding="utf-8")

    usage = usage_tracker(run_name, resume=resume)
    client = ProviderClient(judge_model_key, usage)
    existing = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("af_id")) for r in existing}

    def worker(af: dict[str, Any]) -> dict[str, Any]:
        article = articles[str(af["article_id"])]
        abstract_sentences = article.get("abstract_sentences") or []
        last_exc: Exception | None = None
        obj: dict[str, Any] | None = None
        raw = ""
        saw_response = False
        for attempt in range(1, MAX_ITEM_ATTEMPTS + 1):
            try:
                raw = client.chat(
                    stage=f"module2.judge_abstract:{judge_model_key}",
                    item_id=str(af["af_id"]),
                    messages=[
                        {
                            "role": "user",
                            "content": prompt.replace("{atomic_fact}", str(af["fact"])).replace(
                                "{abstract_sentences}",
                                _format_abstract_sentences(abstract_sentences),
                            ),
                        }
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                saw_response = True
                obj = _loads_model_json(raw)
                break
            except Exception as exc:
                last_exc = exc
        if obj is None:
            if not saw_response:
                raise RuntimeError(
                    f"module2 failed after {MAX_ITEM_ATTEMPTS} attempts "
                    f"for af_id={af['af_id']}: {last_exc}"
                ) from last_exc
            return _build_judgement_row(
                af=af,
                model_key=model_key,
                judge_model_key=judge_model_key,
                obj=None,
                parse_error=True,
                parse_error_message=str(last_exc) if last_exc else "unknown parse error",
                raw_response=raw,
            )

        return _build_judgement_row(
            af=af,
            model_key=model_key,
            judge_model_key=judge_model_key,
            obj=obj,
        )

    todo = [r for r in candidate_rows if str(r["af_id"]) not in done]
    try:
        new_rows = run_parallel(
            todo,
            worker,
            max_workers=max_workers,
            desc=f"module2 | {model_key}",
        )
    except ParallelExecutionError as exc:
        partial_rows = existing + exc.partial_results
        partial_rows.sort(key=lambda r: (str(r["article_id"]), str(r["af_id"])))
        save_jsonl(partial_rows, out_path)
        save_usage(run_name, usage)
        print(f"[module2] checkpointed {len(partial_rows)} clean judgements before failure -> {out_path}")
        raise
    all_rows = existing + new_rows
    all_rows.sort(key=lambda r: (str(r["article_id"]), str(r["af_id"])))
    save_jsonl(all_rows, out_path)
    save_usage(run_name, usage)

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "stage": "02_module2",
            "model_key": model_key,
            "judge_model_key": judge_model_key,
            "judge_model": MODEL_CONFIGS[judge_model_key]["model"],
            "max_workers": max_workers,
            "articles_limit": limit_articles,
            "prompt_file": str(PROMPTS_DIR / "judge_af_against_abstract.txt"),
            "prompt_sha256": sha256_text(prompt),
            "candidate_af_total": len(candidate_rows),
            "judgements_total": len(all_rows),
            "final_keep_total": sum(1 for r in all_rows if r.get("final_keep")),
            "coverage_type_counts": dict(Counter(str(r.get("coverage_type")) for r in all_rows)),
            "confidence_counts": dict(Counter(str(r.get("confidence")) for r in all_rows)),
            "parse_error_count": sum(1 for r in all_rows if r.get("parse_error")),
            "final_keep_rule": {
                "aligned_with_abstract": True,
                "coverage_type": ["exact", "paraphrase", "lay_generalization"],
                "confidence": ["high", "medium"],
                "abstract_sentence_idx": "non-null",
                "abstract_sentence": "non-empty",
                "parse_error": False,
            },
            "usage_summary": usage.summarize(),
        },
        out_dir / "module2_judgements_metadata.json",
    )

    print(
        f"[module2] {model_key} judgements={len(all_rows)} "
        f"final_keep={sum(1 for r in all_rows if r.get('final_keep'))}"
    )
    print(f"[module2] output -> {out_path}")
    return all_rows


def build_final_keep_af(*, run_name: str, model_key: str) -> list[dict[str, Any]]:
    from collections import defaultdict

    from src.thesis_laysumm.llm_utils import load_articles

    out_dir = _module2_dir(run_name, model_key)
    judgements = load_jsonl(out_dir / "module2_judgements.jsonl")
    articles = {str(r["id"]): r for r in load_articles(run_name)}

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    pre_dedup_count = 0
    for row in judgements:
        if not row.get("final_keep"):
            continue
        pre_dedup_count += 1
        idx_raw = row.get("abstract_sentence_idx")
        if idx_raw in {None, "", "null", "None"}:
            continue
        idx = int(idx_raw)
        grouped[(str(row["article_id"]), idx)].append(row)

    rows: list[dict[str, Any]] = []
    skipped_invalid_idx = 0
    for (article_id, idx), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        article = articles.get(article_id)
        if article is None:
            continue
        abstract_sentences = article.get("abstract_sentences") or []
        if idx < 1 or idx > len(abstract_sentences):
            skipped_invalid_idx += 1
            continue

        canonical_fact = str(abstract_sentences[idx - 1]).strip()
        if not canonical_fact:
            skipped_invalid_idx += 1
            continue

        group.sort(key=lambda r: str(r["af_id"]))
        rep = group[0]
        merged_ids = sorted({str(r["af_id"]) for r in group})
        rows.append(
            {
                "af_id": rep["af_id"],
                "model_key": model_key,
                "article_id": article_id,
                "source_dataset": rep["source_dataset"],
                "original_index": rep["original_index"],
                "fact": canonical_fact,
                "source_span": rep.get("source_span", ""),
                "abstract_sentence_idx": idx,
                "abstract_sentence": canonical_fact,
                "merged_from_af_ids": merged_ids if len(merged_ids) > 1 else [],
                "merged_af_count": len(merged_ids),
                "module2_coverage_type": rep.get("coverage_type"),
                "module2_confidence": rep.get("confidence"),
                "module2_reasoning": rep.get("reasoning"),
                "judge_model_key": rep.get("judge_model_key"),
                "judge_model": rep.get("judge_model"),
            }
        )

    rows.sort(key=lambda r: (str(r["article_id"]), int(r["abstract_sentence_idx"])))
    save_jsonl(rows, out_dir / "final_keep_af.jsonl")
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model_key": model_key,
            "final_keep_rule": "dedupe_by_abstract_sentence_idx; fact=canonical abstract sentence",
            "final_keep_pre_dedup": pre_dedup_count,
            "final_keep_post_dedup": len(rows),
            "skipped_invalid_abstract_idx": skipped_invalid_idx,
            "final_keep_by_source": dict(Counter(str(r["source_dataset"]) for r in rows)),
            "final_keep_by_article": dict(Counter(str(r["article_id"]) for r in rows)),
        },
        out_dir / "final_keep_af_metadata.json",
    )
    print(
        f"[module2] final_keep_af={len(rows)} (deduped from {pre_dedup_count}) "
        f"-> {out_dir / 'final_keep_af.jsonl'}"
    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Thesis lay-summary pipeline — Module 2: abstract-aligned review of candidate AF."
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--model-key", type=str, default=DEFAULT_MODEL_KEY)
    parser.add_argument("--judge-model-key", type=str, default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit-articles", type=int, default=None)
    parser.add_argument("--judgements-only", action="store_true")
    parser.add_argument("--final-keep-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    judge_model_key = args.judge_model_key or args.model_key
    resume = not args.no_resume

    if not args.final_keep_only:
        judge_candidate_af(
            run_name=args.run_name,
            model_key=args.model_key,
            judge_model_key=judge_model_key,
            max_workers=args.max_workers,
            resume=resume,
            limit_articles=args.limit_articles,
        )
    if not args.judgements_only:
        build_final_keep_af(run_name=args.run_name, model_key=args.model_key)


if __name__ == "__main__":
    main()
