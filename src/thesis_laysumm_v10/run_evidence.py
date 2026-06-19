from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiment_2.run_three_model_exp2 import MODEL_CONFIGS, ProviderClient
from src.thesis_laysumm.llm_utils import load_articles, run_parallel, save_usage, sha256_text, usage_tracker
from src.thesis_laysumm.paths import run_data_dir
from src.thesis_laysumm_v10.common import PROMPTS_DIR, compact_json, format_numbered_sentences, loads_json_object, require_mode
from src.utils import load_jsonl, save_json, save_jsonl

DEFAULT_MODEL_KEY = "gpt41_mini"


def _module2_dir(run_name: str, model_key: str) -> Path:
    return run_data_dir(run_name) / "02_module2" / model_key


def _out_dir(run_name: str, model_key: str, mode: str) -> Path:
    path = run_data_dir(run_name) / "02_5_evidence_table" / mode / model_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _candidate_blocks(rows: list[dict[str, Any]], *, max_per_sentence: int) -> str:
    cov_rank = {"exact": 0, "paraphrase": 1, "lay_generalization": 2}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("final_keep"):
            continue
        idx_raw = row.get("abstract_sentence_idx")
        if idx_raw in {None, "", "null", "None"}:
            continue
        try:
            idx = int(idx_raw)
        except (TypeError, ValueError):
            continue
        grouped[idx].append(row)

    blocks = []
    for idx in sorted(grouped):
        group = grouped[idx]
        group.sort(
            key=lambda r: (
                cov_rank.get(str(r.get("coverage_type")), 9),
                0 if str(r.get("confidence")) == "high" else 1,
                str(r.get("af_id")),
            )
        )
        facts = []
        for row in group[:max_per_sentence]:
            facts.append(
                {
                    "af_id": row.get("af_id"),
                    "fact": row.get("fact"),
                    "source_span": row.get("source_span"),
                    "coverage_type": row.get("coverage_type"),
                    "confidence": row.get("confidence"),
                }
            )
        blocks.append({"abstract_sentence_idx": idx, "candidate_facts": facts})
    return compact_json(blocks)


def _normalize_evidence_rows(
    *,
    obj: dict[str, Any],
    article: dict[str, Any],
    valid_af_ids: set[str],
    mode: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    raw_rows = obj.get("evidence_rows", [])
    if not isinstance(raw_rows, list):
        return [], ["evidence_rows must be a list"]

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_rows, start=1):
        if not isinstance(raw, dict):
            errors.append("each evidence row must be an object")
            continue
        row_id = str(raw.get("evidence_row_id") or f"row_{i:04d}").strip()
        if row_id in seen:
            row_id = f"row_{i:04d}"
        seen.add(row_id)
        idx_raw = raw.get("abstract_sentence_idx")
        try:
            idx = int(idx_raw)
        except (TypeError, ValueError):
            errors.append(f"{row_id}: invalid abstract_sentence_idx")
            continue
        support_af_ids = [
            str(x).strip()
            for x in (raw.get("support_af_ids") or [])
            if str(x).strip() and str(x).strip() in valid_af_ids
        ]
        spans = [str(x).strip() for x in (raw.get("allowed_evidence_spans") or []) if str(x).strip()]
        if not spans:
            claim = str(raw.get("core_keep_af") or raw.get("abstract_claim") or "").strip()
            if claim:
                spans = [claim]
        high_risk = []
        for item in raw.get("high_risk_check_facts") or []:
            if not isinstance(item, dict):
                continue
            fact = str(item.get("fact", "")).strip()
            if not fact:
                continue
            high_risk.append(
                {
                    "fact": fact,
                    "slot_type": str(item.get("slot_type", "entity")).strip() or "entity",
                    "source_span": str(item.get("source_span", "")).strip(),
                }
            )
        rows.append(
            {
                "article_id": str(article["id"]),
                "source_dataset": article.get("source_dataset"),
                "original_index": article.get("original_index"),
                "mode": mode,
                "evidence_row_id": row_id,
                "abstract_sentence_idx": idx,
                "abstract_claim": str(raw.get("abstract_claim", "")).strip(),
                "core_keep_af": str(raw.get("core_keep_af", raw.get("abstract_claim", ""))).strip(),
                "expert_summary_hint": str(raw.get("expert_summary_hint", "")).strip(),
                "lay_context": str(raw.get("lay_context", "")).strip(),
                "story_role": str(raw.get("story_role", "finding")).strip() or "finding",
                "support_af_ids": support_af_ids,
                "allowed_evidence_spans": spans[:4],
                "must_use_terms": [str(x).strip() for x in (raw.get("must_use_terms") or []) if str(x).strip()][:10],
                "forbidden_overgeneralizations": [
                    str(x).strip() for x in (raw.get("forbidden_overgeneralizations") or []) if str(x).strip()
                ][:8],
                "high_risk_check_facts": high_risk[:3],
            }
        )
    return rows, errors


def build_evidence_tables(
    *,
    run_name: str,
    model_key: str,
    mode: str,
    max_workers: int,
    resume: bool,
    limit_articles: int | None = None,
) -> list[dict[str, Any]]:
    require_mode(mode)
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_key {model_key!r}")

    articles = load_articles(run_name)
    if limit_articles is not None:
        articles = articles[:limit_articles]

    judgements_path = _module2_dir(run_name, model_key) / "module2_judgements.jsonl"
    if not judgements_path.exists():
        raise FileNotFoundError(f"Missing v2 Module 2 judgements: {judgements_path}")
    judgements = load_jsonl(judgements_path)
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    valid_af_by_article: dict[str, set[str]] = defaultdict(set)
    for row in judgements:
        aid = str(row.get("article_id"))
        by_article[aid].append(row)
        if row.get("af_id"):
            valid_af_by_article[aid].add(str(row["af_id"]))

    out_dir = _out_dir(run_name, model_key, mode)
    out_path = out_dir / "evidence_table.jsonl"
    prompt_path = PROMPTS_DIR / "build_evidence_table.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")

    usage = usage_tracker(run_name, resume=resume)
    client = ProviderClient(model_key, usage)
    existing = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("article_id")) for r in existing}

    def worker(article: dict[str, Any]) -> dict[str, Any]:
        aid = str(article["id"])
        max_per_sentence = 10 if mode == "balanced" else 7
        raw = ""
        parse_error = False
        validation_errors: list[str] = []
        rows: list[dict[str, Any]] = []
        try:
            prompt = (
                prompt_template.replace("{mode}", mode)
                .replace("{title}", str(article.get("title", "")))
                .replace("{abstract_sentences}", format_numbered_sentences(article.get("abstract_sentences") or []))
                .replace("{expert_summary}", str(article.get("expert_summary", "")))
                .replace("{candidate_facts}", _candidate_blocks(by_article.get(aid, []), max_per_sentence=max_per_sentence))
            )
            raw = client.chat(
                stage=f"V10.evidence_table.{mode}:{model_key}",
                item_id=aid,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = loads_json_object(raw)
            rows, validation_errors = _normalize_evidence_rows(
                obj=obj,
                article=article,
                valid_af_ids=valid_af_by_article.get(aid, set()),
                mode=mode,
            )
            if validation_errors or not rows:
                raise ValueError("; ".join(validation_errors) or "no evidence rows")
        except Exception as exc:
            parse_error = True
            validation_errors = validation_errors or [str(exc)]

        return {
            "article_id": aid,
            "source_dataset": article.get("source_dataset"),
            "original_index": article.get("original_index"),
            "mode": mode,
            "model_key": model_key,
            "evidence_rows": rows,
            "evidence_row_count": len(rows),
            "parse_error": parse_error,
            "validation_errors": validation_errors,
            "raw_response": raw if parse_error else None,
        }

    todo = [a for a in articles if str(a["id"]) not in done]
    new_rows = run_parallel(todo, worker, max_workers=max_workers, desc=f"V10 evidence | {mode} | {model_key}")
    all_rows = existing + new_rows
    all_rows.sort(key=lambda r: str(r["article_id"]))
    save_jsonl(all_rows, out_path)

    core_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    for article_row in all_rows:
        for erow in article_row.get("evidence_rows") or []:
            core_id = f"{erow['article_id']}_{erow['evidence_row_id']}_core"
            core = {
                "af_id": core_id,
                "article_id": erow["article_id"],
                "source_dataset": erow.get("source_dataset"),
                "original_index": erow.get("original_index"),
                "mode": mode,
                "evidence_row_id": erow["evidence_row_id"],
                "check_type": "core",
                "slot_type": "core_claim",
                "fact": erow["core_keep_af"] or erow["abstract_claim"],
                "source_span": " ".join(erow.get("allowed_evidence_spans") or []),
                "abstract_sentence_idx": erow.get("abstract_sentence_idx"),
                "abstract_sentence": erow.get("abstract_claim"),
            }
            core_rows.append(core)
            check_rows.append(core)
            for j, item in enumerate(erow.get("high_risk_check_facts") or [], start=1):
                check_rows.append(
                    {
                        "af_id": f"{erow['article_id']}_{erow['evidence_row_id']}_risk_{j:02d}",
                        "article_id": erow["article_id"],
                        "source_dataset": erow.get("source_dataset"),
                        "original_index": erow.get("original_index"),
                        "mode": mode,
                        "evidence_row_id": erow["evidence_row_id"],
                        "check_type": "high_risk",
                        "slot_type": item.get("slot_type", "entity"),
                        "fact": item["fact"],
                        "source_span": item.get("source_span") or " ".join(erow.get("allowed_evidence_spans") or []),
                        "abstract_sentence_idx": erow.get("abstract_sentence_idx"),
                        "abstract_sentence": erow.get("abstract_claim"),
                    }
                )

    save_jsonl(core_rows, out_dir / "core_keep_af.jsonl")
    save_jsonl(check_rows, out_dir / "check_af.jsonl")
    save_usage(run_name, usage)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "stage": "V10_02_5_evidence_table",
            "mode": mode,
            "model_key": model_key,
            "prompt_file": str(prompt_path),
            "prompt_sha256": sha256_text(prompt_template),
            "articles_total": len(articles),
            "parse_error_count": sum(1 for r in all_rows if r.get("parse_error")),
            "evidence_rows_total": sum(len(r.get("evidence_rows") or []) for r in all_rows),
            "core_keep_af_total": len(core_rows),
            "check_af_total": len(check_rows),
            "check_type_counts": dict(Counter(str(r.get("check_type")) for r in check_rows)),
            "usage_summary": usage.summarize(),
        },
        out_dir / "evidence_table_metadata.json",
    )
    print(f"[V10 evidence] {mode} {model_key} articles={len(all_rows)} rows={len(core_rows)} checks={len(check_rows)}")
    print(f"[V10 evidence] output -> {out_path}")
    return all_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V10: build evidence tables from v2 Module 2 judgements.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--mode", choices=sorted(["balanced", "factuality_chase"]), required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit-articles", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_evidence_tables(
        run_name=args.run_name,
        model_key=args.model_key,
        mode=args.mode,
        max_workers=args.max_workers,
        resume=not args.no_resume,
        limit_articles=args.limit_articles,
    )


if __name__ == "__main__":
    main()

