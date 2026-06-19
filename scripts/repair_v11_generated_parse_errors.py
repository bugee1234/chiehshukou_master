from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thesis_laysumm.llm_utils import sha256_text
from src.thesis_laysumm.paths import run_data_dir
from src.thesis_laysumm_v11.common import (
    PROMPTS_DIR,
    find_mojibake,
    length_policy,
    loads_json_object,
    readability_proxy_score,
    readability_stats,
    safe_word_count,
)
from src.thesis_laysumm_v11.run_phase_b import _fit_slots_to_length, _normalize_slots, _slots_to_sentences
from src.utils import load_jsonl, save_json, save_jsonl


def _summary_from_slots(slots: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for slot in slots:
        if not slot.get("used_for_summary", True):
            continue
        claim = str(slot.get("claim_sentence", "")).strip()
        clarification = str(slot.get("clarification_sentence", "")).strip()
        if claim:
            parts.append(claim)
        if clarification:
            parts.append(clarification)
    return " ".join(parts).strip()


def repair(*, run_name: str, model_key: str, mode: str) -> None:
    data_dir = run_data_dir(run_name)
    summaries_path = data_dir / "04_summaries" / model_key / "generated_summaries.jsonl"
    metadata_path = data_dir / "04_summaries" / model_key / "generated_summaries_metadata.json"
    evidence_path = data_dir / "02_5_evidence_table" / mode / model_key / "evidence_table.jsonl"
    articles_path = data_dir / "00_articles" / "articles.jsonl"
    prompt_path = PROMPTS_DIR / "generate_summary_v11.txt"

    rows = load_jsonl(summaries_path)
    evidence_by_article = {str(row["article_id"]): row for row in load_jsonl(evidence_path)}
    articles_by_id = {str(row["id"]): row for row in load_jsonl(articles_path)}
    repaired = 0
    still_bad: list[str] = []

    for row in rows:
        if not row.get("parse_error"):
            continue
        raw = str(row.get("raw_response") or "").strip()
        if not raw:
            still_bad.append(f"{row.get('article_id')}: missing raw_response")
            continue
        aid = str(row["article_id"])
        article = articles_by_id[aid]
        evidence = evidence_by_article[aid]
        valid_row_ids = {str(item["evidence_row_id"]) for item in evidence.get("evidence_rows") or []}
        policy = length_policy(
            mode=mode,
            expert_word_count=int(article.get("expert_summary_word_count") or 0),
            source_dataset=str(article.get("source_dataset", "")),
        )
        try:
            obj = loads_json_object(raw)
            slots, errors = _normalize_slots(obj, valid_row_ids)
            if errors:
                raise ValueError("; ".join(errors))
            slots, fit_notes = _fit_slots_to_length(slots, policy)
            summary = str(obj.get("summary", "")).strip() or _summary_from_slots(slots)
            slot_summary = _summary_from_slots(slots)
            if slot_summary:
                summary = slot_summary
            if not summary:
                raise ValueError("empty repaired summary")
        except Exception as exc:
            still_bad.append(f"{aid}: {exc}")
            continue

        wc = safe_word_count(summary)
        row.update(
            {
                "prompt_file": str(prompt_path),
                "prompt_sha256": sha256_text(prompt_path.read_text(encoding="utf-8")),
                "slots": slots,
                "sentences": _slots_to_sentences(slots),
                "generated_summary": summary,
                "generated_word_count": wc,
                "readability_stats": readability_stats(summary),
                "readability_proxy_score": readability_proxy_score(summary),
                "mojibake_warnings": find_mojibake(summary),
                "below_min_length": bool(summary) and wc < int(policy["min_word_count"]),
                "above_max_length": bool(summary) and wc > int(policy["max_word_count"]),
                "parse_error": False,
                "validation_errors": [],
                "length_fit_notes": list(row.get("length_fit_notes") or []) + fit_notes,
                "raw_response": None,
            }
        )
        repaired += 1

    save_jsonl(rows, summaries_path)
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "parse_error_count": sum(1 for row in rows if row.get("parse_error")),
            "below_min_length_count": sum(1 for row in rows if row.get("below_min_length")),
            "above_max_length_count": sum(1 for row in rows if row.get("above_max_length")),
            "offline_repair_repaired_count": repaired,
            "offline_repair_still_bad": still_bad,
        }
    )
    save_json(metadata, metadata_path)
    if still_bad:
        raise ValueError("Some rows could not be repaired:\n" + "\n".join(still_bad))
    print(f"[repair_v11_generated_parse_errors] repaired={repaired} path={summaries_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair V11 generated parse_error rows from saved raw_response.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--mode", default="factuality_chase", choices=["balanced", "factuality_chase"])
    args = parser.parse_args()
    repair(run_name=args.run_name, model_key=args.model_key, mode=args.mode)


if __name__ == "__main__":
    main()
