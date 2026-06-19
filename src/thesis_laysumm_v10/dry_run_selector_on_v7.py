from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.thesis_laysumm.paths import run_data_dir
from src.thesis_laysumm_v10.common import length_policy
from src.thesis_laysumm_v10.run_phase_b import _estimate_repair_wrong_count, _select_final_candidate
from src.utils import load_jsonl


def _slots_for(row: dict[str, Any], variant: str) -> list[dict[str, Any]]:
    slots = row.get("slots")
    if isinstance(slots, list):
        return slots
    candidates = row.get("candidate_summaries")
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("variant") == variant and isinstance(candidate.get("slots"), list):
                return candidate["slots"]
    return []


def dry_run(run_name: str, model_key: str, mode: str) -> None:
    path = run_data_dir(run_name) / "06_rewritten" / model_key / "rewritten_summaries.jsonl"
    rows = load_jsonl(path)
    old_counts = Counter(str(row.get("selected_variant")) for row in rows)
    new_counts: Counter[str] = Counter()
    changes: list[tuple[str, str, str, str]] = []

    for row in rows:
        wrong_count = int(row.get("wrong_feedback_count") or 0)
        policy = length_policy(
            mode=mode,
            expert_word_count=int(row.get("expert_summary_word_count") or 0),
            source_dataset=str(row.get("source_dataset", "")),
        )
        generated = str(row.get("generated_summary", "")).strip()
        expanded = str(row.get("expanded_generated_summary") or generated).strip()
        repair = str(row.get("rewritten_summary", "")).strip()
        edits = row.get("edits") if isinstance(row.get("edits"), list) else []
        repair_wrong_count_proxy = _estimate_repair_wrong_count(wrong_count, edits, bool(row.get("parse_error")))
        candidates = [
            {"variant": "generated", "summary": generated, "slots": _slots_for(row, "generated")},
            {"variant": "expanded_generated", "summary": expanded, "slots": _slots_for(row, "expanded_generated")},
        ]
        if row.get("rewrite_invoked"):
            candidates.append({"variant": "factual_repair", "summary": repair, "slots": _slots_for(row, "factual_repair")})
        selected = _select_final_candidate(
            candidates,
            wrong_count=wrong_count,
            repair_wrong_count_proxy=repair_wrong_count_proxy,
            policy=policy,
        )
        new_variant = str(selected.get("variant"))
        old_variant = str(row.get("selected_variant"))
        new_counts[new_variant] += 1
        if new_variant != old_variant:
            changes.append((str(row.get("article_id")), old_variant, new_variant, str(selected.get("selection_reason"))))

    print(f"v7 selector counts: {dict(old_counts)}")
    print(f"V10 selector counts on v7 rows: {dict(new_counts)}")
    print(f"changed selections: {len(changes)}/{len(rows)}")
    for aid, old_variant, new_variant, reason in changes[:50]:
        print(f"- {aid}: {old_variant} -> {new_variant} ({reason})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run the V10 selector on completed V7 rewritten rows.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--mode", default="factuality_chase")
    args = parser.parse_args()
    dry_run(args.run_name, args.model_key, args.mode)


if __name__ == "__main__":
    main()
