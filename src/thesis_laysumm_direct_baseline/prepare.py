from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiment_2.run_three_model_exp2 import MODEL_CONFIGS, UsageTracker
from src.thesis_laysumm.paths import run_data_dir as pipeline_run_data_dir
from src.thesis_laysumm_direct_baseline.paths import (
    inputs_path,
    references_path,
    run_data_dir,
    summaries_path,
)
from src.utils import load_jsonl, save_json, save_jsonl


DEFAULT_SOURCE_RUN = "v11_constellation_validation284_fixed_m12_pilot20"
DEFAULT_PILOT_SOURCE_RUN = "pilot_n20_v11_factuality_chase"
DEFAULT_MODELS = [
    "gemini25_flash_non_thinking",
    "gpt41_mini",
    "gemini3_flash_preview_minimal",
]

INPUT_KEYS = (
    "id",
    "source_dataset",
    "original_index",
    "split",
    "article",
)
REFERENCE_KEYS = (
    "id",
    "source_dataset",
    "original_index",
    "document",
    "expert_summary",
)


def _pipeline_articles(run_name: str) -> list[dict[str, Any]]:
    path = pipeline_run_data_dir(run_name) / "00_articles" / "articles.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing source articles: {path}")
    return load_jsonl(path)


def create_run(*, run_name: str, source_run: str, scope: str, pilot_source_run: str) -> None:
    source_rows = _pipeline_articles(source_run)
    source_by_id = {str(row["id"]): row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("Source run contains duplicate article IDs")

    if scope == "pilot20":
        pilot_rows = _pipeline_articles(pilot_source_run)
        selected_ids = [str(row["id"]) for row in pilot_rows]
        missing = [aid for aid in selected_ids if aid not in source_by_id]
        if missing:
            raise ValueError(f"Pilot articles are missing from source run: {missing}")
        selected = [source_by_id[aid] for aid in selected_ids]
    elif scope == "full284":
        selected = list(source_rows)
    else:
        raise ValueError(f"Unknown scope: {scope}")

    expected = 20 if scope == "pilot20" else 284
    if len(selected) != expected:
        raise ValueError(f"{scope} expected {expected} articles, found {len(selected)}")
    if any(str(row.get("split")) != "validation" for row in selected):
        raise ValueError("Direct baseline must use validation articles only")

    input_rows = [{key: row.get(key) for key in INPUT_KEYS} for row in selected]
    reference_rows = [{key: row.get(key) for key in REFERENCE_KEYS} for row in selected]
    if any(not str(row.get("article") or "").strip() for row in input_rows):
        raise ValueError("At least one selected article has empty raw article text")
    if any(not str(row.get("expert_summary") or "").strip() for row in reference_rows):
        raise ValueError("At least one selected article has an empty expert reference")

    save_jsonl(input_rows, inputs_path(run_name))
    save_jsonl(reference_rows, references_path(run_name))
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "design": "direct_zero_shot_full_article",
            "run_name": run_name,
            "scope": scope,
            "source_run": source_run,
            "pilot_source_run": pilot_source_run if scope == "pilot20" else None,
            "article_count": len(input_rows),
            "by_source": {
                source: sum(str(row.get("source_dataset")) == source for row in input_rows)
                for source in ("PLOS", "eLife")
            },
            "generation_visible_fields": ["article"],
            "reference_storage_separated": True,
            "forbidden_generation_inputs": [
                "expert_summary",
                "evidence",
                "questions",
                "answers",
                "module outputs",
                "pipeline summaries",
            ],
        },
        run_data_dir(run_name) / "run_manifest.json",
    )
    print(
        f"[direct prepare] {run_name}: scope={scope} articles={len(input_rows)} "
        f"PLOS={sum(r['source_dataset'] == 'PLOS' for r in input_rows)} "
        f"eLife={sum(r['source_dataset'] == 'eLife' for r in input_rows)}"
    )


def _load_usage_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def seed_pilot(*, pilot_run: str, full_run: str, models: list[str]) -> None:
    full_ids = {str(row["id"]) for row in load_jsonl(inputs_path(full_run))}
    pilot_ids = {str(row["id"]) for row in load_jsonl(inputs_path(pilot_run))}
    if len(full_ids) != 284 or len(pilot_ids) != 20 or not pilot_ids.issubset(full_ids):
        raise ValueError("Pilot/full input sets are not a valid 20-in-284 pairing")

    report: dict[str, Any] = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "pilot_run": pilot_run,
        "full_run": full_run,
        "models": {},
    }
    for model_key in models:
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {model_key}")
        src = summaries_path(pilot_run, model_key)
        if not src.exists():
            raise FileNotFoundError(f"Missing pilot summaries: {src}")
        pilot_rows = load_jsonl(src)
        row_ids = {str(row.get("article_id")) for row in pilot_rows}
        if row_ids != pilot_ids or len(pilot_rows) != 20:
            raise ValueError(f"Pilot output mismatch for {model_key}: rows={len(pilot_rows)}")

        dst = summaries_path(full_run, model_key)
        existing = load_jsonl(dst) if dst.exists() else []
        merged = {str(row["article_id"]): row for row in existing}
        for row in pilot_rows:
            merged[str(row["article_id"])] = row
        invalid = set(merged) - full_ids
        if invalid:
            raise ValueError(f"Seed produced IDs outside full run: {sorted(invalid)}")
        save_jsonl(list(merged.values()), dst)
        report["models"][model_key] = {"seeded": len(pilot_rows), "full_rows_after_seed": len(merged)}

    pilot_usage = _load_usage_rows(run_data_dir(pilot_run) / "api_usage_calls.csv")
    full_usage = _load_usage_rows(run_data_dir(full_run) / "api_usage_calls.csv")
    seen = {json.dumps(row, sort_keys=True, ensure_ascii=False) for row in full_usage}
    merged_usage = list(full_usage)
    for row in pilot_usage:
        key = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            merged_usage.append(row)
    if merged_usage:
        UsageTracker(rows=merged_usage).save(run_data_dir(full_run))
    report["usage_rows_seeded"] = len(merged_usage) - len(full_usage)
    save_json(report, run_data_dir(full_run) / "pilot_seed_report.json")
    print(f"[direct seed] {pilot_run} -> {full_run}: models={len(models)} pilot_articles=20")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare isolated direct-baseline datasets and pilot cache.")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create")
    create.add_argument("--run-name", required=True)
    create.add_argument("--scope", choices=["pilot20", "full284"], required=True)
    create.add_argument("--source-run", default=DEFAULT_SOURCE_RUN)
    create.add_argument("--pilot-source-run", default=DEFAULT_PILOT_SOURCE_RUN)

    seed = sub.add_parser("seed-pilot")
    seed.add_argument("--pilot-run", required=True)
    seed.add_argument("--full-run", required=True)
    seed.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "create":
        create_run(
            run_name=args.run_name,
            source_run=args.source_run,
            scope=args.scope,
            pilot_source_run=args.pilot_source_run,
        )
    else:
        seed_pilot(pilot_run=args.pilot_run, full_run=args.full_run, models=args.models)


if __name__ == "__main__":
    main()
