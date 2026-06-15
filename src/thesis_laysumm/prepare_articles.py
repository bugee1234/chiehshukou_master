from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.thesis_laysumm.biolaysumm_data import (
    DATASETS,
    EXPECTED_TEST_SIZE,
    THESIS_SAMPLE_PER_SOURCE,
    VALIDATION_POOL_SIZE,
    article_record_from_item,
    audit_dataset_split,
    load_biolaysumm_split,
    select_indices,
    split_abstract_sentences,
    validate_article_row,
    validation_to_dict,
)
from src.thesis_laysumm.paths import stage_data_dir
from src.utils import save_json, save_jsonl


def _expert_summary_column(column_names: list[str]) -> str | None:
    for name in ("lay_summary", "summary", "reference"):
        if name in column_names:
            return name
    return None


def _paper_alignment_note(*, split: str, n_per_source: int | None) -> str:
    if split == "validation":
        if n_per_source == THESIS_SAMPLE_PER_SOURCE:
            return (
                f"Thesis sample: random {THESIS_SAMPLE_PER_SOURCE} PLOS + "
                f"{THESIS_SAMPLE_PER_SOURCE} eLife from HuggingFace validation split."
            )
        if n_per_source is not None:
            return (
                f"Pilot/sample: random {n_per_source} per source from HuggingFace validation split."
            )
        return "Full HuggingFace validation split (all rows per source)."
    if split == "test" and n_per_source is None:
        return "BioLaySumm 2025 Task 1.1 official test split: 142 PLOS + 142 eLife."
    return f"BioLaySumm {split} split with n_per_source={n_per_source}."


def prepare_articles(
    *,
    run_name: str,
    split: str,
    n_per_source: int | None,
    selection_mode: str,
    seed: int,
    audit_full_split: bool,
    require_expert_summary: bool,
) -> None:
    out_dir = stage_data_dir(run_name, "00_articles")
    out_articles = out_dir / "articles.jsonl"
    out_meta = out_dir / "selection_metadata.json"
    out_validation = out_dir / "article_validation.json"
    out_truth_plos = out_dir / "PLOS_eval_references.jsonl"
    out_truth_elife = out_dir / "eLife_eval_references.jsonl"

    rows: list[dict[str, Any]] = []
    truth_by_source: dict[str, list[dict[str, str]]] = {"PLOS": [], "eLife": []}
    selected_validation: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "pipeline": "thesis_laysumm",
        "run_name": run_name,
        "split": split,
        "seed": seed,
        "n_per_source": n_per_source,
        "selection_mode": selection_mode,
        "require_expert_summary": require_expert_summary,
        "thesis_sample_per_source": THESIS_SAMPLE_PER_SOURCE,
        "validation_pool_size": VALIDATION_POOL_SIZE,
        "paper_alignment": _paper_alignment_note(split=split, n_per_source=n_per_source),
        "datasets": {},
    }
    full_split_audit: dict[str, Any] = {}

    run_order = 0
    for source, (dataset_name, prefix) in DATASETS.items():
        if split == "validation" and source in VALIDATION_POOL_SIZE:
            expected_pool = VALIDATION_POOL_SIZE[source]
            if n_per_source is not None and n_per_source > expected_pool:
                raise RuntimeError(
                    f"{source}: requested {n_per_source} articles but validation pool has only {expected_pool}"
                )

        if audit_full_split:
            full_split_audit[source] = audit_dataset_split(
                source=source,
                split=split,
                require_expert_summary=require_expert_summary,
            )

        ds_obj = load_biolaysumm_split(source, split)
        expert_col = _expert_summary_column(list(ds_obj.column_names))
        split_audit = full_split_audit.get(source)
        if split_audit is None:
            split_audit = audit_dataset_split(
                source=source,
                split=split,
                require_expert_summary=require_expert_summary,
            )

        indices = select_indices(
            len(ds_obj),
            n_per_source,
            selection_mode=selection_mode,
            seed=seed,
        )

        meta["datasets"][source] = {
            "dataset_name": dataset_name,
            "split_size": len(ds_obj),
            "selected_count": len(indices),
            "selected_indices": indices,
            "columns": ds_obj.column_names,
            "expert_summary_column": expert_col,
            "full_split_ok_count": split_audit["ok_count"],
            "full_split_error_count": split_audit["error_count"],
            "full_split_empty_expert_summary_count": split_audit["empty_expert_summary_count"],
        }

        for idx in indices:
            item = ds_obj[int(idx)]
            validation = validate_article_row(
                source_dataset=source,
                original_index=idx,
                item=item,
                require_expert_summary=require_expert_summary,
            )
            selected_validation.append(validation_to_dict(validation))
            if not validation.ok:
                raise RuntimeError(
                    f"{source}[{idx}] failed validation: {', '.join(validation.errors)}"
                )

            run_order += 1
            article_id = f"laysumm_{prefix}_{idx:04d}"
            record = article_record_from_item(
                source_dataset=source,
                dataset_name=dataset_name,
                split=split,
                original_index=idx,
                item=item,
                run_order=run_order,
                article_id=article_id,
            )
            record["abstract_sentences"] = split_abstract_sentences(record["abstract"])
            record["abstract_sentence_count"] = len(record["abstract_sentences"])
            rows.append(record)
            truth_by_source[source].append(
                {
                    "document": record["document"],
                    "reference": record["expert_summary"],
                }
            )

    save_jsonl(rows, out_articles)
    save_jsonl(truth_by_source["PLOS"], out_truth_plos)
    save_jsonl(truth_by_source["eLife"], out_truth_elife)

    meta["total_articles"] = len(rows)
    meta["by_source"] = dict(Counter(r["source_dataset"] for r in rows))
    meta["empty_expert_summary_count"] = sum(
        1 for r in rows if not str(r.get("expert_summary", "")).strip()
    )
    meta["abstract_sentence_stats"] = {
        "min": min(r["abstract_sentence_count"] for r in rows),
        "max": max(r["abstract_sentence_count"] for r in rows),
        "mean": round(
            sum(r["abstract_sentence_count"] for r in rows) / len(rows),
            2,
        ),
    }
    meta["expert_summary_word_stats"] = {
        "min": min(r["expert_summary_word_count"] for r in rows),
        "max": max(r["expert_summary_word_count"] for r in rows),
        "mean": round(
            sum(r["expert_summary_word_count"] for r in rows) / len(rows),
            2,
        ),
    }
    if meta["empty_expert_summary_count"]:
        meta["reference_note"] = (
            "Some selected rows have empty expert summaries. "
            "For thesis validation sampling, pass --require-expert-summary (default on validation split)."
        )

    save_json(meta, out_meta)
    save_json(
        {
            "selected_articles": selected_validation,
            "full_split_audit": full_split_audit if audit_full_split else None,
        },
        out_validation,
    )

    print(f"[prepare_articles] wrote {len(rows)} articles -> {out_articles}")
    print(f"[prepare_articles] metadata -> {out_meta}")
    if meta["empty_expert_summary_count"]:
        print(
            f"[prepare_articles] WARNING: {meta['empty_expert_summary_count']} articles have empty expert_summary"
        )
    if split == "test" and n_per_source is None:
        expected_total = EXPECTED_TEST_SIZE * len(DATASETS)
        if len(rows) != expected_total:
            raise RuntimeError(f"expected {expected_total} articles, got {len(rows)}")
    if split == "validation" and n_per_source == THESIS_SAMPLE_PER_SOURCE:
        expected_total = THESIS_SAMPLE_PER_SOURCE * len(DATASETS)
        if len(rows) != expected_total:
            raise RuntimeError(f"expected {expected_total} articles, got {len(rows)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 0 for thesis lay-summary pipeline: validate and select BioLaySumm articles."
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="HuggingFace split to sample from. Thesis experiments use validation.",
    )
    parser.add_argument(
        "--n-per-source",
        type=int,
        default=10,
        help=(
            "Articles per source. Use 10 for the 20-article pilot; "
            f"use {THESIS_SAMPLE_PER_SOURCE} for the full thesis sample."
        ),
    )
    parser.add_argument(
        "--full-split",
        action="store_true",
        help="Select every row in the split (not the usual thesis 142+142 sample).",
    )
    parser.add_argument(
        "--full-test",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="random",
        choices=["first", "random"],
        help="How to pick articles when n-per-source is set.",
    )
    parser.add_argument(
        "--require-expert-summary",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require non-empty expert summary on every selected row (default: on for validation).",
    )
    parser.add_argument(
        "--skip-full-split-audit",
        action="store_true",
        help="Skip scanning the entire split before selecting articles.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    full_split = args.full_split or args.full_test
    n_per_source = None if full_split else args.n_per_source
    require_expert_summary = (
        args.split == "validation" if args.require_expert_summary is None else args.require_expert_summary
    )
    prepare_articles(
        run_name=args.run_name,
        split=args.split,
        n_per_source=n_per_source,
        selection_mode=args.selection_mode,
        seed=args.seed,
        audit_full_split=not args.skip_full_split_audit,
        require_expert_summary=require_expert_summary,
    )


if __name__ == "__main__":
    main()
