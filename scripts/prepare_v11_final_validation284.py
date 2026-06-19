from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets import Dataset

from src.thesis_laysumm.biolaysumm_data import (
    DATASETS,
    THESIS_SAMPLE_PER_SOURCE,
    article_record_from_item,
    split_abstract_sentences,
    validate_article_row,
    validation_to_dict,
)
from src.thesis_laysumm.paths import run_data_dir
from src.utils import load_jsonl, save_json, save_jsonl

MODE = "factuality_chase"
FIXED_MODEL = "gemini25_flash_non_thinking"
PHASE_MODELS = ["gemini25_flash_non_thinking", "gpt41_mini", "gemini3_flash_preview_minimal"]
DEFAULT_FINAL_RUN = "v11_constellation_validation284_fixed_m12_pilot20"
DEFAULT_PILOT_RUN = "pilot_n20_v11_factuality_chase"
DEFAULT_FIXED_PHASE_RUN = "pilot_n20_v11_fixed_m12_gemini25_same_model_phaseb_factuality_chase"


def _prefix_for_source(source: str) -> str:
    return DATASETS[source][1]


def _article_id(source: str, original_index: int) -> str:
    return f"laysumm_{_prefix_for_source(source)}_{int(original_index):04d}"


def _load_validation_split(source: str) -> Dataset:
    """Load validation directly from local HF arrow cache.

    `datasets.load_dataset()` can spend minutes trying Hub resolution on this
    Windows setup even when the cache exists. The final run must use validation
    only, so reading the cached validation arrow directly is both stricter and
    faster.
    """
    dataset_name, prefix = DATASETS[source]
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets"
    patterns = [
        f"BioLaySumm___bio_lay_summ2025-{prefix}/default/0.0.0/*/bio_lay_summ2025-{prefix}-validation.arrow",
        f"BioLaySumm___bio_lay_summ2025-{prefix.replace('elife', 'e_life')}/default/0.0.0/*/bio_lay_summ2025-{prefix.replace('elife', 'e_life')}-validation.arrow",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(cache_root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"Missing local validation arrow cache for {dataset_name}. "
            "Run the existing prepare_articles once with validation split to populate HuggingFace cache."
        )
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    return Dataset.from_file(str(path))


def _load_pilot_indices(pilot_run: str) -> dict[str, set[int]]:
    path = run_data_dir(pilot_run) / "00_articles" / "articles.jsonl"
    rows = load_jsonl(path)
    out: dict[str, set[int]] = {"PLOS": set(), "eLife": set()}
    for row in rows:
        if row.get("split") != "validation":
            raise RuntimeError(f"Pilot row is not validation split: {row.get('id')} split={row.get('split')}")
        source = str(row["source_dataset"])
        out[source].add(int(row["original_index"]))
    for source, indices in out.items():
        if len(indices) != 10:
            raise RuntimeError(f"Expected 10 pilot {source} indices, found {len(indices)}")
    return out


def _quantile_bins(values: list[int], bin_count: int) -> list[list[int]]:
    ordered = sorted(range(len(values)), key=lambda i: (values[i], i))
    bins: list[list[int]] = []
    for b in range(bin_count):
        start = round(b * len(ordered) / bin_count)
        end = round((b + 1) * len(ordered) / bin_count)
        bins.append(ordered[start:end])
    return bins


def _allocate_counts(pool_bin_sizes: list[int], total: int, mandatory_by_bin: list[int]) -> list[int]:
    pool_total = sum(pool_bin_sizes)
    if pool_total < total:
        raise RuntimeError(f"Pool has only {pool_total} rows, need {total}")
    raw = [total * size / pool_total for size in pool_bin_sizes]
    counts = [int(x) for x in raw]
    while sum(counts) < total:
        remainders = [raw[i] - counts[i] for i in range(len(counts))]
        for i in sorted(range(len(counts)), key=lambda j: (-remainders[j], j)):
            if counts[i] < pool_bin_sizes[i]:
                counts[i] += 1
                break
    for i, mandatory in enumerate(mandatory_by_bin):
        counts[i] = max(counts[i], mandatory)
    while sum(counts) > total:
        candidates = [
            i
            for i in range(len(counts))
            if counts[i] > mandatory_by_bin[i]
        ]
        if not candidates:
            raise RuntimeError("Mandatory rows exceed requested total")
        i = max(candidates, key=lambda j: (counts[j] - raw[j], counts[j], -j))
        counts[i] -= 1
    return counts


def _select_stratified_indices(
    *,
    source: str,
    target_n: int,
    seed: int,
    mandatory_indices: set[int],
    bin_count: int,
) -> tuple[list[int], dict[str, Any]]:
    ds = _load_validation_split(source)
    validations = []
    valid_indices = []
    lengths = []
    for idx in range(len(ds)):
        validation = validate_article_row(
            source_dataset=source,
            original_index=idx,
            item=ds[int(idx)],
            require_expert_summary=True,
        )
        validations.append(validation_to_dict(validation))
        if validation.ok:
            valid_indices.append(idx)
            lengths.append(int(validation.expert_summary_word_count))
    valid_set = set(valid_indices)
    missing_mandatory = sorted(mandatory_indices - valid_set)
    if missing_mandatory:
        raise RuntimeError(f"{source} mandatory pilot indices failed validation: {missing_mandatory}")

    bins_local = _quantile_bins(lengths, bin_count)
    global_by_bin: list[list[int]] = [[valid_indices[i] for i in local_bin] for local_bin in bins_local]
    bin_for_index: dict[int, int] = {}
    for b, indices in enumerate(global_by_bin):
        for idx in indices:
            bin_for_index[idx] = b
    mandatory_by_bin = [0 for _ in range(bin_count)]
    for idx in mandatory_indices:
        mandatory_by_bin[bin_for_index[idx]] += 1

    counts = _allocate_counts([len(x) for x in global_by_bin], target_n, mandatory_by_bin)
    rng = random.Random(seed + (17 if source == "PLOS" else 31))
    selected: set[int] = set(mandatory_indices)
    bin_details = []
    for b, indices in enumerate(global_by_bin):
        need = counts[b] - sum(1 for idx in mandatory_indices if idx in set(indices))
        available = [idx for idx in indices if idx not in mandatory_indices]
        if need < 0:
            raise RuntimeError(f"{source} bin {b} has more mandatory rows than allocated")
        if need > len(available):
            raise RuntimeError(f"{source} bin {b} needs {need} rows, only {len(available)} available")
        picked = sorted(rng.sample(available, need))
        selected.update(picked)
        bin_lengths = [lengths[valid_indices.index(idx)] for idx in indices]
        bin_details.append(
            {
                "bin": b + 1,
                "pool_count": len(indices),
                "target_count": counts[b],
                "mandatory_count": mandatory_by_bin[b],
                "selected_count": counts[b],
                "expert_summary_word_min": min(bin_lengths),
                "expert_summary_word_max": max(bin_lengths),
            }
        )
    selected_sorted = sorted(selected)
    if len(selected_sorted) != target_n:
        raise RuntimeError(f"{source}: selected {len(selected_sorted)}, expected {target_n}")
    metadata = {
        "source_dataset": source,
        "split_size": len(ds),
        "valid_count": len(valid_indices),
        "target_count": target_n,
        "mandatory_indices": sorted(mandatory_indices),
        "selected_indices": selected_sorted,
        "bin_count": bin_count,
        "bins": bin_details,
        "validation_error_count": len(ds) - len(valid_indices),
        "validations": validations,
    }
    return selected_sorted, metadata


def _write_articles(final_run: str, selected_by_source: dict[str, list[int]], selection_meta: dict[str, Any]) -> None:
    articles = []
    validations = []
    truth_by_source: dict[str, list[dict[str, str]]] = {"PLOS": [], "eLife": []}
    run_order = 0
    for source in ("PLOS", "eLife"):
        ds = _load_validation_split(source)
        dataset_name, _ = DATASETS[source]
        for idx in selected_by_source[source]:
            item = ds[int(idx)]
            validation = validate_article_row(
                source_dataset=source,
                original_index=idx,
                item=item,
                require_expert_summary=True,
            )
            validations.append(validation_to_dict(validation))
            if not validation.ok:
                raise RuntimeError(f"{source}[{idx}] failed validation: {validation.errors}")
            run_order += 1
            record = article_record_from_item(
                source_dataset=source,
                dataset_name=dataset_name,
                split="validation",
                original_index=idx,
                item=item,
                run_order=run_order,
                article_id=_article_id(source, idx),
            )
            record["abstract_sentences"] = split_abstract_sentences(record["abstract"])
            record["abstract_sentence_count"] = len(record["abstract_sentences"])
            articles.append(record)
            truth_by_source[source].append({"document": record["document"], "reference": record["expert_summary"]})

    ids = [str(r["id"]) for r in articles]
    if len(ids) != len(set(ids)):
        duplicates = [aid for aid, count in Counter(ids).items() if count > 1]
        raise RuntimeError(f"Duplicate article IDs: {duplicates[:10]}")
    by_source = Counter(str(r["source_dataset"]) for r in articles)
    if by_source != {"PLOS": THESIS_SAMPLE_PER_SOURCE, "eLife": THESIS_SAMPLE_PER_SOURCE}:
        raise RuntimeError(f"Unexpected source counts: {dict(by_source)}")
    if any(not str(r.get("expert_summary", "")).strip() for r in articles):
        raise RuntimeError("At least one selected article has empty expert_summary")

    out_dir = run_data_dir(final_run) / "00_articles"
    save_jsonl(articles, out_dir / "articles.jsonl")
    save_jsonl(truth_by_source["PLOS"], out_dir / "PLOS_eval_references.jsonl")
    save_jsonl(truth_by_source["eLife"], out_dir / "eLife_eval_references.jsonl")
    save_json(
        {
            **selection_meta,
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": final_run,
            "split": "validation",
            "total_articles": len(articles),
            "by_source": dict(by_source),
            "empty_expert_summary_count": 0,
            "article_id_unique": True,
            "expert_summary_word_stats": {
                source: {
                    "min": min(int(r["expert_summary_word_count"]) for r in articles if r["source_dataset"] == source),
                    "max": max(int(r["expert_summary_word_count"]) for r in articles if r["source_dataset"] == source),
                    "mean": round(
                        sum(int(r["expert_summary_word_count"]) for r in articles if r["source_dataset"] == source)
                        / THESIS_SAMPLE_PER_SOURCE,
                        2,
                    ),
                }
                for source in ("PLOS", "eLife")
            },
        },
        out_dir / "selection_metadata.json",
    )
    save_json({"selected_articles": validations}, out_dir / "article_validation.json")


def _filter_jsonl_by_article(src: Path, dst: Path, pilot_ids: set[str]) -> int:
    if not src.exists():
        raise FileNotFoundError(src)
    rows = load_jsonl(src)
    filtered = [
        row for row in rows
        if str(row.get("article_id") or row.get("id")) in pilot_ids
    ]
    save_jsonl(filtered, dst)
    return len(filtered)


def _copy_seed_file(src: Path, dst: Path, pilot_ids: set[str]) -> int:
    if src.suffix == ".jsonl":
        return _filter_jsonl_by_article(src, dst, pilot_ids)
    return 0


def _seed_directory(src_dir: Path, dst_dir: Path, pilot_ids: set[str], patterns: list[str]) -> dict[str, int]:
    counts = {}
    dst_dir.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for src in src_dir.glob(pattern):
            if src.name.endswith("_metadata.json") or src.name == "selection_metadata.json":
                continue
            dst = dst_dir / src.name
            counts[src.name] = _copy_seed_file(src, dst, pilot_ids)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "seeded_from": str(src_dir),
            "pilot_article_count": len(pilot_ids),
            "seeded_row_counts": counts,
            "note": "Partial pilot cache for resume; stage metadata is regenerated by the pipeline.",
        },
        dst_dir / "cache_seed_metadata.json",
    )
    return counts


def seed_pilot_cache(final_run: str, pilot_run: str, fixed_phase_run: str) -> dict[str, Any]:
    final_articles = load_jsonl(run_data_dir(final_run) / "00_articles" / "articles.jsonl")
    final_ids = {str(r["id"]) for r in final_articles}
    pilot_rows = load_jsonl(run_data_dir(pilot_run) / "00_articles" / "articles.jsonl")
    pilot_ids = {str(r["id"]) for r in pilot_rows}
    missing = sorted(pilot_ids - final_ids)
    if missing:
        raise RuntimeError(f"Pilot IDs missing from final article set: {missing}")

    final_dir = run_data_dir(final_run)
    seed_report: dict[str, Any] = {"pilot_article_count": len(pilot_ids), "stages": {}}

    seed_report["stages"]["module1"] = _seed_directory(
        run_data_dir(pilot_run) / "01_module1" / FIXED_MODEL,
        final_dir / "01_module1" / FIXED_MODEL,
        pilot_ids,
        ["*.jsonl"],
    )
    seed_report["stages"]["module2"] = _seed_directory(
        run_data_dir(pilot_run) / "02_module2" / FIXED_MODEL,
        final_dir / "02_module2" / FIXED_MODEL,
        pilot_ids,
        ["*.jsonl"],
    )

    for model in PHASE_MODELS:
        seed_report["stages"][f"evidence:{model}"] = _seed_directory(
            run_data_dir(fixed_phase_run) / "02_5_evidence_table" / MODE / model,
            final_dir / "02_5_evidence_table" / MODE / model,
            pilot_ids,
            ["*.jsonl"],
        )
        seed_report["stages"][f"questions:{model}"] = _seed_directory(
            run_data_dir(fixed_phase_run) / "03_questions_v11" / MODE / model,
            final_dir / "03_questions_v11" / MODE / model,
            pilot_ids,
            ["*.jsonl"],
        )
        seed_report["stages"][f"summaries:{model}"] = _seed_directory(
            run_data_dir(fixed_phase_run) / "04_summaries" / model,
            final_dir / "04_summaries" / model,
            pilot_ids,
            ["*.jsonl"],
        )
        seed_report["stages"][f"answers:{model}"] = _seed_directory(
            run_data_dir(fixed_phase_run) / "05_module3_answers_v11" / MODE / model,
            final_dir / "05_module3_answers_v11" / MODE / model,
            pilot_ids,
            ["*.jsonl"],
        )
        seed_report["stages"][f"rewritten:{model}"] = _seed_directory(
            run_data_dir(fixed_phase_run) / "06_rewritten" / model,
            final_dir / "06_rewritten" / model,
            pilot_ids,
            ["*.jsonl"],
        )

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "final_run": final_run,
            "pilot_run": pilot_run,
            "fixed_phase_run": fixed_phase_run,
            **seed_report,
        },
        final_dir / "00_articles" / "pilot_cache_seed_report.json",
    )
    return seed_report


def sync_fixed_cache(final_run: str) -> None:
    final_dir = run_data_dir(final_run)
    fixed_evidence = final_dir / "02_5_evidence_table" / MODE / FIXED_MODEL
    fixed_questions = final_dir / "03_questions_v11" / MODE / FIXED_MODEL
    for required in (fixed_evidence / "evidence_table.jsonl", fixed_questions / "questions_1t3f_nota.jsonl"):
        if not required.exists():
            raise FileNotFoundError(f"Run fixed upstream first; missing {required}")
    for model in PHASE_MODELS:
        if model == FIXED_MODEL:
            continue
        dst_evidence = final_dir / "02_5_evidence_table" / MODE / model
        dst_questions = final_dir / "03_questions_v11" / MODE / model
        if dst_evidence.exists():
            shutil.rmtree(dst_evidence)
        if dst_questions.exists():
            shutil.rmtree(dst_questions)
        shutil.copytree(fixed_evidence, dst_evidence)
        shutil.copytree(fixed_questions, dst_questions)
        save_json(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "source_model": FIXED_MODEL,
                "target_model": model,
                "note": "Full fixed upstream evidence/questions copied for phase-B comparison.",
            },
            dst_evidence / "fixed_cache_sync_metadata.json",
        )
        save_json(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "source_model": FIXED_MODEL,
                "target_model": model,
                "note": "Full fixed upstream evidence/questions copied for phase-B comparison.",
            },
            dst_questions / "fixed_cache_sync_metadata.json",
        )
    print(f"[sync-fixed-cache] copied {FIXED_MODEL} evidence/questions to phase models")


def prepare(args: argparse.Namespace) -> None:
    out_dir = run_data_dir(args.final_run) / "00_articles"
    articles_path = out_dir / "articles.jsonl"
    if articles_path.exists() and not args.overwrite_articles:
        raise RuntimeError(f"{articles_path} already exists. Use --overwrite-articles only if you intend to rebuild it.")
    if out_dir.exists() and args.overwrite_articles:
        shutil.rmtree(out_dir)

    pilot_indices = _load_pilot_indices(args.pilot_run)
    selected_by_source = {}
    source_meta = {}
    for source in ("PLOS", "eLife"):
        selected, meta = _select_stratified_indices(
            source=source,
            target_n=THESIS_SAMPLE_PER_SOURCE,
            seed=args.seed,
            mandatory_indices=pilot_indices[source],
            bin_count=args.bin_count,
        )
        selected_by_source[source] = selected
        source_meta[source] = meta
    selection_meta = {
        "pipeline": "thesis_laysumm_v11",
        "sample_title": "V11 Constellation Validation-284 fixed-upstream final sample",
        "selection_mode": "stratified_by_expert_summary_word_count",
        "seed": args.seed,
        "target_per_source": THESIS_SAMPLE_PER_SOURCE,
        "forced_include_pilot_run": args.pilot_run,
        "fixed_upstream_model": FIXED_MODEL,
        "phase_b_models": PHASE_MODELS,
        "datasets": source_meta,
    }
    _write_articles(args.final_run, selected_by_source, selection_meta)
    report = seed_pilot_cache(args.final_run, args.pilot_run, args.fixed_phase_run)
    print(f"[prepare] final run: {args.final_run}")
    print("[prepare] articles: 284 total, 142 PLOS, 142 eLife, validation split, expert summaries required")
    print(f"[prepare] seeded pilot cache rows: {report['stages']}")


def audit(args: argparse.Namespace) -> None:
    run_dir = run_data_dir(args.final_run)
    articles = load_jsonl(run_dir / "00_articles" / "articles.jsonl")
    ids = [str(r["id"]) for r in articles]
    by_source = Counter(str(r["source_dataset"]) for r in articles)
    if len(articles) != 284:
        raise RuntimeError(f"Expected 284 articles, found {len(articles)}")
    if by_source != {"PLOS": 142, "eLife": 142}:
        raise RuntimeError(f"Expected 142+142, found {dict(by_source)}")
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate article IDs")
    if {str(r["split"]) for r in articles} != {"validation"}:
        raise RuntimeError("Non-validation split found")
    if any(not str(r.get("expert_summary", "")).strip() for r in articles):
        raise RuntimeError("Empty expert summary found")
    pilot_ids = {str(r["id"]) for r in load_jsonl(run_data_dir(args.pilot_run) / "00_articles" / "articles.jsonl")}
    missing = sorted(pilot_ids - set(ids))
    if missing:
        raise RuntimeError(f"Pilot IDs missing: {missing}")
    print(f"[audit] ok {args.final_run}: 284 validation articles, pilot20 included, expert summaries non-empty")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare/audit V11 final validation-284 run with pilot cache reuse.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--final-run", default=DEFAULT_FINAL_RUN)
    p.add_argument("--pilot-run", default=DEFAULT_PILOT_RUN)
    p.add_argument("--fixed-phase-run", default=DEFAULT_FIXED_PHASE_RUN)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bin-count", type=int, default=4)
    p.add_argument("--overwrite-articles", action="store_true")
    a = sub.add_parser("audit")
    a.add_argument("--final-run", default=DEFAULT_FINAL_RUN)
    a.add_argument("--pilot-run", default=DEFAULT_PILOT_RUN)
    s = sub.add_parser("sync-fixed-cache")
    s.add_argument("--final-run", default=DEFAULT_FINAL_RUN)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "audit":
        audit(args)
    elif args.cmd == "sync-fixed-cache":
        sync_fixed_cache(args.final_run)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
