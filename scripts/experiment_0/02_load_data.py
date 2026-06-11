from __future__ import annotations

import json
import random
import statistics
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

from datasets import load_dataset


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.utils import save_json, save_jsonl, setup_logger


DATASET_SPECS = [
    ("PLOS", "BioLaySumm/BioLaySumm2025-PLOS", "plos"),
    ("eLife", "BioLaySumm/BioLaySumm2025-eLife", "elife"),
]


def _word_count(text: Any) -> int:
    if text is None:
        return 0
    return len(str(text).split())


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_to_original = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    for candidate in candidates:
        for col in columns:
            if candidate in col.lower():
                return col
    return None


def _series_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0}
    return {
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": int(min(values)),
        "max": int(max(values)),
    }


def _preview_value(value: Any, max_chars: int = 200) -> str:
    if isinstance(value, (dict, list)):
        content = json.dumps(value, ensure_ascii=False)
    else:
        content = str(value)
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "...(truncated)"


def _load_validation_split(dataset_name: str, hf_token: str | None):
    kwargs: dict[str, Any] = {"split": "validation"}
    if hf_token:
        kwargs["token"] = hf_token
    try:
        return load_dataset(dataset_name, **kwargs)
    except TypeError:
        if "token" in kwargs:
            kwargs.pop("token")
            kwargs["use_auth_token"] = hf_token
        return load_dataset(dataset_name, **kwargs)


def inspect_dataset(source_name: str, dataset_name: str, logger):
    logger.info("Loading dataset: %s (%s, split=validation)", source_name, dataset_name)
    try:
        ds = _load_validation_split(dataset_name, config.HF_TOKEN or None)
    except Exception as exc:
        logger.exception("Failed to load dataset %s", dataset_name)
        print(f"[ERROR] Failed to load {dataset_name} validation split: {exc}")
        print("Hint: check internet, dataset access permissions, and HF login/token.")
        raise

    columns = list(ds.column_names)
    total = len(ds)
    first_record = ds[0] if total > 0 else {}

    article_col = _pick_column(
        columns, ["article", "full_text", "fulltext", "body", "text", "content"]
    )
    lay_col = _pick_column(
        columns,
        [
            "lay_summary",
            "summary",
            "plain_language_summary",
            "plain_english_summary",
            "laysummary",
            "lay",
        ],
    )
    abstract_col = _pick_column(columns, ["abstract", "paper_abstract", "article_abstract"])

    if lay_col is None:
        raise ValueError(
            f"Cannot identify lay summary column for {source_name}. Columns: {columns}"
        )
    if article_col is None:
        raise ValueError(f"Cannot identify article column for {source_name}. Columns: {columns}")

    lay_lengths = [_word_count(row.get(lay_col, "")) for row in ds]
    lay_stats = _series_stats(lay_lengths)

    print("\n" + "=" * 72)
    print(f"[Schema Inspection] {source_name} ({dataset_name})")
    print(f"Total records (validation): {total}")
    print(f"Columns: {columns}")
    print("First record previews (first 200 chars):")
    if first_record:
        for col in columns:
            print(f"- {col}: {_preview_value(first_record.get(col))}")
    print(
        "Lay summary length stats (words): "
        f"mean={lay_stats['mean']}, median={lay_stats['median']}, "
        f"min={lay_stats['min']}, max={lay_stats['max']}"
    )
    print(
        f"Resolved mapping => article: '{article_col}', lay_summary: '{lay_col}', "
        f"abstract: '{abstract_col}'"
    )

    logger.info(
        "%s loaded | total=%s | columns=%s | mapping(article=%s, lay=%s, abstract=%s)",
        source_name,
        total,
        columns,
        article_col,
        lay_col,
        abstract_col,
    )

    return ds, {
        "article": article_col,
        "lay_summary": lay_col,
        "abstract": abstract_col,
    }, lay_stats


def sample_records(
    ds,
    source_name: str,
    id_prefix: str,
    mapping: dict[str, str | None],
    sample_size: int = 20,
) -> tuple[list[dict[str, Any]], list[int]]:
    total = len(ds)
    if total < sample_size:
        raise ValueError(f"{source_name} validation size {total} < requested sample size {sample_size}")

    indices = random.sample(range(total), sample_size)
    sampled: list[dict[str, Any]] = []
    article_col = mapping["article"]
    lay_col = mapping["lay_summary"]
    abstract_col = mapping["abstract"]

    for i, idx in enumerate(indices, start=1):
        row = ds[idx]
        article = row.get(article_col, "") if article_col else ""
        lay_summary = row.get(lay_col, "") if lay_col else ""
        abstract = row.get(abstract_col, "") if abstract_col else ""

        sampled.append(
            {
                "id": f"{id_prefix}_{i:03d}",
                "source_dataset": source_name,
                "original_index": idx,
                "article": article,
                "lay_summary": lay_summary,
                "abstract": abstract,
                "lay_summary_word_count": _word_count(lay_summary),
                "article_word_count": _word_count(article),
            }
        )
    return sampled, indices


def main() -> None:
    logger = setup_logger("load_data", str(config.LOGS_DIR / "02_load_data.log"))
    logger.info("Start loading BioLaySumm validation datasets.")

    random.seed(config.SEED)

    all_records: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "sampling_time": datetime.now().isoformat(timespec="seconds"),
        "seed": config.SEED,
        "datasets": {},
    }

    for source_name, dataset_name, id_prefix in DATASET_SPECS:
        ds, mapping, full_lay_stats = inspect_dataset(source_name, dataset_name, logger)
        sampled, sampled_indices = sample_records(
            ds=ds,
            source_name=source_name,
            id_prefix=id_prefix,
            mapping=mapping,
            sample_size=20,
        )
        all_records.extend(sampled)
        sampled_lay_stats = _series_stats([item["lay_summary_word_count"] for item in sampled])

        metadata["datasets"][source_name] = {
            "dataset_name": dataset_name,
            "split": "validation",
            "sample_size": len(sampled),
            "full_validation_size": len(ds),
            "sampled_indices": sampled_indices,
            "field_mapping": mapping,
            "full_validation_lay_summary_stats": full_lay_stats,
            "sampled_lay_summary_stats": sampled_lay_stats,
        }
        logger.info(
            "Sampled %s records from %s | sampled_lay_stats=%s",
            len(sampled),
            source_name,
            sampled_lay_stats,
        )

    out_jsonl = config.DATA_RAW_DIR / "sampled_articles.jsonl"
    out_metadata = config.DATA_RAW_DIR / "sampling_metadata.json"
    save_jsonl(all_records, out_jsonl)
    save_json(metadata, out_metadata)

    by_source: dict[str, list[dict[str, Any]]] = {"PLOS": [], "eLife": []}
    for rec in all_records:
        by_source[rec["source_dataset"]].append(rec)

    print("\n" + "=" * 72)
    print("[Sampling Summary]")
    print(f"Total sampled: {len(all_records)}")
    print(f"PLOS sampled: {len(by_source['PLOS'])}")
    print(f"eLife sampled: {len(by_source['eLife'])}")
    for source_name in ["PLOS", "eLife"]:
        lay_stats = _series_stats(
            [item["lay_summary_word_count"] for item in by_source[source_name]]
        )
        print(
            f"{source_name} lay_summary words -> "
            f"mean={lay_stats['mean']}, median={lay_stats['median']}, "
            f"min={lay_stats['min']}, max={lay_stats['max']}"
        )

    print(f"Saved sampled JSONL: {out_jsonl}")
    print(f"Saved metadata JSON: {out_metadata}")

    if all_records:
        one = random.choice(all_records)
        print("\n[Random Sample Preview]")
        print(f"id: {one['id']}")
        print("lay_summary:")
        print(one["lay_summary"])

    logger.info("Data loading and sampling completed. total_sampled=%s", len(all_records))


if __name__ == "__main__":
    main()
