from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from atlas.io import save_jsonl


DATASETS = {
    "PLOS": "BioLaySumm/BioLaySumm2025-PLOS",
    "eLife": "BioLaySumm/BioLaySumm2025-eLife",
}


def reference_text(item: dict) -> str:
    for field in ("lay_summary", "summary", "reference"):
        value = str(item.get(field, "") or "").strip()
        if value:
            return value
    raise ValueError("Dataset record has no expert-written lay summary")


def abstract_text(item: dict) -> str:
    article = str(item.get("article", "") or "").strip()
    if not article:
        raise ValueError("Dataset record has no source article")
    sections = article.split("\n")
    headings = item.get("section_headings") or []
    if headings and str(headings[0]).strip().lower() != "abstract":
        raise ValueError("The first source-article section is not the abstract")
    abstract = sections[0].strip()
    if not abstract:
        raise ValueError("Dataset record has an empty abstract")
    return abstract


def prepare(split: str, output: Path) -> None:
    from datasets import load_dataset

    rows: List[dict] = []
    token = os.environ.get("HF_TOKEN") or None
    for source_dataset, dataset_name in DATASETS.items():
        dataset = load_dataset(dataset_name, split=split, token=token)
        for original_index, item in enumerate(dataset):
            rows.append(
                {
                    "id": "{}_{}".format(source_dataset.lower(), original_index),
                    "source_dataset": source_dataset,
                    "article": str(item.get("article", "") or "").strip(),
                    "abstract": abstract_text(item),
                    "expert_summary": reference_text(item),
                }
            )
    save_jsonl(rows, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BioLaySumm input for the thesis experiments.")
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.split, args.output)


if __name__ == "__main__":
    main()
