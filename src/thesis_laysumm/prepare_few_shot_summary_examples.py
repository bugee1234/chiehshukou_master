from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.thesis_laysumm.biolaysumm_data import (
    extract_article_sections,
    load_biolaysumm_split,
    word_count,
)
from src.thesis_laysumm.paths import PROMPTS_DIR
from src.utils import load_json, save_json

DEFAULT_OUT = PROMPTS_DIR / "few_shot_summary_examples.json"


def _pilot_indices(run_name: str) -> dict[str, set[int]]:
    meta_path = ROOT_DIR / "data" / "laysumm_pipeline" / "runs" / run_name / "00_articles" / "selection_metadata.json"
    if not meta_path.exists():
        return {"PLOS": set(), "eLife": set()}
    meta = load_json(meta_path)
    out: dict[str, set[int]] = {}
    for source in ("PLOS", "eLife"):
        indices = meta.get("datasets", {}).get(source, {}).get("selected_indices", [])
        out[source] = {int(i) for i in indices}
    return out


def pick_example(source: str, excluded_indices: set[int]) -> dict:
    ds = load_biolaysumm_split(source, "validation")
    for idx in range(len(ds)):
        if idx in excluded_indices:
            continue
        item = ds[idx]
        sections = extract_article_sections(item["article"], item.get("section_headings") or [])
        abstract = sections["abstract"]
        expert = str(item.get("summary") or item.get("lay_summary") or "").strip()
        if not abstract or not expert:
            continue
        return {
            "source_dataset": source,
            "validation_index": idx,
            "title": str(item.get("title") or ""),
            "abstract": abstract,
            "expert_summary": expert,
            "expert_summary_word_count": word_count(expert),
            "abstract_word_count": word_count(abstract),
        }
    raise RuntimeError(f"Could not find a held-out validation example for {source}")


def prepare_few_shot_examples(
    *,
    run_name: str | None,
    out_path: Path,
    plos_index: int | None,
    elife_index: int | None,
) -> list[dict]:
    excluded = _pilot_indices(run_name) if run_name else {"PLOS": set(), "eLife": set()}

    if plos_index is not None:
        plos_example = _example_at_index("PLOS", plos_index)
    else:
        plos_example = pick_example("PLOS", excluded.get("PLOS", set()))

    if elife_index is not None:
        elife_example = _example_at_index("eLife", elife_index)
    else:
        elife_example = pick_example("eLife", excluded.get("eLife", set()))

    examples = [plos_example, elife_example]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(examples, out_path)
    return examples


def _example_at_index(source: str, index: int) -> dict:
    ds = load_biolaysumm_split(source, "validation")
    if index < 0 or index >= len(ds):
        raise IndexError(f"{source} validation index {index} out of range (size={len(ds)})")
    item = ds[index]
    sections = extract_article_sections(item["article"], item.get("section_headings") or [])
    abstract = sections["abstract"]
    expert = str(item.get("summary") or item.get("lay_summary") or "").strip()
    if not abstract or not expert:
        raise ValueError(f"{source} validation index {index} has empty abstract or expert summary")
    return {
        "source_dataset": source,
        "validation_index": index,
        "title": str(item.get("title") or ""),
        "abstract": abstract,
        "expert_summary": expert,
        "expert_summary_word_count": word_count(expert),
        "abstract_word_count": word_count(abstract),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare fixed held-out few-shot summary examples (1 PLOS + 1 eLife)."
    )
    parser.add_argument("--run-name", type=str, default="pilot_n20_v1")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--plos-index", type=int, default=None)
    parser.add_argument("--elife-index", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    examples = prepare_few_shot_examples(
        run_name=args.run_name,
        out_path=args.out,
        plos_index=args.plos_index,
        elife_index=args.elife_index,
    )
    for ex in examples:
        print(
            f"[few-shot] {ex['source_dataset']} idx={ex['validation_index']} "
            f"expert_wc={ex['expert_summary_word_count']} title={ex['title'][:70]}"
        )
    print(f"[few-shot] wrote {args.out}")


if __name__ == "__main__":
    main()
