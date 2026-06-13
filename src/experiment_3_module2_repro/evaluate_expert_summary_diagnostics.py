from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

REFERENCE_SELF_CHECK = "reference-self-check"
REFERENCE_AS_DOCUMENT_SELF_CHECK = "reference-as-document-self-check"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_scores(scores: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{key}: {value}" for key, value in scores.items()) + "\n",
        encoding="utf-8",
    )


def load_articles(run_dir: Path) -> list[dict[str, Any]]:
    articles = read_jsonl(run_dir / "00_inputs" / "articles.jsonl")
    if not articles:
        raise ValueError(f"No articles found in {run_dir / '00_inputs' / 'articles.jsonl'}")
    return articles


def build_refs_dicts(rows: list[dict[str, Any]], variant: str) -> tuple[list[str], list[dict[str, str]]]:
    preds: list[str] = []
    refs_dicts: list[dict[str, str]] = []
    for row in rows:
        expert_summary = str(row.get("expert_summary", "") or "")
        article = str(row.get("article", "") or "")
        if not expert_summary.strip():
            raise ValueError(f"Missing expert_summary for id={row.get('id')}")
        if not article.strip():
            raise ValueError(f"Missing article for id={row.get('id')}")

        preds.append(expert_summary)
        document = expert_summary if variant == REFERENCE_AS_DOCUMENT_SELF_CHECK else article
        refs_dicts.append({"document": document, "reference": expert_summary})
    return preds, refs_dicts


def evaluate_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    from evaluation.evaluation_final import evaluate_all

    per_source: dict[str, dict[str, float]] = {}
    for source in ("eLife", "PLOS"):
        source_rows = [row for row in rows if str(row.get("source_dataset")) == source]
        if not source_rows:
            raise ValueError(f"No rows found for source_dataset={source}")
        preds, refs_dicts = build_refs_dicts(source_rows, variant)
        per_source[source] = evaluate_all(preds, refs_dicts, "lay_summ")
        torch.cuda.empty_cache()

    overall = {
        key: float(np.mean([per_source["eLife"][key], per_source["PLOS"][key]]))
        for key in per_source["eLife"]
    }
    return {"variant": variant, "eLife": per_source["eLife"], "PLOS": per_source["PLOS"], "overall": overall}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate expert-summary diagnostics for Experiment 3 module2 repro runs."
    )
    parser.add_argument("--run-name", required=True, type=str)
    parser.add_argument(
        "--variant",
        choices=[REFERENCE_SELF_CHECK, REFERENCE_AS_DOCUMENT_SELF_CHECK, "both"],
        default="both",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT_DIR / "data" / "experiment_3_module2_repro" / "runs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT_DIR / "results" / "experiment_3_module2_repro",
    )
    args = parser.parse_args()

    run_dir = args.data_root / args.run_name
    rows = load_articles(run_dir)
    variants = (
        [REFERENCE_SELF_CHECK, REFERENCE_AS_DOCUMENT_SELF_CHECK]
        if args.variant == "both"
        else [args.variant]
    )

    out_dir = args.output_root / args.run_name / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        result = evaluate_variant(rows, variant)
        variant_dir = out_dir / variant.replace("-", "_")
        write_scores(result["overall"], variant_dir / "scores.txt")
        (variant_dir / "scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[evaluate_expert_summary_diagnostics] wrote -> {variant_dir}")


if __name__ == "__main__":
    main()
