from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_scores(scores: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{key}: {value}" for key, value in scores.items()) + "\n",
        encoding="utf-8",
    )


def _assert_refs_available(refs: list[dict[str, Any]], path: Path) -> None:
    empty = sum(1 for r in refs if not str(r.get("reference", "")).strip())
    if empty:
        raise ValueError(
            f"{path} contains {empty}/{len(refs)} empty references. "
            "HF BioLaySumm test summaries are hidden/blank; provide official reference JSONL files "
            "with --truth-dir when running final evaluation."
        )


def read_predictions(
    eval_dir: Path,
    variant: str,
    source: str,
    refs: list[dict[str, Any]],
) -> list[str]:
    paired_path = eval_dir / "paired_summaries.jsonl"
    rows = [row for row in read_jsonl(paired_path) if row.get("source_dataset") == source]
    try:
        rows.sort(key=lambda row: int(row["original_index"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{paired_path} contains an invalid original_index for {source}") from exc

    indices = [int(row["original_index"]) for row in rows]
    expected_indices = list(range(len(refs)))
    if indices != expected_indices:
        raise ValueError(
            f"{paired_path} has invalid {source} original_index values: "
            f"expected {expected_indices}, found {indices}"
        )

    mismatched_refs = [
        index
        for index, (row, ref) in enumerate(zip(rows, refs))
        if str(row.get("reference", "")) != str(ref.get("reference", ""))
    ]
    if mismatched_refs:
        raise ValueError(
            f"{paired_path} {source} references do not match the truth JSONL "
            f"at original_index values {mismatched_refs}"
        )

    summary_key = f"{variant}_summary"
    predictions = [str(row.get(summary_key, "")) for row in rows]
    if not predictions or any(not prediction.strip() for prediction in predictions):
        raise ValueError(f"{paired_path} contains missing {summary_key} values for {source}")
    return predictions


def _assert_matching_counts(preds: list[str], refs: list[dict[str, Any]], source: str, variant: str) -> None:
    if len(preds) != len(refs):
        raise ValueError(
            f"{source} {variant} prediction/reference count mismatch: {len(preds)} != {len(refs)}"
        )


def evaluate_variant(run_dir: Path, variant: str, truth_dir: Path | None = None) -> dict[str, Any]:
    from evaluation.evaluation_final import evaluate_all

    eval_dir = run_dir / "06_eval_inputs"
    truth_dir = truth_dir or (run_dir / "00_inputs")
    elife_refs = read_jsonl(truth_dir / "eLife_test.jsonl")
    plos_refs = read_jsonl(truth_dir / "PLOS_test.jsonl")
    _assert_refs_available(elife_refs, truth_dir / "eLife_test.jsonl")
    _assert_refs_available(plos_refs, truth_dir / "PLOS_test.jsonl")

    elife_preds = read_predictions(eval_dir, variant, "eLife", elife_refs)
    plos_preds = read_predictions(eval_dir, variant, "PLOS", plos_refs)
    _assert_matching_counts(elife_preds, elife_refs, "eLife", variant)
    _assert_matching_counts(plos_preds, plos_refs, "PLOS", variant)

    elife_scores = evaluate_all(
        elife_preds,
        elife_refs,
        "lay_summ",
    )
    plos_scores = evaluate_all(
        plos_preds,
        plos_refs,
        "lay_summ",
    )
    overall = {key: float(np.mean([elife_scores[key], plos_scores[key]])) for key in elife_scores}
    return {"variant": variant, "eLife": elife_scores, "PLOS": plos_scores, "overall": overall}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Experiment 3 initial/rewritten outputs with BioLaySumm metrics.")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--variant", choices=["initial", "rewritten", "both"], default="both")
    parser.add_argument("--data-root", type=Path, default=ROOT_DIR / "data" / "experiment_3" / "runs")
    parser.add_argument("--output-root", type=Path, default=ROOT_DIR / "results" / "experiment_3")
    parser.add_argument(
        "--truth-dir",
        type=Path,
        default=None,
        help="Directory containing official eLife_test.jsonl and PLOS_test.jsonl references.",
    )
    args = parser.parse_args()

    run_dir = args.data_root / args.run_name
    variants = ["initial", "rewritten"] if args.variant == "both" else [args.variant]
    results = {variant: evaluate_variant(run_dir, variant, truth_dir=args.truth_dir) for variant in variants}

    out_dir = args.output_root / args.run_name / "official_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant, result in results.items():
        write_scores(result["overall"], out_dir / f"{variant}_scores.txt")
        (out_dir / f"{variant}_scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if "initial" in results and "rewritten" in results:
        delta = {
            key: results["rewritten"]["overall"][key] - results["initial"]["overall"][key]
            for key in results["initial"]["overall"]
        }
        write_scores(delta, out_dir / "rewritten_minus_initial.txt")
        (out_dir / "rewritten_minus_initial.json").write_text(json.dumps(delta, indent=2), encoding="utf-8")

    print(f"[evaluate_exp3] wrote metrics -> {out_dir}")


if __name__ == "__main__":
    main()
