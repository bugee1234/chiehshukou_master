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

from src.thesis_laysumm.paths import run_results_dir as pipeline_results_dir
from src.thesis_laysumm_direct_baseline.paths import run_results_dir
from src.utils import load_json, save_json


DEFAULT_MODELS = [
    "gemini25_flash_non_thinking",
    "gpt41_mini",
    "gemini3_flash_preview_minimal",
]
METRICS = ["ROUGE", "BLEU", "METEOR", "BERTScore", "FKGL", "DCRS", "CLI", "LENS", "AlignScore", "SummaC"]


def summarize(*, run_name: str, pipeline_run: str, models: list[str]) -> None:
    out_root = run_results_dir(run_name)
    records: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "direct_run": run_name,
        "pipeline_run": pipeline_run,
        "caution": "Comparisons are valid only when direct and pipeline scores use the same article set.",
        "models": {},
    }
    for model in models:
        direct_path = out_root / model / "generated_scores.json"
        if not direct_path.exists():
            raise FileNotFoundError(f"Missing direct scores: {direct_path}")
        direct = load_json(direct_path)
        model_payload: dict[str, Any] = {"direct": direct["overall"], "pipeline": {}}
        for variant in ("generated", "rewritten"):
            pipeline_path = pipeline_results_dir(pipeline_run) / model / f"{variant}_scores.json"
            if not pipeline_path.exists():
                raise FileNotFoundError(f"Missing pipeline scores: {pipeline_path}")
            pipeline = load_json(pipeline_path)
            if int(direct["article_count"]) != int(pipeline["article_count"]):
                raise ValueError(
                    f"Article-count mismatch for {model}/{variant}: "
                    f"direct={direct['article_count']} pipeline={pipeline['article_count']}"
                )
            delta = {
                metric: float(direct["overall"][metric]) - float(pipeline["overall"][metric])
                for metric in METRICS
            }
            model_payload["pipeline"][variant] = {
                "scores": pipeline["overall"],
                "direct_minus_pipeline": delta,
            }
            for metric in METRICS:
                records.append(
                    {
                        "model": model,
                        "pipeline_variant": variant,
                        "metric": metric,
                        "direct": direct["overall"][metric],
                        "pipeline": pipeline["overall"][metric],
                        "direct_minus_pipeline": delta[metric],
                    }
                )
        payload["models"][model] = model_payload
    save_json(payload, out_root / "three_model_direct_vs_pipeline.json")
    with (out_root / "three_model_direct_vs_pipeline.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"[direct comparison] models={len(models)} -> {out_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine direct-vs-pipeline metrics across models.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--pipeline-run", required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summarize(run_name=args.run_name, pipeline_run=args.pipeline_run, models=args.models)


if __name__ == "__main__":
    main()

