from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from atlas.io import load_articles, save_json, save_jsonl
from atlas.pipeline import AtlasPipeline


EXTRACTION_MODELS = ("gpt41_mini", "gemini3_flash_preview", "gemini25_flash")


def run(input_path: Path, output_dir: Path) -> None:
    articles = load_articles(input_path)
    reference_pipeline = AtlasPipeline("gpt41")
    coverage_judge = AtlasPipeline("gpt41")
    cases: List[dict] = []

    for article in articles:
        article_id = str(article["id"])
        reference_facts = reference_pipeline.extract_reference_atomic_facts(
            article_id, str(article["expert_summary"])
        )
        for model_key in EXTRACTION_MODELS:
            candidate_facts = AtlasPipeline(model_key).extract_candidate_atomic_facts(
                article_id, str(article["article"])
            )
            candidate_texts = [candidate["atomic_fact"] for candidate in candidate_facts]
            for reference_fact in reference_facts:
                cases.append(
                    {
                        "article_id": article_id,
                        "reference_fact_id": reference_fact["fact_id"],
                        "extraction_model": model_key,
                        "covered": coverage_judge.candidate_set_covers(
                            reference_fact["atomic_fact"], candidate_texts
                        ),
                    }
                )

    save_jsonl(cases, output_dir / "module_1_fact_coverage_cases.jsonl")
    metrics: Dict[str, float] = {
        model_key: sum(
            row["covered"] for row in cases if row["extraction_model"] == model_key
        )
        / sum(1 for row in cases if row["extraction_model"] == model_key)
        for model_key in EXTRACTION_MODELS
    }
    save_json({"expert_summary_recall": metrics}, output_dir / "pre_experiment_3_metrics.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-Experiment 3: Fact Coverage of Module 1 Extraction."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
