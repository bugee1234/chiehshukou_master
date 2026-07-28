from __future__ import annotations

import argparse
from pathlib import Path

from atlas.io import load_articles, save_jsonl
from atlas.pipeline import AtlasPipeline, run_direct_generation


BACKBONE_MODELS = ("gemini25_flash", "gemini3_flash_preview", "gpt41_mini")


def run(input_path: Path, output_dir: Path) -> None:
    articles = load_articles(input_path)
    for model_key in BACKBONE_MODELS:
        direct_rows = run_direct_generation(articles, model_key, output_dir / "direct_generation")
        direct_by_id = {str(row["article_id"]): row["summary"] for row in direct_rows}
        pipeline = AtlasPipeline(model_key)
        atlas_rows = pipeline.run(articles, output_dir / "atlas")
        article_by_id = {str(row["id"]): row for row in articles}
        ablation_rows = []
        for atlas_row in atlas_rows:
            article_id = str(atlas_row["article_id"])
            module_1_summary = pipeline.generate_module_1_only_summary(
                str(article_by_id[article_id]["article"]),
                atlas_row["candidate_atomic_facts"],
            )
            ablation_rows.append(
                {
                    "article_id": article_id,
                    "source_dataset": atlas_row["source_dataset"],
                    "direct_generation": direct_by_id[article_id],
                    "module_1_omission_control": module_1_summary,
                    "modules_1_and_2_hallucination_control": atlas_row["initial_summary"],
                    "modules_1_2_and_3_expression_control": atlas_row["final_summary"],
                }
            )
        save_jsonl(ablation_rows, output_dir / "ablation" / model_key / "stage_summaries.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Main experiment: direct generation versus ATLAS.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
