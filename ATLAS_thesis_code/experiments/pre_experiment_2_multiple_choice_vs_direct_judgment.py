from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from atlas.experiment_support import false_option_letter, proportion
from atlas.io import load_articles, save_json, save_jsonl
from atlas.pipeline import AtlasPipeline


def run(input_path: Path, output_dir: Path) -> None:
    articles = load_articles(input_path)
    question_pipeline = AtlasPipeline("gpt41")
    controlled_summary_builder = AtlasPipeline("gpt41_mini")
    checking_pipeline = AtlasPipeline("gpt41_mini")
    cases: List[dict] = []

    for article in articles:
        article_id = str(article["id"])
        expert_summary = str(article["expert_summary"])
        reference_facts = question_pipeline.extract_reference_atomic_facts(article_id, expert_summary)
        for fact in reference_facts:
            question = question_pipeline.generate_question(fact)
            _, altered_statement = false_option_letter(question)
            conditions = {
                "F_keep": controlled_summary_builder.rewrite_controlled_summary(
                    expert_summary,
                    fact["atomic_fact"],
                    "F_keep",
                ),
                "F_delete": controlled_summary_builder.rewrite_controlled_summary(
                    expert_summary,
                    fact["atomic_fact"],
                    "F_delete",
                ),
                "F_alter": controlled_summary_builder.rewrite_controlled_summary(
                    expert_summary,
                    fact["atomic_fact"],
                    "F_alter",
                    altered_statement,
                ),
            }
            for condition, test_summary in conditions.items():
                expected_error = condition != "F_keep"
                direct_supported = checking_pipeline.direct_support_judgment(
                    test_summary, fact["atomic_fact"]
                )
                multiple_choice = checking_pipeline.answer_pre_experiment_question(
                    "pre_experiment_2_multiple_choice_checking.txt",
                    "test_summary",
                    test_summary,
                    question,
                    include_reasoning=False,
                )
                cases.append(
                    {
                        "article_id": article_id,
                        "fact_id": fact["fact_id"],
                        "condition": condition,
                        "expected_error": expected_error,
                        "direct_judgment_detected_error": not direct_supported,
                        "multiple_choice_detected_error": (
                            multiple_choice["answer"] != question["correct_letter"]
                        ),
                    }
                )

    save_jsonl(cases, output_dir / "controlled_error_cases.jsonl")
    true_errors = [row for row in cases if row["expected_error"]]
    save_json(
        {
            "direct_judgment_error_detection_rate": proportion(
                row["direct_judgment_detected_error"] for row in true_errors
            ),
            "multiple_choice_error_detection_rate": proportion(
                row["multiple_choice_detected_error"] for row in true_errors
            ),
        },
        output_dir / "pre_experiment_2_metrics.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-Experiment 2: Error Detection—Multiple-Choice vs. Direct Judgment."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
