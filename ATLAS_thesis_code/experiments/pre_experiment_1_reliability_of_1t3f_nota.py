from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from atlas.experiment_support import false_option_letter, proportion, replace_supporting_sentence
from atlas.io import load_articles, save_json, save_jsonl
from atlas.pipeline import AtlasPipeline


def run(input_path: Path, output_dir: Path) -> None:
    articles = load_articles(input_path)
    question_pipeline = AtlasPipeline("gpt41")
    answerability_judge = AtlasPipeline("gpt41")
    full_context_judges = {
        "gpt41": AtlasPipeline("gpt41"),
        "gpt4o_mini": AtlasPipeline("gpt4o_mini"),
    }
    answerability_cases: List[dict] = []
    full_context_cases: List[dict] = []

    for article in articles:
        article_id = str(article["id"])
        expert_summary = str(article["expert_summary"])
        reference_facts = question_pipeline.extract_reference_atomic_facts(article_id, expert_summary)
        for fact in reference_facts:
            question = question_pipeline.generate_question(fact)
            answer = answerability_judge.answer_pre_experiment_question(
                "pre_experiment_1_answerability_test.txt",
                "expert_summary",
                expert_summary,
                question,
                include_reasoning=True,
            )
            answerability_cases.append(
                {
                    "article_id": article_id,
                    "fact_id": fact["fact_id"],
                    "selected_answer": answer["answer"],
                    "expected_answer": question["correct_letter"],
                    "correct": answer["answer"] == question["correct_letter"],
                }
            )

            context_supported_letter, altered_statement = false_option_letter(question)
            altered_summary = replace_supporting_sentence(
                expert_summary,
                fact["source_span"],
                fact["atomic_fact"],
                altered_statement,
            )
            for judge_name, judge in full_context_judges.items():
                context_answer = judge.answer_pre_experiment_question(
                    "pre_experiment_1_full_context_following_test.txt",
                    "altered_expert_summary",
                    altered_summary,
                    question,
                    include_reasoning=True,
                )
                full_context_cases.append(
                    {
                        "article_id": article_id,
                        "fact_id": fact["fact_id"],
                        "judge": judge_name,
                        "selected_answer": context_answer["answer"],
                        "context_supported_answer": context_supported_letter,
                        "followed_complete_context": (
                            context_answer["answer"] == context_supported_letter
                        ),
                    }
                )

    save_jsonl(answerability_cases, output_dir / "answerability_test_cases.jsonl")
    save_jsonl(full_context_cases, output_dir / "full_context_following_test_cases.jsonl")
    save_json(
        {
            "answerability_accuracy": proportion(row["correct"] for row in answerability_cases),
            "robustness_to_knowledge_conflicts": {
                judge: proportion(
                    row["followed_complete_context"]
                    for row in full_context_cases
                    if row["judge"] == judge
                )
                for judge in full_context_judges
            },
        },
        output_dir / "pre_experiment_1_metrics.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-Experiment 1: Reliability of the 1T3F+NOTA Question Format."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
