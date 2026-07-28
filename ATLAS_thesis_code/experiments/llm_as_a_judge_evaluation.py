from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from atlas.io import load_articles, load_jsonl, mean, save_json, save_jsonl
from atlas.pipeline import AtlasPipeline


def run(article_path: Path, summary_path: Path, output_dir: Path, judge_model: str) -> None:
    articles = {str(row["id"]): row for row in load_articles(article_path)}
    summaries = load_jsonl(summary_path)
    judge = AtlasPipeline(judge_model)
    cases: List[dict] = []
    for summary_row in summaries:
        article_id = str(summary_row["article_id"])
        article = articles[article_id]
        result = judge.judge_lay_summary(
            str(article["article"]),
            str(article["expert_summary"]),
            str(summary_row["summary"]),
        )
        cases.append({"article_id": article_id, **result})
    save_jsonl(cases, output_dir / "llm_as_a_judge_cases.jsonl")
    averages: Dict[str, dict] = {}
    for dimension in ("relevance", "readability", "factuality"):
        raw = mean(float(row[dimension]["score"]) for row in cases)
        averages[dimension] = {"mean_score": raw, "normalized_score": (raw - 1.0) / 4.0}
    save_json(averages, output_dir / "llm_as_a_judge_scores.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Evaluation.")
    parser.add_argument("--articles", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-model", default="gpt54_mini")
    args = parser.parse_args()
    run(args.articles, args.summaries, args.output_dir, args.judge_model)


if __name__ == "__main__":
    main()
