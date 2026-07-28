from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from atlas.experiment_support import proportion
from atlas.io import load_articles, load_jsonl, save_json, save_jsonl
from atlas.pipeline import AtlasPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate omission with Expert Summary Recall.")
    parser.add_argument("--articles", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-model", default="gpt41")
    args = parser.parse_args()

    articles = {str(row["id"]): row for row in load_articles(args.articles)}
    summaries = load_jsonl(args.summaries)
    judge = AtlasPipeline(args.judge_model)
    cases: List[dict] = []
    for summary_row in summaries:
        article_id = str(summary_row["article_id"])
        article = articles[article_id]
        reference_facts = judge.extract_reference_atomic_facts(
            article_id, str(article["expert_summary"])
        )
        target_text = str(summary_row["summary"])
        for fact in reference_facts:
            cases.append(
                {
                    "article_id": article_id,
                    "source_dataset": article["source_dataset"],
                    "reference_fact_id": fact["fact_id"],
                    "covered": judge.target_text_covers(fact["atomic_fact"], target_text),
                }
            )
    save_jsonl(cases, args.output_dir / "expert_summary_recall_cases.jsonl")
    source_names = sorted({row["source_dataset"] for row in cases})
    save_json(
        {
            "overall": proportion(row["covered"] for row in cases),
            "by_source": {
                source: proportion(
                    row["covered"] for row in cases if row["source_dataset"] == source
                )
                for source in source_names
            },
        },
        args.output_dir / "expert_summary_recall.json",
    )


if __name__ == "__main__":
    main()
