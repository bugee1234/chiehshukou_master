from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from atlas.io import load_articles, load_jsonl, save_json


def aligned_rows(articles: List[dict], summaries: List[dict], source_dataset: str) -> List[dict]:
    article_by_id = {
        str(article["id"]): article
        for article in articles
        if article["source_dataset"] == source_dataset
    }
    summary_by_id = {str(row["article_id"]): row for row in summaries}
    if len(summary_by_id) != len(summaries):
        raise ValueError("Summary file contains duplicate article identifiers")
    if set(article_by_id) != set(summary_by_id).intersection(article_by_id):
        missing = sorted(set(article_by_id).difference(summary_by_id))
        raise ValueError("Summary file is missing source-dataset articles: {}".format(missing))
    return [
        {
            "prediction": str(summary_by_id[article_id]["summary"]),
            "reference": str(article["expert_summary"]),
            "document": str(article["article"]),
        }
        for article_id, article in article_by_id.items()
    ]


def run(article_path: Path, summary_path: Path, output_path: Path) -> None:
    from evaluation.evaluation_final import evaluate_all

    articles = load_articles(article_path)
    summaries = load_jsonl(summary_path)
    article_ids = {str(row["id"]) for row in articles}
    summary_ids = [str(row["article_id"]) for row in summaries]
    if len(summary_ids) != len(set(summary_ids)):
        raise ValueError("Summary file contains duplicate article identifiers")
    if article_ids != set(summary_ids):
        raise ValueError("Article and summary identifiers must match exactly")
    by_source: Dict[str, Dict[str, float]] = {}
    for source_dataset in ("PLOS", "eLife"):
        rows = aligned_rows(articles, summaries, source_dataset)
        by_source[source_dataset] = evaluate_all(
            [row["prediction"] for row in rows],
            [
                {"reference": row["reference"], "document": row["document"]}
                for row in rows
            ],
            "lay_summ",
        )
    overall = {
        metric: float(np.mean([by_source["PLOS"][metric], by_source["eLife"][metric]]))
        for metric in by_source["PLOS"]
    }
    save_json({"by_source": by_source, "overall": overall}, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the organizer-released BioLaySumm local evaluation implementation."
    )
    parser.add_argument("--articles", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.articles, args.summaries, args.output)


if __name__ == "__main__":
    main()
