from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RELEVANCE = ("ROUGE", "BLEU", "METEOR", "BERTScore")
READABILITY = ("FKGL", "DCRS", "CLI", "LENS")
FACTUALITY = ("AlignScore", "SummaC")
LOWER_IS_BETTER = {"FKGL", "DCRS", "CLI"}
ALL_METRICS = RELEVANCE + READABILITY + FACTUALITY


def normalize_and_rank(table: pd.DataFrame) -> pd.DataFrame:
    required = {"system", *ALL_METRICS}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError("Input table is missing columns: {}".format(sorted(missing)))
    normalized = table.copy()
    for metric in ALL_METRICS:
        values = table[metric].astype(float)
        spread = values.max() - values.min()
        normalized[metric] = 0.0 if spread == 0 else (values - values.min()) / spread
        if metric in LOWER_IS_BETTER:
            normalized[metric] = 1.0 - normalized[metric]
    normalized["Relevance"] = normalized[list(RELEVANCE)].mean(axis=1)
    normalized["Readability"] = normalized[list(READABILITY)].mean(axis=1)
    normalized["Factuality"] = normalized[list(FACTUALITY)].mean(axis=1)
    normalized["Final Score"] = normalized[["Relevance", "Readability", "Factuality"]].mean(axis=1)
    normalized = normalized.sort_values("Final Score", ascending=False).reset_index(drop=True)
    normalized["Rank"] = normalized.index + 1
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Leaderboard Comparison.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = normalize_and_rank(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
