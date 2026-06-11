from typing import List, Optional

import pandas as pd


def min_max_normalize(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Apply min-max normalization to specified columns in a DataFrame."""
    result_df = df.copy()

    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()

            # Avoid division by zero
            if max_val == min_val:
                result_df[col] = 0.0
            else:
                result_df[col] = (df[col] - min_val) / (max_val - min_val)

    return result_df


def main():
    SCORES = ["ROUGE", "BLEU", "METEOR", "BERTScore", "FKGL", "DCRS", "CLI", "LENS(Task1)", "AlignScore(Task1)",
              "SummaC(Task1)"]
    RELEVANCE = ["ROUGE", "BLEU", "METEOR", "BERTScore"]
    READABILITY = ["FKGL", "DCRS", "CLI", "LENS(Task1)"]
    FACTUALITY = ["AlignScore(Task1)", "SummaC(Task1)"]
    LOWER_IS_BETTER = ["FKGL", "DCRS", "CLI"]

    input_file = "leaderboard.csv"

    # Read CSV
    df = pd.read_csv(input_file)
    df = df[df["Select Task"] == "SubTask 1.1"]

    # Min-max normalize
    normalized_df = min_max_normalize(df, SCORES)

    # Reverse LOWER_IS_BETTER metrics
    for col in LOWER_IS_BETTER:
        normalized_df[col] = 1 - normalized_df[col]

    # Calculate the average for each category
    normalized_df["Relevance"] = normalized_df[RELEVANCE].mean(axis=1)
    normalized_df["Readability"] = normalized_df[READABILITY].mean(axis=1)
    normalized_df["Factuality"] = normalized_df[FACTUALITY].mean(axis=1)
    normalized_df["Final"] = normalized_df[["Relevance", "Readability", "Factuality"]].mean(axis=1)

    # Sort by final score
    normalized_df = normalized_df.sort_values("Final", ascending=False)

    # Format table
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 3)

    # Print table
    print("\nNormalized Leaderboard:")
    print(normalized_df)


if __name__ == "__main__":
    main()
