from __future__ import annotations

import argparse
from pathlib import Path

from atlas.io import load_articles
from atlas.pipeline import AtlasPipeline


PUBLIC_BACKBONES = ("qwen25_7b_instruct", "llama3_8b_instruct")


def main() -> None:
    parser = argparse.ArgumentParser(description="Supplementary Backbone Validation.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    articles = load_articles(args.input)
    for model_key in PUBLIC_BACKBONES:
        AtlasPipeline(model_key).run(articles, args.output_dir)


if __name__ == "__main__":
    main()
