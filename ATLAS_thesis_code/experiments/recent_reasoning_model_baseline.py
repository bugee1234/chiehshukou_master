from __future__ import annotations

import argparse
from pathlib import Path

from atlas.io import load_articles
from atlas.pipeline import run_direct_generation


def main() -> None:
    parser = argparse.ArgumentParser(description="Recent Reasoning-Model Baseline.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_direct_generation(load_articles(args.input), "gpt54_mini", args.output_dir)


if __name__ == "__main__":
    main()
