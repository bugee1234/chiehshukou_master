from __future__ import annotations

import argparse

from run_1a import stage02_extract_af


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp1-1A Stage 03: extract AF from pilot summaries")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    args = parser.parse_args()
    stage02_extract_af(args.model)


if __name__ == "__main__":
    main()
