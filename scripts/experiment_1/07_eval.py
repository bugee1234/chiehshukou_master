from __future__ import annotations

import argparse

from run_1a import stage06_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp1-1A Stage 07: evaluate and export detailed metrics")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--pilot-first-n", type=int, default=None)
    args = parser.parse_args()
    stage06_eval(args.top_k, args.pilot_first_n)


if __name__ == "__main__":
    main()
