from __future__ import annotations

import argparse

from run_1a import stage03_perturb


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp1-1A Stage 04: perturb AF into F_del/F_error/F_kept")
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--ratio", type=float, default=0.30)
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    args = parser.parse_args()
    stage03_perturb(args.seed, args.ratio, args.model)


if __name__ == "__main__":
    main()
