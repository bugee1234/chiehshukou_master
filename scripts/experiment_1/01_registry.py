from __future__ import annotations

import argparse

from run_1a import stage00_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp1-1A Stage 01: build registry pool and pilot manifest")
    parser.add_argument("--seed-pool", type=int, default=20260520)
    parser.add_argument("--seed-pilot", type=int, default=20260521)
    parser.add_argument("--n-full", type=int, default=20)
    parser.add_argument("--n-pilot", type=int, default=5)
    args = parser.parse_args()
    stage00_registry(args.seed_pool, args.seed_pilot, args.n_full, args.n_pilot)


if __name__ == "__main__":
    main()
