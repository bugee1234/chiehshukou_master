from __future__ import annotations

import argparse

from run_1a import stage05_judgement


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp1-1A Stage 06: run setup A/B judgement")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--parallel-setups", action="store_true")
    parser.add_argument("--reuse-questions", action="store_true")
    parser.add_argument("--pilot-first-n", type=int, default=None)
    parser.add_argument(
        "--c-recheck-mode",
        type=str,
        default="strict",
        choices=["off", "lenient", "strict"],
    )
    args = parser.parse_args()
    stage05_judgement(
        args.model,
        args.top_k,
        parallel_setups=args.parallel_setups,
        reuse_questions=args.reuse_questions,
        pilot_first_n=args.pilot_first_n,
        c_recheck_mode=args.c_recheck_mode,
    )


if __name__ == "__main__":
    main()
