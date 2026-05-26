from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_1A = ROOT / "data" / "experiment_1" / "1a"
ABLA_DIR = DATA_1A / "ablation_runs"
JUDGE_DIR = DATA_1A / "04_judgement"
METRIC_DIR = DATA_1A / "05_metrics"


RUNS = [
    ("k3_off", 3, "off"),
    ("k3_lenient", 3, "lenient"),
    ("k3_strict", 3, "strict"),
    ("k5_off", 5, "off"),
    ("k5_lenient", 5, "lenient"),
    ("k5_strict", 5, "strict"),
]


def _run_cmd(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def _copy_result_files(run_id: str, top_k: int) -> None:
    out_dir = ABLA_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        JUDGE_DIR / "setup_a_predictions.jsonl",
        JUDGE_DIR / "setup_b_predictions.jsonl",
        JUDGE_DIR / "setup_c_predictions.jsonl",
        JUDGE_DIR / "judgement_metadata.json",
        JUDGE_DIR / "questions_1t3f.jsonl",
        METRIC_DIR / "metrics_setup_a.json",
        METRIC_DIR / "metrics_setup_b.json",
        METRIC_DIR / "metrics_setup_c.json",
        METRIC_DIR / "confusion_matrix.csv",
        METRIC_DIR / f"metrics_detailed_k{top_k}.json",
    ]
    for src in files_to_copy:
        if src.exists():
            shutil.copy2(src, out_dir / src.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp1-1A C ablation batch and archive outputs.")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--pilot-first-n", type=int, default=10)
    parser.add_argument(
        "--run-ids",
        type=str,
        default="k3_off,k3_lenient,k3_strict,k5_off,k5_lenient,k5_strict",
        help="Comma-separated subset of run ids to execute",
    )
    parser.add_argument(
        "--reuse-questions",
        action="store_true",
        help="Reuse existing questions_1t3f.jsonl when available",
    )
    args = parser.parse_args()

    wanted = {x.strip() for x in args.run_ids.split(",") if x.strip()}
    selected = [r for r in RUNS if r[0] in wanted]
    if not selected:
        raise SystemExit("No valid run ids selected.")

    ABLA_DIR.mkdir(parents=True, exist_ok=True)

    for run_id, top_k, mode in selected:
        print(f"\n=== Running {run_id} (k={top_k}, mode={mode}) ===")
        cmd = [
            "python3",
            "scripts/experiment_1/06_judge_ab.py",
            "--model",
            args.model,
            "--top-k",
            str(top_k),
            "--parallel-setups",
            "--pilot-first-n",
            str(args.pilot_first_n),
            "--c-recheck-mode",
            mode,
        ]
        if args.reuse_questions:
            cmd.append("--reuse-questions")
        _run_cmd(cmd)

        _run_cmd(
            [
                "python3",
                "scripts/experiment_1/07_eval.py",
                "--top-k",
                str(top_k),
                "--pilot-first-n",
                str(args.pilot_first_n),
            ]
        )
        _copy_result_files(run_id, top_k)
        print(f"Saved to: {ABLA_DIR / run_id}")

    print("\nAll selected runs finished.")


if __name__ == "__main__":
    main()
