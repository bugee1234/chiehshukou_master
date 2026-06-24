from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "laysumm_ablation_m1_recall"
RESULTS_DIR = ROOT_DIR / "results" / "laysumm_ablation_m1_recall"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def run_data_dir(run_name: str) -> Path:
    return DATA_DIR / "runs" / run_name


def run_results_dir(run_name: str) -> Path:
    return RESULTS_DIR / "runs" / run_name


def usage_path(run_name: str) -> Path:
    return run_data_dir(run_name) / "api_usage_calls.csv"

