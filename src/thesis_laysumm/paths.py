from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
PIPELINE_SRC_DIR = SRC_DIR / "thesis_laysumm"
PROMPTS_DIR = PIPELINE_SRC_DIR / "prompts"

# Distinct from legacy experiment_3 / experiment_3_module2_repro paths.
DATA_DIR = ROOT_DIR / "data" / "laysumm_pipeline"
RESULTS_DIR = ROOT_DIR / "results" / "laysumm_pipeline"


def run_data_dir(run_name: str) -> Path:
    return DATA_DIR / "runs" / run_name


def run_results_dir(run_name: str) -> Path:
    return RESULTS_DIR / "runs" / run_name


def stage_data_dir(run_name: str, stage: str) -> Path:
    path = run_data_dir(run_name) / stage
    path.mkdir(parents=True, exist_ok=True)
    return path
