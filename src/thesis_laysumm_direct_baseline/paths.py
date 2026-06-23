from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT_DIR / "data" / "laysumm_direct_baseline" / "runs"
RESULTS_ROOT = ROOT_DIR / "results" / "laysumm_direct_baseline" / "runs"
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "direct_lay_summary.txt"


def run_data_dir(run_name: str) -> Path:
    return DATA_ROOT / run_name


def run_results_dir(run_name: str) -> Path:
    return RESULTS_ROOT / run_name


def inputs_path(run_name: str) -> Path:
    return run_data_dir(run_name) / "00_inputs" / "articles.jsonl"


def references_path(run_name: str) -> Path:
    return run_data_dir(run_name) / "00_references" / "references.jsonl"


def summaries_path(run_name: str, model_key: str) -> Path:
    return run_data_dir(run_name) / "01_direct_summaries" / model_key / "generated_summaries.jsonl"

