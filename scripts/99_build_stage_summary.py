from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.utils import load_json, save_json


def _get_cost(metadata: dict[str, Any]) -> float:
    return float(metadata.get("estimated_cost_usd", 0.0) or 0.0)


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def main() -> None:
    sampling = _safe_load_json(config.DATA_RAW_DIR / "sampling_metadata.json")
    af_extraction = _safe_load_json(config.DATA_ATOMIC_FACTS_DIR / "extraction_metadata.json")
    question_generation = _safe_load_json(config.DATA_QUESTIONS_DIR / "generation_metadata.json")
    setup_gold = _safe_load_json(config.DATA_SOLVER_RESULTS_DIR / "setup_gold_metadata.json")

    total_cost = (
        _get_cost(af_extraction) + _get_cost(question_generation) + _get_cost(setup_gold)
    )

    summary = {
        "experiment_name": config.EXPERIMENT_NAME,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": config.SEED,
        "step_2_load_data": {
            "sampling_time": sampling.get("sampling_time"),
            "source_counts": {
                key: value.get("sample_size", 0)
                for key, value in sampling.get("datasets", {}).items()
            },
            "total_sampled_articles": sum(
                value.get("sample_size", 0) for value in sampling.get("datasets", {}).values()
            ),
            "field_mapping": {
                key: value.get("field_mapping", {})
                for key, value in sampling.get("datasets", {}).items()
            },
        },
        "step_3_af_extraction": {
            "model": af_extraction.get("model"),
            "temperature": af_extraction.get("temperature"),
            "total_afs_extracted": af_extraction.get("total_afs_extracted"),
            "articles_processed": af_extraction.get("articles_processed"),
            "articles_failed_count": len(af_extraction.get("articles_failed", [])),
            "afs_per_source": af_extraction.get("afs_per_source", {}),
            "token_usage": {
                "input": af_extraction.get("total_input_tokens", 0),
                "output": af_extraction.get("total_output_tokens", 0),
                "total": af_extraction.get("total_tokens", 0),
            },
            "estimated_cost_usd": _get_cost(af_extraction),
        },
        "step_4_question_generation": {
            "model": question_generation.get("model"),
            "temperature": question_generation.get("temperature"),
            "successful_count": question_generation.get("successful_count", 0),
            "failed_count": question_generation.get("failed_count", 0),
            "non_distinct_strategy_questions": question_generation.get(
                "non_distinct_strategy_questions", 0
            ),
            "correct_letter_distribution": question_generation.get(
                "correct_letter_distribution", {}
            ),
            "strategy_distribution": question_generation.get("strategy_distribution", {}),
            "token_usage": {
                "input": question_generation.get("total_input_tokens", 0),
                "output": question_generation.get("total_output_tokens", 0),
                "total": question_generation.get("total_tokens", 0),
            },
            "estimated_cost_usd": _get_cost(question_generation),
        },
        "step_5_setup_gold": {
            "model": setup_gold.get("model"),
            "temperature": setup_gold.get("temperature"),
            "total_questions": setup_gold.get("total", 0),
            "correct": setup_gold.get("correct", 0),
            "incorrect": setup_gold.get("incorrect", 0),
            "parse_error": setup_gold.get("parse_error", 0),
            "acc_gold_overall": setup_gold.get("acc_gold_overall", 0.0),
            "acc_gold_per_source": setup_gold.get("acc_gold_per_source", {}),
            "token_usage": {
                "input": setup_gold.get("total_input_tokens", 0),
                "output": setup_gold.get("total_output_tokens", 0),
                "total": setup_gold.get("total_tokens", 0),
            },
            "estimated_cost_usd": _get_cost(setup_gold),
        },
        "cost_summary": {
            "total_estimated_cost_usd": round(total_cost, 6),
            "breakdown": {
                "step_3_af_extraction": _get_cost(af_extraction),
                "step_4_question_generation": _get_cost(question_generation),
                "step_5_setup_gold": _get_cost(setup_gold),
            },
        },
    }

    out_path = config.EXPERIMENT_DIR / "stage_summary.json"
    save_json(summary, out_path)
    print(f"Saved stage summary: {out_path}")


if __name__ == "__main__":
    main()
