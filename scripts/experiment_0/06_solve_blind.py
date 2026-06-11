from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Any, Literal

from pydantic import BaseModel, ValidationError
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.utils import OpenAIClient, load_jsonl, save_json, save_jsonl, setup_logger


PRICING = {
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

DISTRACTOR_STRATEGIES = {
    "numerical_perturbation",
    "entity_swap",
    "direction_reversal",
    "detail_fabrication",
}

AnswerLetter = Literal["A", "B", "C", "D"]


class SolverResponse(BaseModel):
    answer: AnswerLetter
    reasoning: str


def _safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _estimate_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING.get(model_name)
    if not pricing:
        return 0.0
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


def _model_key(model_name: str) -> str:
    return model_name


def _model_output_paths(model_name: str) -> tuple[Path, Path]:
    key = _model_key(model_name)
    results_path = config.DATA_SOLVER_RESULTS_DIR / f"setup_blind_{key}_results.jsonl"
    metadata_path = config.DATA_SOLVER_RESULTS_DIR / f"setup_blind_{key}_metadata.json"
    return results_path, metadata_path


def run_setup_blind(
    model_name: str, questions: list[dict[str, Any]], prompt_template: str, logger
) -> dict[str, Any]:
    logger.info("Setup Blind started for model=%s", model_name)
    client = OpenAIClient(logger=logger)

    results: list[dict[str, Any]] = []
    counts = {"total": len(questions), "correct": 0, "incorrect": 0, "parse_error": 0}
    source_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "incorrect": 0, "parse_error": 0}
    )
    wrong_pick_strategy_counter: Counter[str] = Counter()

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    for q in tqdm(questions, desc=f"Blind solving [{model_name}]", total=len(questions)):
        question_id = str(q.get("question_id", ""))
        af_id = str(q.get("af_id", ""))
        article_id = str(q.get("article_id", ""))
        source_dataset = str(q.get("source_dataset", ""))
        options = q.get("options", [])
        correct_letter = str(q.get("correct_letter", ""))
        false_statements = q.get("false_statements", [])
        source_stats[source_dataset]["total"] += 1

        if not isinstance(options, list) or len(options) != 4:
            counts["parse_error"] += 1
            source_stats[source_dataset]["parse_error"] += 1
            logger.error("[%s] %s parse error | invalid options format", model_name, question_id)
            results.append(
                {
                    "question_id": question_id,
                    "af_id": af_id,
                    "article_id": article_id,
                    "source_dataset": source_dataset,
                    "solver_model": model_name,
                    "predicted_letter": "",
                    "correct_letter": correct_letter,
                    "is_correct": False,
                    "reasoning": "Invalid question options format.",
                    "parse_error": True,
                }
            )
            continue

        prompt = (
            prompt_template.replace("{option_a}", str(options[0]))
            .replace("{option_b}", str(options[1]))
            .replace("{option_c}", str(options[2]))
            .replace("{option_d}", str(options[3]))
        )

        raw_content = ""
        try:
            response = client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            usage = response.get("usage", {})
            total_input_tokens += int(usage.get("prompt_tokens", 0))
            total_output_tokens += int(usage.get("completion_tokens", 0))
            total_tokens += int(usage.get("total_tokens", 0))

            raw_content = str(response.get("content", ""))
            parsed = SolverResponse.model_validate(json.loads(raw_content))
        except (json.JSONDecodeError, ValidationError) as exc:
            counts["parse_error"] += 1
            source_stats[source_dataset]["parse_error"] += 1
            logger.error(
                "[%s] %s parse error | %s | raw_output_preview=%s",
                model_name,
                question_id,
                exc,
                raw_content[:500],
            )
            results.append(
                {
                    "question_id": question_id,
                    "af_id": af_id,
                    "article_id": article_id,
                    "source_dataset": source_dataset,
                    "solver_model": model_name,
                    "predicted_letter": "",
                    "correct_letter": correct_letter,
                    "is_correct": False,
                    "reasoning": "",
                    "parse_error": True,
                }
            )
            continue
        except Exception as exc:
            counts["parse_error"] += 1
            source_stats[source_dataset]["parse_error"] += 1
            logger.error(
                "[%s] %s runtime error | %s | raw_output_preview=%s",
                model_name,
                question_id,
                exc,
                raw_content[:500],
            )
            results.append(
                {
                    "question_id": question_id,
                    "af_id": af_id,
                    "article_id": article_id,
                    "source_dataset": source_dataset,
                    "solver_model": model_name,
                    "predicted_letter": "",
                    "correct_letter": correct_letter,
                    "is_correct": False,
                    "reasoning": "",
                    "parse_error": True,
                }
            )
            continue

        is_correct = parsed.answer == correct_letter
        if is_correct:
            counts["correct"] += 1
            source_stats[source_dataset]["correct"] += 1
        else:
            counts["incorrect"] += 1
            source_stats[source_dataset]["incorrect"] += 1
            predicted_index = "ABCD".find(parsed.answer)
            if 0 <= predicted_index < 4:
                picked_text = str(options[predicted_index])
                strategy_map = {
                    str(item.get("text", "")): str(item.get("strategy", "unknown"))
                    for item in false_statements
                    if isinstance(item, dict)
                }
                picked_strategy = strategy_map.get(picked_text, "unknown")
                wrong_pick_strategy_counter[picked_strategy] += 1

        results.append(
            {
                "question_id": question_id,
                "af_id": af_id,
                "article_id": article_id,
                "source_dataset": source_dataset,
                "solver_model": model_name,
                "predicted_letter": parsed.answer,
                "correct_letter": correct_letter,
                "is_correct": is_correct,
                "reasoning": parsed.reasoning,
                "parse_error": False,
            }
        )

    answerable_total = counts["total"] - counts["parse_error"]
    acc_blind = _safe_div(counts["correct"], answerable_total)
    acc_per_source: dict[str, float] = {}
    for source_name, stat in source_stats.items():
        source_answerable = stat["total"] - stat["parse_error"]
        acc_per_source[source_name] = _safe_div(stat["correct"], source_answerable)

    estimated_cost = _estimate_cost_usd(model_name, total_input_tokens, total_output_tokens)
    metadata = {
        "solve_time": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "temperature": 0.0,
        "total": counts["total"],
        "correct": counts["correct"],
        "incorrect": counts["incorrect"],
        "parse_error": counts["parse_error"],
        "acc_blind_overall": acc_blind,
        "acc_blind_per_source": acc_per_source,
        "wrong_pick_strategy_counts": dict(wrong_pick_strategy_counter),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost,
    }

    out_results, out_meta = _model_output_paths(model_name)
    save_jsonl(results, out_results)
    save_json(metadata, out_meta)
    logger.info(
        "Setup Blind finished model=%s | total=%s correct=%s incorrect=%s parse_error=%s acc=%s",
        model_name,
        counts["total"],
        counts["correct"],
        counts["incorrect"],
        counts["parse_error"],
        acc_blind,
    )
    logger.info(
        "Model=%s usage | input=%s output=%s total=%s estimated_cost_usd=%s",
        model_name,
        total_input_tokens,
        total_output_tokens,
        total_tokens,
        estimated_cost,
    )
    return metadata


def _format_pct(count: int, denom: int) -> str:
    if denom == 0:
        return "0.0%"
    return f"{(count / denom) * 100:.1f}%"


def main() -> None:
    logger = setup_logger("solve_blind", str(config.LOGS_DIR / "06_solve_blind.log"))
    logger.info("Setup Blind dual-model run started.")

    questions_path = config.DATA_QUESTIONS_DIR / "questions.jsonl"
    prompt_path = config.PROMPTS_DIR / "solver_blind.txt"
    questions = load_jsonl(questions_path)
    prompt_template = prompt_path.read_text(encoding="utf-8")

    model_results: dict[str, dict[str, Any]] = {}
    for model_name in config.MODEL_BLIND_SOLVERS:
        model_results[model_name] = run_setup_blind(model_name, questions, prompt_template, logger)

    print("\n=== Setup Blind Results (Dual-Model) ===\n")
    print("Model              Total  Correct  Acc     Acc(PLOS)  Acc(eLife)  ParseErr  Cost")
    for model_name in config.MODEL_BLIND_SOLVERS:
        m = model_results[model_name]
        acc_plos = m.get("acc_blind_per_source", {}).get("PLOS", 0.0)
        acc_elife = m.get("acc_blind_per_source", {}).get("eLife", 0.0)
        print(
            f"{model_name:<18} "
            f"{m.get('total', 0):<6} "
            f"{m.get('correct', 0):<8} "
            f"{m.get('acc_blind_overall', 0.0):<7} "
            f"{acc_plos:<10} "
            f"{acc_elife:<11} "
            f"{m.get('parse_error', 0):<8} "
            f"${m.get('estimated_cost_usd', 0.0):.4f}"
        )

    print("\nRandom baseline: 0.2500")

    print("\n=== Per-Strategy Analysis (which distractors got picked when wrong) ===")
    for model_name in config.MODEL_BLIND_SOLVERS:
        m = model_results[model_name]
        strat_counts = m.get("wrong_pick_strategy_counts", {})
        wrong_total = m.get("incorrect", 0)
        print(f"\n{model_name} wrong picks by strategy of chosen distractor:")
        for strategy in sorted(DISTRACTOR_STRATEGIES):
            c = int(strat_counts.get(strategy, 0))
            print(f"  {strategy}: {c}  ({_format_pct(c, wrong_total)} of wrong picks)")
        unknown = int(strat_counts.get("unknown", 0))
        if unknown > 0:
            print(f"  unknown: {unknown}  ({_format_pct(unknown, wrong_total)} of wrong picks)")

    total_input_tokens = sum(int(model_results[m].get("total_input_tokens", 0)) for m in model_results)
    total_output_tokens = sum(int(model_results[m].get("total_output_tokens", 0)) for m in model_results)
    total_tokens = sum(int(model_results[m].get("total_tokens", 0)) for m in model_results)
    total_cost = round(sum(float(model_results[m].get("estimated_cost_usd", 0.0)) for m in model_results), 6)

    print("\n=== Overall Token Usage & Cost ===")
    print(f"input_tokens: {total_input_tokens}")
    print(f"output_tokens: {total_output_tokens}")
    print(f"total_tokens: {total_tokens}")
    print(f"total_estimated_cost_usd: {total_cost}")

    try:
        subprocess.run(
            [sys.executable, str(config.ROOT_DIR / "scripts" / "99_build_stage_summary.py")],
            check=True,
        )
        logger.info("Stage summary auto-updated.")
    except Exception as exc:
        logger.warning("Stage summary auto-update failed: %s", exc)


if __name__ == "__main__":
    main()
