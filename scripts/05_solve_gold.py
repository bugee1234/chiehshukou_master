from __future__ import annotations

import json
import random
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Literal

from pydantic import BaseModel, ValidationError
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.utils import OpenAIClient, load_jsonl, save_json, save_jsonl, setup_logger


AnswerLetter = Literal["A", "B", "C", "D"]


class SolverResponse(BaseModel):
    answer: AnswerLetter
    reasoning: str


def _estimate_cost_usd_gpt41(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * 2.0
    output_cost = (output_tokens / 1_000_000) * 8.0
    return round(input_cost + output_cost, 6)


def _safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def main() -> None:
    logger = setup_logger("solve_gold", str(config.LOGS_DIR / "05_solve_gold.log"))
    logger.info("Setup Gold solving started.")

    questions_path = config.DATA_QUESTIONS_DIR / "questions.jsonl"
    prompt_path = config.PROMPTS_DIR / "solver_gold.txt"
    out_results_path = config.DATA_SOLVER_RESULTS_DIR / "setup_gold_results.jsonl"
    out_meta_path = config.DATA_SOLVER_RESULTS_DIR / "setup_gold_metadata.json"

    questions = load_jsonl(questions_path)
    prompt_template = prompt_path.read_text(encoding="utf-8")
    client = OpenAIClient(logger=logger)

    results: list[dict[str, Any]] = []
    wrong_records: list[dict[str, Any]] = []

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    counts = {
        "total": len(questions),
        "correct": 0,
        "incorrect": 0,
        "parse_error": 0,
    }
    source_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "incorrect": 0, "parse_error": 0}
    )

    for q in tqdm(questions, desc="Solving Setup Gold", total=len(questions)):
        question_id = str(q.get("question_id", ""))
        af_id = str(q.get("af_id", ""))
        article_id = str(q.get("article_id", ""))
        source_dataset = str(q.get("source_dataset", ""))
        lay_summary = str(q.get("lay_summary", ""))
        options = q.get("options", [])
        correct_letter = str(q.get("correct_letter", ""))

        source_stats[source_dataset]["total"] += 1
        atomic_fact = str(q.get("atomic_fact", ""))

        if not isinstance(options, list) or len(options) != 4:
            counts["parse_error"] += 1
            source_stats[source_dataset]["parse_error"] += 1
            logger.error("%s parse error | invalid options format in questions.jsonl", question_id)
            results.append(
                {
                    "question_id": question_id,
                    "af_id": af_id,
                    "article_id": article_id,
                    "source_dataset": source_dataset,
                    "solver_model": config.MODEL_GOLD_SOLVER,
                    "predicted_letter": "",
                    "correct_letter": correct_letter,
                    "is_correct": False,
                    "reasoning": "Invalid question options format.",
                    "parse_error": True,
                }
            )
            continue

        prompt = (
            prompt_template.replace("{source_text}", lay_summary)
            .replace("{option_a}", str(options[0]))
            .replace("{option_b}", str(options[1]))
            .replace("{option_c}", str(options[2]))
            .replace("{option_d}", str(options[3]))
        )

        raw_content = ""
        try:
            response = client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=config.MODEL_GOLD_SOLVER,
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
                "%s parse error | %s | raw_output_preview=%s",
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
                    "solver_model": config.MODEL_GOLD_SOLVER,
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
                "%s runtime error | %s | raw_output_preview=%s",
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
                    "solver_model": config.MODEL_GOLD_SOLVER,
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
            logger.info("%s correct | pred=%s", question_id, parsed.answer)
        else:
            counts["incorrect"] += 1
            source_stats[source_dataset]["incorrect"] += 1
            logger.info("%s incorrect | pred=%s gold=%s", question_id, parsed.answer, correct_letter)
            wrong_records.append(
                {
                    "question_id": question_id,
                    "atomic_fact": atomic_fact,
                    "options": options,
                    "correct": correct_letter,
                    "predicted": parsed.answer,
                    "reasoning": parsed.reasoning,
                }
            )

        results.append(
            {
                "question_id": question_id,
                "af_id": af_id,
                "article_id": article_id,
                "source_dataset": source_dataset,
                "solver_model": config.MODEL_GOLD_SOLVER,
                "predicted_letter": parsed.answer,
                "correct_letter": correct_letter,
                "is_correct": is_correct,
                "reasoning": parsed.reasoning,
                "parse_error": False,
            }
        )

    total_answerable = counts["total"] - counts["parse_error"]
    acc_gold = _safe_div(counts["correct"], total_answerable)

    acc_per_source: dict[str, float] = {}
    for source_name, stat in source_stats.items():
        answerable = stat["total"] - stat["parse_error"]
        acc_per_source[source_name] = _safe_div(stat["correct"], answerable)

    estimated_cost = _estimate_cost_usd_gpt41(total_input_tokens, total_output_tokens)

    metadata = {
        "solve_time": datetime.now().isoformat(timespec="seconds"),
        "model": config.MODEL_GOLD_SOLVER,
        "temperature": 0.0,
        "total": counts["total"],
        "correct": counts["correct"],
        "incorrect": counts["incorrect"],
        "parse_error": counts["parse_error"],
        "acc_gold_overall": acc_gold,
        "acc_gold_per_source": acc_per_source,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost,
    }

    save_jsonl(results, out_results_path)
    save_json(metadata, out_meta_path)

    print("\n=== Setup Gold Sanity Check ===")
    print(f"Total questions: {counts['total']}")
    print(f"Correct: {counts['correct']}")
    print(f"Incorrect: {counts['incorrect']}")
    print(f"Parse errors: {counts['parse_error']}")
    print(f"ACC_Gold overall: {acc_gold}")
    for source_name in sorted(acc_per_source):
        print(f"ACC_Gold {source_name}: {acc_per_source[source_name]}")

    sample_n = min(5, len(wrong_records))
    if sample_n > 0:
        samples = random.Random(42).sample(wrong_records, sample_n)
        for i, sample in enumerate(samples, start=1):
            print(f"\n--- Wrong Sample {i} ---")
            print(f"question_id: {sample['question_id']}")
            print(f"atomic_fact: {sample['atomic_fact']}")
            print("options:")
            print(f"  A. {sample['options'][0]}")
            print(f"  B. {sample['options'][1]}")
            print(f"  C. {sample['options'][2]}")
            print(f"  D. {sample['options'][3]}")
            print(f"correct: {sample['correct']}")
            print(f"predicted: {sample['predicted']}")
            print(f"reasoning: {sample['reasoning']}")

    print("\n=== Token Usage & Cost ===")
    print(f"input_tokens: {total_input_tokens}")
    print(f"output_tokens: {total_output_tokens}")
    print(f"total_tokens: {total_tokens}")
    print(f"estimated_cost_usd: {estimated_cost}")
    print(f"Saved results JSONL: {out_results_path}")
    print(f"Saved metadata JSON: {out_meta_path}")

    logger.info(
        "Setup Gold finished | total=%s correct=%s incorrect=%s parse_error=%s acc=%s",
        counts["total"],
        counts["correct"],
        counts["incorrect"],
        counts["parse_error"],
        acc_gold,
    )
    logger.info(
        "Token usage | input=%s output=%s total=%s estimated_cost_usd=%s",
        total_input_tokens,
        total_output_tokens,
        total_tokens,
        estimated_cost,
    )
    try:
        subprocess.run(
            [sys.executable, str(config.ROOT_DIR / "scripts" / "99_build_stage_summary.py")],
            check=True,
        )
        logger.info("Stage summary auto-updated.")
    except Exception as exc:
        logger.warning("Stage summary auto-update failed: %s", exc)

    if acc_gold < 0.90:
        print("\n[STOP] ACC_Gold < 0.90. Stop before Step 6 and review wrong samples first.")


if __name__ == "__main__":
    main()
