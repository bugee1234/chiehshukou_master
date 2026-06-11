from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
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


PRICING = {
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
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
    price = PRICING.get(model_name, PRICING["gpt-4o-mini"])
    input_cost = (input_tokens / 1_000_000) * price["input"]
    output_cost = (output_tokens / 1_000_000) * price["output"]
    return round(input_cost + output_cost, 6)


def _output_paths(model_name: str) -> tuple[Path, Path]:
    return (
        config.DATA_SOLVER_RESULTS_DIR / f"setup_counterfactual_{model_name}_results.jsonl",
        config.DATA_SOLVER_RESULTS_DIR / f"setup_counterfactual_{model_name}_metadata.json",
    )


def run_model(
    model_name: str,
    rows: list[dict[str, Any]],
    prompt_template: str,
    logger,
) -> dict[str, Any]:
    logger.info("Counterfactual solve started for model=%s", model_name)
    client = OpenAIClient(logger=logger)

    results: list[dict[str, Any]] = []
    counts = {"total": len(rows), "context_faithful": 0, "prior_dependent": 0, "other": 0, "parse_error": 0}
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_conf: dict[str, Counter[str]] = defaultdict(Counter)

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    for row in tqdm(rows, desc=f"Counterfactual [{model_name}]", total=len(rows)):
        question_id = str(row.get("question_id", ""))
        af_id = str(row.get("af_id", ""))
        article_id = str(row.get("article_id", ""))
        source_dataset = str(row.get("source_dataset", ""))
        source_text = str(row.get("counterfactual_lay_summary", ""))
        options = row.get("options", [])
        promoted_letter = str(row.get("promoted_distractor_letter", ""))
        original_true_letter = str(row.get("original_true_letter", ""))
        in_true = str(row.get("in_true", ""))
        in_false = str(row.get("in_false", ""))
        confidence = str(row.get("swap_confidence", ""))

        if not isinstance(options, list) or len(options) != 4:
            counts["parse_error"] += 1
            logger.error("[%s] %s parse error | invalid options format", model_name, question_id)
            results.append(
                {
                    "question_id": question_id,
                    "af_id": af_id,
                    "article_id": article_id,
                    "source_dataset": source_dataset,
                    "solver_model": model_name,
                    "predicted_letter": "",
                    "promoted_distractor_letter": promoted_letter,
                    "original_true_letter": original_true_letter,
                    "in_true": in_true,
                    "in_false": in_false,
                    "classification": "",
                    "reasoning": "Invalid options format",
                    "parse_error": True,
                    "swap_confidence": confidence,
                }
            )
            continue

        prompt = (
            prompt_template.replace("{source_text}", source_text)
            .replace("{option_a}", str(options[0]))
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
                    "promoted_distractor_letter": promoted_letter,
                    "original_true_letter": original_true_letter,
                    "in_true": in_true,
                    "in_false": in_false,
                    "classification": "",
                    "reasoning": "",
                    "parse_error": True,
                    "swap_confidence": confidence,
                }
            )
            continue
        except Exception as exc:
            counts["parse_error"] += 1
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
                    "promoted_distractor_letter": promoted_letter,
                    "original_true_letter": original_true_letter,
                    "in_true": in_true,
                    "in_false": in_false,
                    "classification": "",
                    "reasoning": "",
                    "parse_error": True,
                    "swap_confidence": confidence,
                }
            )
            continue

        predicted_letter = parsed.answer
        if predicted_letter == promoted_letter:
            classification = "context_faithful"
        elif predicted_letter == original_true_letter:
            classification = "prior_dependent"
        else:
            classification = "other"

        counts[classification] += 1
        by_source[source_dataset][classification] += 1
        by_conf[confidence][classification] += 1

        results.append(
            {
                "question_id": question_id,
                "af_id": af_id,
                "article_id": article_id,
                "source_dataset": source_dataset,
                "solver_model": model_name,
                "predicted_letter": predicted_letter,
                "promoted_distractor_letter": promoted_letter,
                "original_true_letter": original_true_letter,
                "in_true": in_true,
                "in_false": in_false,
                "classification": classification,
                "reasoning": parsed.reasoning,
                "parse_error": False,
                "swap_confidence": confidence,
            }
        )

    answerable = counts["total"] - counts["parse_error"]
    acc_faithful = _safe_div(counts["context_faithful"], answerable)
    acc_prior = _safe_div(counts["prior_dependent"], answerable)
    acc_other = _safe_div(counts["other"], answerable)
    cost = _estimate_cost_usd(model_name, total_input_tokens, total_output_tokens)

    metadata = {
        "solve_time": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "temperature": 0.0,
        "total": counts["total"],
        "context_faithful": counts["context_faithful"],
        "prior_dependent": counts["prior_dependent"],
        "other": counts["other"],
        "parse_error": counts["parse_error"],
        "acc_cf_faithful": acc_faithful,
        "acc_cf_prior": acc_prior,
        "acc_cf_other": acc_other,
        "per_source_breakdown": {k: dict(v) for k, v in by_source.items()},
        "per_confidence_breakdown": {k: dict(v) for k, v in by_conf.items()},
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost,
    }

    out_results, out_meta = _output_paths(model_name)
    save_jsonl(results, out_results)
    save_json(metadata, out_meta)

    logger.info(
        "Counterfactual solve finished model=%s | faithful=%s prior=%s other=%s parse_error=%s",
        model_name,
        counts["context_faithful"],
        counts["prior_dependent"],
        counts["other"],
        counts["parse_error"],
    )
    logger.info(
        "Model=%s usage | input=%s output=%s total=%s estimated_cost_usd=%s",
        model_name,
        total_input_tokens,
        total_output_tokens,
        total_tokens,
        cost,
    )

    return {"metadata": metadata, "results": results}


def _count_from_results(results: list[dict[str, Any]], source: str, cls: str) -> int:
    return sum(
        1
        for r in results
        if not r.get("parse_error", False)
        and str(r.get("source_dataset", "")) == source
        and str(r.get("classification", "")) == cls
    )


def _count_from_results_conf(results: list[dict[str, Any]], conf: str, cls: str) -> int:
    return sum(
        1
        for r in results
        if not r.get("parse_error", False)
        and str(r.get("swap_confidence", "")) == conf
        and str(r.get("classification", "")) == cls
    )


def main() -> None:
    logger = setup_logger(
        "solve_counterfactual", str(config.LOGS_DIR / "06c_solve_counterfactual.log")
    )
    logger.info("Setup Counterfactual dual-model run started.")

    input_path = config.DATA_QUESTIONS_DIR / "counterfactual_contexts.jsonl"
    prompt_path = config.PROMPTS_DIR / "solver_counterfactual.txt"
    rows = load_jsonl(input_path)
    prompt_template = prompt_path.read_text(encoding="utf-8")

    all_outputs: dict[str, dict[str, Any]] = {}
    for model_name in config.MODEL_BLIND_SOLVERS:
        all_outputs[model_name] = run_model(model_name, rows, prompt_template, logger)

    print("\n=== Setup Counterfactual Results (Dual-Model) ===\n")
    print(
        "Model         Total  Faithful  Prior  Other  ParseErr  "
        "ACC_Faithful  ACC_Prior  ACC_Other  Cost"
    )
    for model_name in config.MODEL_BLIND_SOLVERS:
        m = all_outputs[model_name]["metadata"]
        print(
            f"{model_name:<13} {m['total']:<6} {m['context_faithful']:<9} {m['prior_dependent']:<6} "
            f"{m['other']:<6} {m['parse_error']:<8} {m['acc_cf_faithful']:<13} "
            f"{m['acc_cf_prior']:<10} {m['acc_cf_other']:<10} ${m['estimated_cost_usd']:.4f}"
        )

    print("\n=== Per-Source Breakdown ===")
    print("            gpt-4.1                          gpt-4o-mini")
    print("            Faithful  Prior  Other          Faithful  Prior  Other")
    for source in ["PLOS", "eLife"]:
        a = all_outputs["gpt-4.1"]["results"]
        b = all_outputs["gpt-4o-mini"]["results"]
        print(
            f"{source:<11}"
            f"{_count_from_results(a, source, 'context_faithful'):<10}"
            f"{_count_from_results(a, source, 'prior_dependent'):<7}"
            f"{_count_from_results(a, source, 'other'):<14}"
            f"{_count_from_results(b, source, 'context_faithful'):<10}"
            f"{_count_from_results(b, source, 'prior_dependent'):<7}"
            f"{_count_from_results(b, source, 'other'):<7}"
        )

    print("\n=== Per-Confidence Breakdown ===")
    print("            gpt-4.1                          gpt-4o-mini")
    print("            Faithful  Prior  Other          Faithful  Prior  Other")
    for conf in ["high", "medium", "low"]:
        a = all_outputs["gpt-4.1"]["results"]
        b = all_outputs["gpt-4o-mini"]["results"]
        print(
            f"{conf:<11}"
            f"{_count_from_results_conf(a, conf, 'context_faithful'):<10}"
            f"{_count_from_results_conf(a, conf, 'prior_dependent'):<7}"
            f"{_count_from_results_conf(a, conf, 'other'):<14}"
            f"{_count_from_results_conf(b, conf, 'context_faithful'):<10}"
            f"{_count_from_results_conf(b, conf, 'prior_dependent'):<7}"
            f"{_count_from_results_conf(b, conf, 'other'):<7}"
        )

    print("\n=== 5 Random Samples of Each Classification ===")
    for model_name in config.MODEL_BLIND_SOLVERS:
        prior_samples = [
            r
            for r in all_outputs[model_name]["results"]
            if not r.get("parse_error", False) and r.get("classification") == "prior_dependent"
        ]
        n = min(5, len(prior_samples))
        chosen = random.Random(42).sample(prior_samples, n) if n > 0 else []
        for idx, s in enumerate(chosen, start=1):
            print(f"\n--- Prior-Dependent Sample ({model_name}, {idx}) ---")
            print(f"question_id: {s['question_id']}")
            print(f"swap: \"{s['in_true']}\" → \"{s['in_false']}\"")
            print(f"counterfactual context 支持: {s['promoted_distractor_letter']} (講 {s['in_false']})")
            print(f"solver 卻選: {s['predicted_letter']} (講 {s['in_true']},即 original true)")
            print(f"reasoning: {s['reasoning']}")

    total_input_tokens = sum(
        int(all_outputs[m]["metadata"].get("total_input_tokens", 0))
        for m in config.MODEL_BLIND_SOLVERS
    )
    total_output_tokens = sum(
        int(all_outputs[m]["metadata"].get("total_output_tokens", 0))
        for m in config.MODEL_BLIND_SOLVERS
    )
    total_tokens = sum(
        int(all_outputs[m]["metadata"].get("total_tokens", 0))
        for m in config.MODEL_BLIND_SOLVERS
    )
    total_cost = round(
        sum(float(all_outputs[m]["metadata"].get("estimated_cost_usd", 0.0)) for m in config.MODEL_BLIND_SOLVERS),
        6,
    )
    print("\n=== Overall Token Usage & Cost ===")
    print(f"input_tokens: {total_input_tokens}")
    print(f"output_tokens: {total_output_tokens}")
    print(f"total_tokens: {total_tokens}")
    print(f"total_estimated_cost_usd: {total_cost}")


if __name__ == "__main__":
    main()
