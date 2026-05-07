from __future__ import annotations

import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Literal, Optional

from pydantic import BaseModel, ValidationError
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.utils import OpenAIClient, load_jsonl, save_json, save_jsonl, setup_logger


class SwapPair(BaseModel):
    in_true: Optional[str]
    in_false: Optional[str]
    confidence: Literal["high", "medium", "low"]


def _estimate_cost_usd_gpt41(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * 2.00
    output_cost = (output_tokens / 1_000_000) * 8.00
    return round(input_cost + output_cost, 6)


def _replace_with_fallback(text: str, in_true: str, in_false: str) -> tuple[str, int, str | None]:
    count_cs = text.count(in_true)
    if count_cs > 0:
        replaced = text.replace(in_true, in_false)
        return replaced, count_cs, in_true

    pattern = re.compile(re.escape(in_true), flags=re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return text, 0, None

    replacement_target = match.group(0)
    replacement_count = text.count(replacement_target)
    replaced = text.replace(replacement_target, in_false)
    return replaced, replacement_count, replacement_target


def _find_promoted_letter(options: list[str], target: str) -> str | None:
    for idx, opt in enumerate(options):
        if opt == target:
            return "ABCD"[idx]
    for idx, opt in enumerate(options):
        if str(opt).strip() == str(target).strip():
            return "ABCD"[idx]
    return None


def main() -> None:
    logger = setup_logger("counterfactual_build", str(config.LOGS_DIR / "06b_counterfactual_build.log"))
    logger.info("Counterfactual build started.")

    questions_path = config.DATA_QUESTIONS_DIR / "questions.jsonl"
    prompt_path = config.PROMPTS_DIR / "counterfactual_extract_swap.txt"
    out_passed_path = config.DATA_QUESTIONS_DIR / "counterfactual_contexts.jsonl"
    out_skipped_path = config.DATA_QUESTIONS_DIR / "counterfactual_skipped.jsonl"
    out_meta_path = config.DATA_QUESTIONS_DIR / "counterfactual_metadata.json"

    questions = load_jsonl(questions_path)
    prompt_template = prompt_path.read_text(encoding="utf-8")
    client = OpenAIClient(logger=logger)

    passed_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    skip_counter: Counter[str] = Counter()
    confidence_counter: Counter[str] = Counter()

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    replacement_counts: list[int] = []

    for q in tqdm(questions, desc="Building counterfactual contexts", total=len(questions)):
        question_id = str(q.get("question_id", ""))
        af_id = str(q.get("af_id", ""))
        article_id = str(q.get("article_id", ""))
        source_dataset = str(q.get("source_dataset", ""))
        lay_summary = str(q.get("lay_summary", ""))
        options = q.get("options", [])
        atomic_fact = str(q.get("atomic_fact", ""))
        original_true_letter = str(q.get("correct_letter", ""))
        true_statement = str(q.get("true_statement", ""))
        false_statements = q.get("false_statements", [])

        entity_swap_item = None
        if isinstance(false_statements, list):
            for item in false_statements:
                if isinstance(item, dict) and item.get("strategy") == "entity_swap":
                    entity_swap_item = item
                    break

        if entity_swap_item is None:
            skip_counter["no_entity_swap"] += 1
            skipped_records.append(
                {
                    "question_id": question_id,
                    "skip_reason": "no_entity_swap",
                    "details": "No entity_swap distractor in false_statements.",
                }
            )
            logger.info("%s skipped | no_entity_swap", question_id)
            continue

        false_statement_text = str(entity_swap_item.get("text", ""))
        prompt = (
            prompt_template.replace("{true_statement}", true_statement)
            .replace("{false_statement}", false_statement_text)
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
            swap = SwapPair.model_validate(json.loads(raw_content))
            confidence_counter[swap.confidence] += 1
        except (json.JSONDecodeError, ValidationError) as exc:
            skip_counter["no_clean_swap"] += 1
            skipped_records.append(
                {
                    "question_id": question_id,
                    "skip_reason": "no_clean_swap",
                    "details": f"Parse/validation error: {exc}",
                }
            )
            logger.warning(
                "%s skipped | no_clean_swap | err=%s | raw_output_preview=%s",
                question_id,
                exc,
                raw_content[:500],
            )
            continue
        except Exception as exc:
            skip_counter["no_clean_swap"] += 1
            skipped_records.append(
                {
                    "question_id": question_id,
                    "skip_reason": "no_clean_swap",
                    "details": f"Runtime/API error: {exc}",
                }
            )
            logger.warning("%s skipped | no_clean_swap | err=%s", question_id, exc)
            continue

        if swap.in_true is None or swap.in_false is None:
            skip_counter["no_clean_swap"] += 1
            skipped_records.append(
                {
                    "question_id": question_id,
                    "skip_reason": "no_clean_swap",
                    "details": f"Swap pair has null entity values (confidence={swap.confidence}).",
                }
            )
            logger.info("%s skipped | no_clean_swap | null swap pair", question_id)
            continue

        if swap.confidence == "low":
            skip_counter["low_confidence"] += 1
            skipped_records.append(
                {
                    "question_id": question_id,
                    "skip_reason": "low_confidence",
                    "details": f"Swap confidence is low for pair {swap.in_true} -> {swap.in_false}.",
                }
            )
            logger.info("%s skipped | low_confidence", question_id)
            continue

        counterfactual_context, replacement_count, matched_text = _replace_with_fallback(
            lay_summary, swap.in_true, swap.in_false
        )
        if replacement_count <= 0:
            skip_counter["entity_not_found"] += 1
            skipped_records.append(
                {
                    "question_id": question_id,
                    "skip_reason": "entity_not_found",
                    "details": f"Entity '{swap.in_true}' not found in lay_summary.",
                }
            )
            logger.info("%s skipped | entity_not_found | entity=%s", question_id, swap.in_true)
            continue

        promoted_letter = _find_promoted_letter(options, false_statement_text)
        if promoted_letter is None:
            skip_counter["no_clean_swap"] += 1
            skipped_records.append(
                {
                    "question_id": question_id,
                    "skip_reason": "no_clean_swap",
                    "details": "Entity-swap distractor text not found in options.",
                }
            )
            logger.info("%s skipped | no_clean_swap | distractor text not in options", question_id)
            continue

        replacement_counts.append(replacement_count)
        passed_records.append(
            {
                "question_id": question_id,
                "af_id": af_id,
                "article_id": article_id,
                "source_dataset": source_dataset,
                "original_lay_summary": lay_summary,
                "counterfactual_lay_summary": counterfactual_context,
                "in_true": swap.in_true,
                "in_false": swap.in_false,
                "replacement_count": replacement_count,
                "swap_confidence": swap.confidence,
                "promoted_distractor_text": false_statement_text,
                "promoted_distractor_letter": promoted_letter,
                "original_true_letter": original_true_letter,
                "options": options,
                "atomic_fact": atomic_fact,
                "matched_text_in_context": matched_text,
            }
        )
        logger.info(
            "%s passed | %s -> %s | confidence=%s | replacement_count=%s",
            question_id,
            swap.in_true,
            swap.in_false,
            swap.confidence,
            replacement_count,
        )

    save_jsonl(passed_records, out_passed_path)
    save_jsonl(skipped_records, out_skipped_path)

    avg_replacement_count = round(sum(replacement_counts) / len(replacement_counts), 2) if replacement_counts else 0.0
    estimated_cost = _estimate_cost_usd_gpt41(total_input_tokens, total_output_tokens)

    metadata = {
        "build_time": datetime.now().isoformat(timespec="seconds"),
        "model": config.MODEL_GOLD_SOLVER,
        "temperature": 0.0,
        "total_questions": len(questions),
        "passed_count": len(passed_records),
        "skip_counts": {
            "no_entity_swap": int(skip_counter.get("no_entity_swap", 0)),
            "no_clean_swap": int(skip_counter.get("no_clean_swap", 0)),
            "low_confidence": int(skip_counter.get("low_confidence", 0)),
            "entity_not_found": int(skip_counter.get("entity_not_found", 0)),
        },
        "confidence_distribution": {
            "high": int(confidence_counter.get("high", 0)),
            "medium": int(confidence_counter.get("medium", 0)),
            "low": int(confidence_counter.get("low", 0)),
        },
        "average_replacement_count": avg_replacement_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost,
    }
    save_json(metadata, out_meta_path)

    print("\n=== Counterfactual Build Sanity Check ===")
    print(f"Total questions: {len(questions)}")
    print(f"Passed (counterfactual built): {len(passed_records)}")
    print("Skipped breakdown:")
    print(f"  no_entity_swap: {skip_counter.get('no_entity_swap', 0)}")
    print(f"  no_clean_swap: {skip_counter.get('no_clean_swap', 0)}")
    print(f"  low_confidence: {skip_counter.get('low_confidence', 0)}")
    print(f"  entity_not_found: {skip_counter.get('entity_not_found', 0)}")
    print(f"\nAverage replacement_count: {avg_replacement_count}")
    print(
        "Confidence distribution: "
        f"high={confidence_counter.get('high', 0)} "
        f"medium={confidence_counter.get('medium', 0)} "
        f"low={confidence_counter.get('low', 0)}"
    )

    print("\n=== 5 Random Samples (manual review needed) ===")
    sample_n = min(5, len(passed_records))
    if sample_n > 0:
        samples = random.Random(42).sample(passed_records, sample_n)
        for i, sample in enumerate(samples, start=1):
            promoted_letter = sample["promoted_distractor_letter"]
            promoted_idx = "ABCD".find(promoted_letter)
            original_true_idx = "ABCD".find(sample["original_true_letter"])
            promoted_text = (
                sample["options"][promoted_idx] if 0 <= promoted_idx < 4 else sample["promoted_distractor_text"]
            )
            original_true_text = (
                sample["options"][original_true_idx] if 0 <= original_true_idx < 4 else ""
            )

            print(f"\n--- Sample {i} ---")
            print(f"question_id: {sample['question_id']}")
            print(
                f"swap: \"{sample['in_true']}\" → \"{sample['in_false']}\" "
                f"(confidence: {sample['swap_confidence']}, replaced {sample['replacement_count']} times)"
            )
            print("\nORIGINAL lay_summary (前 600 字):")
            print(sample["original_lay_summary"][:600])
            print("\nCOUNTERFACTUAL lay_summary (前 600 字):")
            print(sample["counterfactual_lay_summary"][:600])
            print("\ntarget distractor (counterfactual setup 下「應該」的正解):")
            print(f"{promoted_letter}. {promoted_text}")
            print("\noriginal true (counterfactual 下變錯):")
            print(f"{sample['original_true_letter']}. {original_true_text}")

    print("\n=== Token Usage & Cost ===")
    print(f"input_tokens: {total_input_tokens}")
    print(f"output_tokens: {total_output_tokens}")
    print(f"total_tokens: {total_tokens}")
    print(f"estimated_cost_usd: {estimated_cost}")

    logger.info(
        "Counterfactual build finished | total=%s passed=%s skipped=%s",
        len(questions),
        len(passed_records),
        len(skipped_records),
    )
    logger.info(
        "Token usage | input=%s output=%s total=%s estimated_cost_usd=%s",
        total_input_tokens,
        total_output_tokens,
        total_tokens,
        estimated_cost,
    )


if __name__ == "__main__":
    main()
