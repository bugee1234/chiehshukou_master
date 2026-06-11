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


class VariantList(BaseModel):
    variants: list[str]


class MappingItem(BaseModel):
    from_: str
    to: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MappingItem":
        return cls(from_=str(d.get("from", "")), to=str(d.get("to", "")))


def _estimate_cost_usd_gpt41(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * 2.00
    output_cost = (output_tokens / 1_000_000) * 8.00
    return round(input_cost + output_cost, 6)


def _find_promoted_letter(options: list[str], target: str) -> str | None:
    for idx, opt in enumerate(options):
        if opt == target:
            return "ABCD"[idx]
    for idx, opt in enumerate(options):
        if str(opt).strip() == str(target).strip():
            return "ABCD"[idx]
    return None


def _contains_with_fallback(text: str, token: str) -> bool:
    if token in text:
        return True
    return re.search(re.escape(token), text, flags=re.IGNORECASE) is not None


def _normalize_variants(variants: list[str], source: str, canonical: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in variants:
        vv = str(v).strip()
        if not vv:
            continue
        if vv not in source:
            continue
        if vv not in seen:
            out.append(vv)
            seen.add(vv)
    if canonical in source and canonical not in seen:
        out.insert(0, canonical)
    return out


def _apply_variant_mapping(source: str, mapping_items: list[MappingItem]) -> tuple[str, int]:
    counterfactual = source
    total_replacements = 0
    dedup: dict[str, str] = {}
    for m in mapping_items:
        if m.from_ and m.to and m.from_ != m.to:
            dedup[m.from_] = m.to

    sorted_pairs = sorted(dedup.items(), key=lambda x: -len(x[0]))
    for from_text, to_text in sorted_pairs:
        count = counterfactual.count(from_text)
        if count > 0:
            counterfactual = counterfactual.replace(from_text, to_text)
            total_replacements += count
    return counterfactual, total_replacements


def main() -> None:
    logger = setup_logger("counterfactual_build", str(config.LOGS_DIR / "06b_counterfactual_build.log"))
    logger.info("Counterfactual build v2 started.")

    questions_path = config.DATA_QUESTIONS_DIR / "questions.jsonl"
    extract_prompt_path = config.PROMPTS_DIR / "counterfactual_extract_swap.txt"
    find_variants_prompt_path = config.PROMPTS_DIR / "counterfactual_find_variants.txt"
    map_variants_prompt_path = config.PROMPTS_DIR / "counterfactual_map_variants.txt"

    out_passed_path = config.DATA_QUESTIONS_DIR / "counterfactual_contexts.jsonl"
    out_skipped_path = config.DATA_QUESTIONS_DIR / "counterfactual_skipped.jsonl"
    out_meta_path = config.DATA_QUESTIONS_DIR / "counterfactual_metadata.json"

    questions = load_jsonl(questions_path)
    extract_prompt = extract_prompt_path.read_text(encoding="utf-8")
    find_variants_prompt = find_variants_prompt_path.read_text(encoding="utf-8")
    map_variants_prompt = map_variants_prompt_path.read_text(encoding="utf-8")

    client = OpenAIClient(logger=logger)

    passed_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    skip_counter: Counter[str] = Counter()
    confidence_counter: Counter[str] = Counter()
    replacement_counts: list[int] = []
    variants_count_list: list[int] = []

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    variant_detection_attempts = 0
    variant_detection_success = 0
    extra_variant_forms_beyond_canonical = 0
    abbreviation_or_shortform_cases = 0

    for q in tqdm(questions, desc="Building counterfactual contexts v2", total=len(questions)):
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
                {"question_id": question_id, "skip_reason": "no_entity_swap", "details": "No entity_swap distractor in false_statements."}
            )
            continue

        false_statement_text = str(entity_swap_item.get("text", ""))

        # Step 1: extract in_true/in_false
        raw_content = ""
        try:
            prompt = extract_prompt.replace("{true_statement}", true_statement).replace(
                "{false_statement}", false_statement_text
            )
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
                {"question_id": question_id, "skip_reason": "no_clean_swap", "details": f"Extract parse/validation error: {exc}"}
            )
            logger.warning("%s skipped | no_clean_swap | extract_err=%s", question_id, exc)
            continue
        except Exception as exc:
            skip_counter["no_clean_swap"] += 1
            skipped_records.append(
                {"question_id": question_id, "skip_reason": "no_clean_swap", "details": f"Extract runtime/API error: {exc}"}
            )
            logger.warning("%s skipped | no_clean_swap | extract_runtime=%s", question_id, exc)
            continue

        if swap.in_true is None or swap.in_false is None:
            skip_counter["no_clean_swap"] += 1
            skipped_records.append(
                {"question_id": question_id, "skip_reason": "no_clean_swap", "details": f"Swap pair has null entity values (confidence={swap.confidence})."}
            )
            continue

        if swap.confidence == "low":
            skip_counter["low_confidence"] += 1
            skipped_records.append(
                {"question_id": question_id, "skip_reason": "low_confidence", "details": f"Swap confidence is low for pair {swap.in_true} -> {swap.in_false}."}
            )
            continue

        # keep old prerequisite
        if not _contains_with_fallback(lay_summary, swap.in_true):
            skip_counter["entity_not_found"] += 1
            skipped_records.append(
                {"question_id": question_id, "skip_reason": "entity_not_found", "details": f"Entity '{swap.in_true}' not found in lay_summary."}
            )
            continue

        # Step 2: find variants
        variant_detection_attempts += 1
        try:
            prompt = find_variants_prompt.replace("{entity}", swap.in_true).replace(
                "{source_text}", lay_summary
            )
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
            parsed = VariantList.model_validate(json.loads(raw_content))
            variants_found = _normalize_variants(parsed.variants, lay_summary, swap.in_true)
        except (json.JSONDecodeError, ValidationError) as exc:
            skip_counter["no_variant_mapping"] += 1
            skipped_records.append(
                {"question_id": question_id, "skip_reason": "no_variant_mapping", "details": f"Variant parse/validation error: {exc}"}
            )
            logger.warning("%s skipped | no_variant_mapping | variant_err=%s", question_id, exc)
            continue
        except Exception as exc:
            skip_counter["no_variant_mapping"] += 1
            skipped_records.append(
                {"question_id": question_id, "skip_reason": "no_variant_mapping", "details": f"Variant runtime/API error: {exc}"}
            )
            logger.warning("%s skipped | no_variant_mapping | variant_runtime=%s", question_id, exc)
            continue

        if not variants_found:
            skip_counter["no_variant_mapping"] += 1
            skipped_records.append(
                {"question_id": question_id, "skip_reason": "no_variant_mapping", "details": "No usable variants found in source text."}
            )
            continue

        variant_detection_success += 1
        variants_count_list.append(len(variants_found))
        extra_variant_forms_beyond_canonical += max(0, len(variants_found) - 1)
        if any(
            v != swap.in_true and (re.search(r"[A-Z]{2,}", v) or "." in v)
            for v in variants_found
        ):
            abbreviation_or_shortform_cases += 1

        # Step 3: map variants
        variants_list_json = json.dumps(variants_found, ensure_ascii=False)
        try:
            prompt = (
                map_variants_prompt.replace("{in_true}", swap.in_true)
                .replace("{in_false}", swap.in_false)
                .replace("{variants_list}", variants_list_json)
            )
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
            parsed_json = json.loads(raw_content)
            mapping_raw = parsed_json.get("mapping", [])
            mapping_items = [MappingItem.from_dict(m) for m in mapping_raw if isinstance(m, dict)]
            # Keep only mapping forms that are in variants_found
            mapping_items = [m for m in mapping_items if m.from_ in variants_found and m.to.strip()]
        except Exception as exc:
            skip_counter["no_variant_mapping"] += 1
            skipped_records.append(
                {"question_id": question_id, "skip_reason": "no_variant_mapping", "details": f"Mapping error: {exc}"}
            )
            logger.warning("%s skipped | no_variant_mapping | mapping_err=%s", question_id, exc)
            continue

        if not mapping_items:
            skip_counter["no_variant_mapping"] += 1
            skipped_records.append(
                {"question_id": question_id, "skip_reason": "no_variant_mapping", "details": "No valid mapping pairs produced."}
            )
            continue

        # Step 4: apply replacements longest to shortest
        counterfactual_context, total_replacements = _apply_variant_mapping(lay_summary, mapping_items)
        if total_replacements < 1:
            skip_counter["no_replacement"] += 1
            skipped_records.append(
                {"question_id": question_id, "skip_reason": "no_replacement", "details": "Mapping generated but no replacements were applied."}
            )
            continue

        promoted_letter = _find_promoted_letter(options, false_statement_text)
        if promoted_letter is None:
            skip_counter["no_clean_swap"] += 1
            skipped_records.append(
                {"question_id": question_id, "skip_reason": "no_clean_swap", "details": "Entity-swap distractor text not found in options."}
            )
            continue

        replacement_counts.append(total_replacements)
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
                "replacement_count": total_replacements,
                "total_replacements": total_replacements,
                "swap_confidence": swap.confidence,
                "promoted_distractor_text": false_statement_text,
                "promoted_distractor_letter": promoted_letter,
                "original_true_letter": original_true_letter,
                "options": options,
                "atomic_fact": atomic_fact,
                "variants_found": variants_found,
                "variant_mapping": [{"from": m.from_, "to": m.to} for m in mapping_items],
            }
        )
        logger.info(
            "%s passed | %s -> %s | variants=%s | replacements=%s",
            question_id,
            swap.in_true,
            swap.in_false,
            len(variants_found),
            total_replacements,
        )

    save_jsonl(passed_records, out_passed_path)
    save_jsonl(skipped_records, out_skipped_path)

    avg_replacement_count = (
        round(sum(replacement_counts) / len(replacement_counts), 2) if replacement_counts else 0.0
    )
    avg_variant_count = (
        round(sum(variants_count_list) / len(variants_count_list), 2) if variants_count_list else 0.0
    )
    variant_detection_success_rate = _safe_rate(variant_detection_success, variant_detection_attempts)
    estimated_cost = _estimate_cost_usd_gpt41(total_input_tokens, total_output_tokens)

    metadata = {
        "build_time": datetime.now().isoformat(timespec="seconds"),
        "model": config.MODEL_GOLD_SOLVER,
        "temperature": 0.0,
        "total_questions": len(questions),
        "passed_count": len(passed_records),
        "skip_counts": dict(skip_counter),
        "confidence_distribution": {
            "high": int(confidence_counter.get("high", 0)),
            "medium": int(confidence_counter.get("medium", 0)),
            "low": int(confidence_counter.get("low", 0)),
        },
        "average_replacement_count": avg_replacement_count,
        "average_variant_count": avg_variant_count,
        "variant_detection_attempts": variant_detection_attempts,
        "variant_detection_success": variant_detection_success,
        "variant_detection_success_rate": variant_detection_success_rate,
        "extra_variant_forms_beyond_canonical": extra_variant_forms_beyond_canonical,
        "abbreviation_or_shortform_cases": abbreviation_or_shortform_cases,
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
    for key in [
        "no_entity_swap",
        "no_clean_swap",
        "low_confidence",
        "entity_not_found",
        "no_variant_mapping",
        "no_replacement",
    ]:
        print(f"  {key}: {skip_counter.get(key, 0)}")
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
                f"(confidence: {sample['swap_confidence']}, replaced {sample['total_replacements']} times)"
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
        "Counterfactual build v2 finished | total=%s passed=%s skipped=%s",
        len(questions),
        len(passed_records),
        len(skipped_records),
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


if __name__ == "__main__":
    main()
