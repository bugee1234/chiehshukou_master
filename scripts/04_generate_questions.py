from __future__ import annotations

import json
import random
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.utils import OpenAIClient, load_jsonl, save_json, save_jsonl, setup_logger


Strategy = Literal["numerical_perturbation", "entity_swap", "direction_reversal", "detail_fabrication"]
STRATEGIES = set(Strategy.__args__)
PRICING = {
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


class FalseStatement(BaseModel):
    text: str
    strategy: Strategy


class QuestionRaw(BaseModel):
    true_statement: str
    false_statements: list[FalseStatement] = Field(min_length=3, max_length=3)


def _estimate_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING.get(model_name, PRICING["gpt-4o-mini"])
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


def _to_letter(idx: int) -> str:
    return "ABCD"[idx]


def main() -> None:
    logger = setup_logger("question_generation", str(config.LOGS_DIR / "04_generate_questions.log"))
    logger.info("Question generation started.")

    input_path = config.DATA_ATOMIC_FACTS_DIR / "atomic_facts.jsonl"
    prompt_path = config.PROMPTS_DIR / "question_generation.txt"
    output_path = config.DATA_QUESTIONS_DIR / "questions.jsonl"
    metadata_path = config.DATA_QUESTIONS_DIR / "generation_metadata.json"

    af_records = load_jsonl(input_path)
    prompt_template = prompt_path.read_text(encoding="utf-8")
    client = OpenAIClient(logger=logger)

    questions: list[dict[str, Any]] = []
    failed_af_ids: list[str] = []
    failed_error_summaries: dict[str, str] = {}
    strategy_counter: Counter[str] = Counter()
    correct_letter_counter: Counter[str] = Counter()
    non_distinct_strategy_questions = 0

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    for af in tqdm(af_records, desc="Generating questions", total=len(af_records)):
        af_id = str(af.get("af_id", ""))
        article_id = str(af.get("article_id", ""))
        source_dataset = str(af.get("source_dataset", ""))
        atomic_fact = str(af.get("fact", ""))
        lay_summary = str(af.get("lay_summary", ""))

        prompt = prompt_template.replace("{source_paragraph}", lay_summary).replace(
            "{atomic_fact}", atomic_fact
        )

        raw_content = ""
        try:
            response = client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=config.MODEL_GENERATOR,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            usage = response.get("usage", {})
            total_input_tokens += int(usage.get("prompt_tokens", 0))
            total_output_tokens += int(usage.get("completion_tokens", 0))
            total_tokens += int(usage.get("total_tokens", 0))

            raw_content = str(response.get("content", ""))
            parsed_json = json.loads(raw_content)
            parsed = QuestionRaw.model_validate(parsed_json)
        except (json.JSONDecodeError, ValidationError) as exc:
            failed_af_ids.append(af_id)
            failed_error_summaries[af_id] = str(exc)
            logger.error(
                "%s failed | parse/validation error=%s | raw_output_preview=%s",
                af_id,
                exc,
                raw_content[:800],
            )
            continue
        except Exception as exc:
            failed_af_ids.append(af_id)
            failed_error_summaries[af_id] = str(exc)
            logger.error(
                "%s failed | api/runtime error=%s | raw_output_preview=%s",
                af_id,
                exc,
                raw_content[:800],
            )
            continue

        try:
            strategy_list = [item.strategy for item in parsed.false_statements]
            if len(set(strategy_list)) != 3:
                non_distinct_strategy_questions += 1
                logger.warning(
                    "%s warning | false_statements contain repeated strategies=%s",
                    af_id,
                    strategy_list,
                )

            options_with_label = [("TRUE", parsed.true_statement)]
            for fs in parsed.false_statements:
                options_with_label.append((fs.strategy, fs.text))

            shuffler = random.Random(hash(af_id) % (2**32))
            shuffler.shuffle(options_with_label)

            options = [text for _, text in options_with_label]
            correct_index = next(i for i, (label, _) in enumerate(options_with_label) if label == "TRUE")
            correct_letter = _to_letter(correct_index)
        except Exception as exc:
            failed_af_ids.append(af_id)
            failed_error_summaries[af_id] = str(exc)
            logger.error(
                "%s failed | shuffle/post-process error=%s | raw_output_preview=%s",
                af_id,
                exc,
                raw_content[:800],
            )
            continue

        for fs in parsed.false_statements:
            strategy_counter[fs.strategy] += 1
        correct_letter_counter[correct_letter] += 1

        questions.append(
            {
                "question_id": f"{af_id}_q",
                "af_id": af_id,
                "article_id": article_id,
                "source_dataset": source_dataset,
                "atomic_fact": atomic_fact,
                "lay_summary": lay_summary,
                "options": options,
                "correct_index": correct_index,
                "correct_letter": correct_letter,
                "true_statement": parsed.true_statement,
                "false_statements": [item.model_dump() for item in parsed.false_statements],
            }
        )
        logger.info("%s success | correct_letter=%s", af_id, correct_letter)

    save_jsonl(questions, output_path)

    estimated_cost = _estimate_cost_usd(
        config.MODEL_GENERATOR, total_input_tokens, total_output_tokens
    )
    metadata = {
        "generation_time": datetime.now().isoformat(timespec="seconds"),
        "model": config.MODEL_GENERATOR,
        "temperature": 0.0,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost,
        "successful_count": len(questions),
        "failed_count": len(failed_af_ids),
        "failed_af_ids": failed_af_ids,
        "non_distinct_strategy_questions": non_distinct_strategy_questions,
        "strategy_distribution": dict(strategy_counter),
        "correct_letter_distribution": dict(correct_letter_counter),
    }
    save_json(metadata, metadata_path)

    print("\n=== Question Generation Sanity Check ===")
    print(f"Successful questions: {len(questions)}")
    print(f"Failed questions: {len(failed_af_ids)}")
    print(f"Non-distinct strategy questions kept: {non_distinct_strategy_questions}")
    if failed_af_ids:
        print("Top 3 failures:")
        for af_id in failed_af_ids[:3]:
            print(f"- {af_id}: {failed_error_summaries.get(af_id, 'unknown error')}")

    sample_rng = random.Random(42)
    sample_n = min(5, len(questions))
    samples = sample_rng.sample(questions, sample_n) if sample_n > 0 else []
    for i, q in enumerate(samples, start=1):
        print(f"\n--- Sample {i} ---")
        print(f"question_id: {q['question_id']}")
        print(f"source_dataset: {q['source_dataset']}")
        print(f"atomic_fact: {q['atomic_fact']}")
        for idx, opt in enumerate(q["options"]):
            print(f"{_to_letter(idx)}. {opt}")
        print(f"correct_answer: {q['correct_letter']} (index={q['correct_index']})")
        print("distractor_strategies:")
        for item in q["false_statements"]:
            print(f"- {item['strategy']}: {item['text']}")

    print("\nCorrect letter distribution:")
    for letter in "ABCD":
        print(f"{letter}: {correct_letter_counter.get(letter, 0)}")

    print("\nStrategy distribution:")
    for key in sorted(STRATEGIES):
        print(f"{key}: {strategy_counter.get(key, 0)}")

    print("\n=== Token Usage & Cost ===")
    print(f"input_tokens: {total_input_tokens}")
    print(f"output_tokens: {total_output_tokens}")
    print(f"total_tokens: {total_tokens}")
    print(f"estimated_cost_usd: {estimated_cost}")
    print(f"Saved questions JSONL: {output_path}")
    print(f"Saved metadata JSON: {metadata_path}")

    logger.info("Question generation finished.")
    logger.info(
        "Final usage | input=%s output=%s total=%s estimated_cost_usd=%s",
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


if __name__ == "__main__":
    main()
