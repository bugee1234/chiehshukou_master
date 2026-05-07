from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

from pydantic import BaseModel, ValidationError
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.utils import OpenAIClient, load_jsonl, save_json, save_jsonl, setup_logger


class AtomicFactItem(BaseModel):
    fact: str
    source_sentence: str


class AtomicFactsResponse(BaseModel):
    atomic_facts: list[AtomicFactItem]


def _estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * 0.15
    output_cost = (output_tokens / 1_000_000) * 0.60
    return round(input_cost + output_cost, 6)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _pick_sample_afs(afs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not afs:
        return []

    rng = random.Random(42)
    if len(afs) <= 5:
        return rng.sample(afs, len(afs))

    plos = [x for x in afs if x.get("source_dataset") == "PLOS"]
    elife = [x for x in afs if x.get("source_dataset") == "eLife"]

    if len(plos) >= 2 and len(elife) >= 2:
        selected = []
        selected.extend(rng.sample(plos, 2))
        selected.extend(rng.sample(elife, 2))
        selected_ids = {x["af_id"] for x in selected}
        remaining = [x for x in afs if x["af_id"] not in selected_ids]
        selected.extend(rng.sample(remaining, 1))
        rng.shuffle(selected)
        return selected

    return rng.sample(afs, 5)


def main() -> None:
    logger = setup_logger("af_extraction", str(config.LOGS_DIR / "03_af_extraction.log"))
    logger.info("AF extraction started.")

    input_path = config.DATA_RAW_DIR / "sampled_articles.jsonl"
    prompt_path = config.ROOT_DIR / "prompts" / "af_extraction.txt"
    out_afs_path = config.DATA_ATOMIC_FACTS_DIR / "atomic_facts.jsonl"
    out_meta_path = config.DATA_ATOMIC_FACTS_DIR / "extraction_metadata.json"

    records = load_jsonl(input_path)
    prompt_template = prompt_path.read_text(encoding="utf-8")
    client = OpenAIClient(logger=logger)

    output_afs: list[dict[str, Any]] = []
    failed_article_ids: list[str] = []
    afs_per_article: dict[str, int] = {}
    afs_per_source: dict[str, int] = defaultdict(int)

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    source_article_ids: dict[str, list[str]] = defaultdict(list)
    for item in records:
        source_article_ids[_safe_text(item.get("source_dataset"))].append(_safe_text(item.get("id")))

    for item in tqdm(records, desc="Extracting atomic facts", total=len(records)):
        article_id = _safe_text(item.get("id"))
        source_dataset = _safe_text(item.get("source_dataset"))
        lay_summary = _safe_text(item.get("lay_summary"))
        prompt = prompt_template.replace("{source_text}", lay_summary)
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

            raw_content = _safe_text(response.get("content", ""))
            parsed = json.loads(raw_content)
            validated = AtomicFactsResponse.model_validate(parsed)

            af_count = 0
            for idx, af_item in enumerate(validated.atomic_facts, start=1):
                fact_text = _safe_text(af_item.fact).strip()
                source_sentence_text = _safe_text(af_item.source_sentence).strip()
                if not fact_text or not source_sentence_text:
                    continue
                af_count += 1
                output_afs.append(
                    {
                        "af_id": f"{article_id}_af_{af_count:03d}",
                        "article_id": article_id,
                        "source_dataset": source_dataset,
                        "fact": fact_text,
                        "source_sentence": source_sentence_text,
                        "lay_summary": lay_summary,
                    }
                )

            afs_per_article[article_id] = af_count
            afs_per_source[source_dataset] += af_count
            logger.info("%s success | extracted_afs=%s", article_id, af_count)
        except (json.JSONDecodeError, ValidationError) as exc:
            failed_article_ids.append(article_id)
            afs_per_article[article_id] = 0
            logger.error(
                "%s failed | parse/validation error=%s | raw_output_preview=%s",
                article_id,
                exc,
                raw_content[:500],
            )
        except Exception as exc:
            failed_article_ids.append(article_id)
            afs_per_article[article_id] = 0
            logger.error(
                "%s failed | api/runtime error=%s | raw_output_preview=%s",
                article_id,
                exc,
                raw_content[:500],
            )

    save_jsonl(output_afs, out_afs_path)

    estimated_cost_usd = _estimate_cost_usd(total_input_tokens, total_output_tokens)
    metadata = {
        "extraction_time": datetime.now().isoformat(timespec="seconds"),
        "model": config.MODEL_GENERATOR,
        "temperature": 0.0,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "articles_processed": len(records) - len(failed_article_ids),
        "articles_failed": failed_article_ids,
        "total_afs_extracted": len(output_afs),
        "afs_per_article": afs_per_article,
        "afs_per_source": dict(afs_per_source),
    }
    save_json(metadata, out_meta_path)

    plos_counts = [afs_per_article.get(aid, 0) for aid in source_article_ids.get("PLOS", [])]
    elife_counts = [afs_per_article.get(aid, 0) for aid in source_article_ids.get("eLife", [])]
    plos_avg = round(statistics.mean(plos_counts), 2) if plos_counts else 0.0
    elife_avg = round(statistics.mean(elife_counts), 2) if elife_counts else 0.0

    print("\n=== AF Extraction Sanity Check ===")
    print(f"Total AFs extracted: {len(output_afs)}")
    print(f"Average AFs per PLOS article: {plos_avg}")
    print(f"Average AFs per eLife article: {elife_avg}")
    print(f"Failed articles: {len(failed_article_ids)}")

    sample_afs = _pick_sample_afs(output_afs)
    for i, af in enumerate(sample_afs, start=1):
        print(f"\n--- Sample {i} ---")
        print(f"af_id: {af['af_id']}")
        print(f"source_dataset: {af['source_dataset']}")
        print(f"fact: {af['fact']}")
        print(f"source_sentence: {af['source_sentence']}")

    print("\n=== Token Usage ===")
    print(f"input_tokens: {total_input_tokens}")
    print(f"output_tokens: {total_output_tokens}")
    print(f"total_tokens: {total_tokens}")
    print(f"estimated_cost_usd: {estimated_cost_usd}")
    print(f"Saved AFs JSONL: {out_afs_path}")
    print(f"Saved metadata JSON: {out_meta_path}")

    logger.info("AF extraction finished.")
    logger.info(
        "Token usage | input=%s output=%s total=%s estimated_cost_usd=%s",
        total_input_tokens,
        total_output_tokens,
        total_tokens,
        estimated_cost_usd,
    )


if __name__ == "__main__":
    main()
