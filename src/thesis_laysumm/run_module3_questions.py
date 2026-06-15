from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiment_2.run_three_model_exp2 import MODEL_CONFIGS, ProviderClient
from src.thesis_laysumm.llm_utils import (
    mutate_sentence,
    norm_text,
    run_parallel,
    save_usage,
    sha256_text,
    usage_tracker,
)
from src.thesis_laysumm.paths import run_data_dir
from src.utils import load_jsonl, save_json, save_jsonl

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_MODEL_KEY = "gemini3_flash_preview_minimal"


def _module2_dir(run_name: str, model_key: str) -> Path:
    return run_data_dir(run_name) / "02_module2" / model_key


def _questions_dir(run_name: str, model_key: str) -> Path:
    path = run_data_dir(run_name) / "03_questions" / model_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_question(
    *,
    af: dict[str, Any],
    model_key: str,
    false_texts: list[str],
    false_strategies: list[str],
    parse_error: bool,
) -> dict[str, Any]:
    true_statement = str(af["fact"]).strip()
    false_texts = [str(x).strip() for x in false_texts if str(x).strip()]
    while len(false_texts) < 3:
        candidate = mutate_sentence(true_statement if len(false_texts) % 2 == 0 else true_statement)
        if norm_text(candidate) != norm_text(true_statement):
            false_texts.append(candidate)
        else:
            false_texts.append(f"{true_statement} (unsupported variant {len(false_texts) + 1})")

    options_with_label = [("TRUE", true_statement)] + [("FALSE", x) for x in false_texts[:3]]
    seed = int(hashlib.md5(str(af["af_id"]).encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    rng.shuffle(options_with_label)
    correct_idx = next(i for i, (label, _) in enumerate(options_with_label) if label == "TRUE")
    options_abcd = [text for _, text in options_with_label]
    options = {letter: options_abcd[i] for i, letter in enumerate("ABCD")}
    options["E"] = "None of the above"

    false_statements = []
    for i, text in enumerate(false_texts[:3]):
        strategy = false_strategies[i] if i < len(false_strategies) else "fallback_mutate"
        false_statements.append({"text": text, "strategy": strategy})

    return {
        "question_id": f"{af['af_id']}_q",
        "af_id": af["af_id"],
        "model_key": model_key,
        "article_id": af["article_id"],
        "source_dataset": af["source_dataset"],
        "original_index": af["original_index"],
        "fact": true_statement,
        "abstract_sentence_idx": af.get("abstract_sentence_idx"),
        "abstract_sentence": af.get("abstract_sentence"),
        "options": options,
        "correct_letter": "ABCD"[correct_idx],
        "true_statement": true_statement,
        "true_option_source": "af_verbatim",
        "false_statements": false_statements,
        "nota_included": True,
        "question_generator_model_key": model_key,
        "question_generator_provider": MODEL_CONFIGS[model_key]["provider"],
        "question_generator_model": MODEL_CONFIGS[model_key]["model"],
        "parse_error": parse_error,
    }


def generate_questions(
    *,
    run_name: str,
    model_key: str,
    max_workers: int,
    resume: bool,
    limit_articles: int | None = None,
) -> list[dict[str, Any]]:
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_key {model_key!r}")

    final_af_path = _module2_dir(run_name, model_key) / "final_keep_af.jsonl"
    if not final_af_path.exists():
        raise FileNotFoundError(f"Missing Module 2 output: {final_af_path}")

    final_af = load_jsonl(final_af_path)
    if limit_articles is not None:
        from src.thesis_laysumm.llm_utils import load_articles

        allowed_ids = {str(r["id"]) for r in load_articles(run_name)[:limit_articles]}
        final_af = [r for r in final_af if str(r["article_id"]) in allowed_ids]

    out_dir = _questions_dir(run_name, model_key)
    out_path = out_dir / "questions_1t3f_nota.jsonl"
    prompt = (PROMPTS_DIR / "generate_1t3f_distractors.txt").read_text(encoding="utf-8")

    usage = usage_tracker(run_name, resume=resume)
    client = ProviderClient(model_key, usage)
    existing = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("af_id")) for r in existing}

    def worker(af: dict[str, Any]) -> dict[str, Any]:
        raw = ""
        parse_error = False
        false_texts: list[str] = []
        false_strategies: list[str] = []
        try:
            raw = client.chat(
                stage=f"module3.questions:{model_key}",
                item_id=str(af["af_id"]),
                messages=[
                    {
                        "role": "user",
                        "content": prompt.replace("{atomic_fact}", str(af["fact"])).replace(
                            "{source_span}", str(af.get("source_span", ""))
                        ),
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = json.loads(raw)
            false_items = obj.get("false_statements", [])
            if not isinstance(false_items, list):
                false_items = []
            for item in false_items:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                if not text or norm_text(text) == norm_text(af["fact"]):
                    continue
                false_texts.append(text)
                false_strategies.append(str(item.get("strategy", "")).strip() or "llm")
        except Exception:
            parse_error = True

        return _build_question(
            af=af,
            model_key=model_key,
            false_texts=false_texts,
            false_strategies=false_strategies,
            parse_error=parse_error,
        )

    todo = [r for r in final_af if str(r["af_id"]) not in done]
    new_rows = run_parallel(
        todo,
        worker,
        max_workers=max_workers,
        desc=f"module3 | {model_key}",
    )
    all_rows = existing + new_rows
    all_rows.sort(key=lambda r: (str(r["article_id"]), str(r["af_id"])))
    save_jsonl(all_rows, out_path)
    save_usage(run_name, usage)

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "stage": "03_questions",
            "model_key": model_key,
            "provider": MODEL_CONFIGS[model_key]["provider"],
            "model": MODEL_CONFIGS[model_key]["model"],
            "max_workers": max_workers,
            "articles_limit": limit_articles,
            "prompt_file": str(PROMPTS_DIR / "generate_1t3f_distractors.txt"),
            "prompt_sha256": sha256_text(prompt),
            "true_option_policy": "af_verbatim",
            "final_keep_af_total": len(final_af),
            "questions_total": len(all_rows),
            "questions_by_source": dict(Counter(str(r["source_dataset"]) for r in all_rows)),
            "questions_by_article": dict(Counter(str(r["article_id"]) for r in all_rows)),
            "parse_error_count": sum(1 for r in all_rows if r.get("parse_error")),
            "usage_summary": usage.summarize(),
        },
        out_dir / "questions_metadata.json",
    )

    print(f"[module3] {model_key} questions={len(all_rows)} -> {out_path}")
    return all_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Thesis lay-summary pipeline — Module 3: 1T3F + E questions for final keep AF."
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--model-key", type=str, default=DEFAULT_MODEL_KEY)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit-articles", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_questions(
        run_name=args.run_name,
        model_key=args.model_key,
        max_workers=args.max_workers,
        resume=not args.no_resume,
        limit_articles=args.limit_articles,
    )


if __name__ == "__main__":
    main()
