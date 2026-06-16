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
from src.thesis_laysumm.llm_utils import mutate_sentence, norm_text, run_parallel, save_usage, sha256_text, usage_tracker
from src.thesis_laysumm.paths import run_data_dir
from src.thesis_laysumm_v3.common import require_mode
from src.utils import load_jsonl, save_json, save_jsonl

PROMPTS_DIR_V2 = ROOT_DIR / "src" / "thesis_laysumm" / "prompts"
DEFAULT_MODEL_KEY = "gpt41_mini"


def _evidence_dir(run_name: str, model_key: str, mode: str) -> Path:
    return run_data_dir(run_name) / "02_5_evidence_table" / mode / model_key


def _questions_dir(run_name: str, model_key: str, mode: str) -> Path:
    path = run_data_dir(run_name) / "03_questions_v3" / mode / model_key
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
        candidate = mutate_sentence(true_statement)
        if norm_text(candidate) != norm_text(true_statement):
            false_texts.append(candidate)
        else:
            false_texts.append(f"{true_statement} (unsupported variant {len(false_texts) + 1})")

    options_with_label = [("TRUE", true_statement)] + [("FALSE", x) for x in false_texts[:3]]
    seed = int(hashlib.md5(str(af["af_id"]).encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    rng.shuffle(options_with_label)
    correct_idx = next(i for i, (label, _) in enumerate(options_with_label) if label == "TRUE")
    options = {letter: options_with_label[i][1] for i, letter in enumerate("ABCD")}
    options["E"] = "None of the above"

    return {
        "question_id": f"{af['af_id']}_q",
        "af_id": af["af_id"],
        "article_id": af["article_id"],
        "source_dataset": af.get("source_dataset"),
        "original_index": af.get("original_index"),
        "model_key": model_key,
        "mode": af.get("mode"),
        "evidence_row_id": af.get("evidence_row_id"),
        "check_type": af.get("check_type"),
        "slot_type": af.get("slot_type"),
        "fact": true_statement,
        "source_span": af.get("source_span", ""),
        "abstract_sentence_idx": af.get("abstract_sentence_idx"),
        "abstract_sentence": af.get("abstract_sentence"),
        "options": options,
        "correct_letter": "ABCD"[correct_idx],
        "true_statement": true_statement,
        "false_statements": [
            {
                "text": false_texts[i],
                "strategy": false_strategies[i] if i < len(false_strategies) else "fallback_mutate",
            }
            for i in range(3)
        ],
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
    mode: str,
    max_workers: int,
    resume: bool,
    limit_articles: int | None = None,
) -> list[dict[str, Any]]:
    require_mode(mode)
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_key {model_key!r}")

    check_path = _evidence_dir(run_name, model_key, mode) / "check_af.jsonl"
    if not check_path.exists():
        raise FileNotFoundError(f"Missing v3 check AF: {check_path}")
    check_af = load_jsonl(check_path)
    if limit_articles is not None:
        allowed = sorted({str(r["article_id"]) for r in check_af})[:limit_articles]
        check_af = [r for r in check_af if str(r["article_id"]) in set(allowed)]

    out_dir = _questions_dir(run_name, model_key, mode)
    out_path = out_dir / "questions_1t3f_nota.jsonl"
    prompt_path = PROMPTS_DIR_V2 / "generate_1t3f_distractors.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")

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
                stage=f"v3.questions.{mode}:{model_key}",
                item_id=str(af["af_id"]),
                messages=[
                    {
                        "role": "user",
                        "content": prompt_template.replace("{atomic_fact}", str(af["fact"])).replace(
                            "{source_span}", str(af.get("source_span", ""))
                        ),
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = json.loads(raw)
            raw_false = obj.get("false_statements", [])
            if not isinstance(raw_false, list):
                raw_false = []
            for item in raw_false:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                if text and norm_text(text) != norm_text(af["fact"]):
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

    todo = [r for r in check_af if str(r.get("af_id")) not in done]
    new_rows = run_parallel(todo, worker, max_workers=max_workers, desc=f"v3 questions | {mode} | {model_key}")
    all_rows = existing + new_rows
    all_rows.sort(key=lambda r: (str(r["article_id"]), str(r["af_id"])))
    save_jsonl(all_rows, out_path)
    save_usage(run_name, usage)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "stage": "v3_03_questions",
            "mode": mode,
            "model_key": model_key,
            "prompt_file": str(prompt_path),
            "prompt_sha256": sha256_text(prompt_template),
            "check_af_total": len(check_af),
            "questions_total": len(all_rows),
            "check_type_counts": dict(Counter(str(r.get("check_type")) for r in all_rows)),
            "slot_type_counts": dict(Counter(str(r.get("slot_type")) for r in all_rows)),
            "parse_error_count": sum(1 for r in all_rows if r.get("parse_error")),
            "usage_summary": usage.summarize(),
        },
        out_dir / "questions_metadata.json",
    )
    print(f"[v3 questions] {mode} {model_key} questions={len(all_rows)} -> {out_path}")
    return all_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V3: generate 1T3F questions for core and high-risk checks.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--mode", choices=["balanced", "factuality_chase"], required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit-articles", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_questions(
        run_name=args.run_name,
        model_key=args.model_key,
        mode=args.mode,
        max_workers=args.max_workers,
        resume=not args.no_resume,
        limit_articles=args.limit_articles,
    )


if __name__ == "__main__":
    main()

