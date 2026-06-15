from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiment_2.run_three_model_exp2 import MODEL_CONFIGS, ProviderClient
from src.thesis_laysumm.biolaysumm_data import word_count
from src.thesis_laysumm.llm_utils import (
    load_articles,
    run_parallel,
    save_usage,
    sha256_text,
    usage_tracker,
)
from src.thesis_laysumm.paths import run_data_dir
from src.utils import load_json, load_jsonl, save_json, save_jsonl

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_MODEL_KEY = "gemini3_flash_preview_minimal"
PROMPT_VERSION = "generate_lay_summary_v1"
REWRITE_PROMPT_VERSION = "rewrite_lay_summary_v1"
VALID_SECTIONS = {"background", "methods", "results", "implications"}


def _module2_dir(run_name: str, model_key: str) -> Path:
    return run_data_dir(run_name) / "02_module2" / model_key


def _summaries_dir(run_name: str, model_key: str) -> Path:
    path = run_data_dir(run_name) / "04_summaries" / model_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_few_shot_examples() -> list[dict[str, Any]]:
    path = PROMPTS_DIR / "few_shot_summary_examples.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing few-shot examples: {path}. "
            "Run: python -m src.thesis_laysumm.prepare_few_shot_summary_examples --run-name <run_name>"
        )
    examples = load_json(path)
    if not isinstance(examples, list) or len(examples) < 1:
        raise ValueError(f"few_shot_summary_examples.json must be a non-empty list: {path}")
    return examples


def _format_few_shot_examples(examples: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for i, ex in enumerate(examples, start=1):
        blocks.append(
            f"### Example {i} ({ex.get('source_dataset', 'unknown')})\n"
            f"Title: {ex.get('title', '')}\n\n"
            f"Abstract:\n{ex.get('abstract', '')}\n\n"
            f"Expert lay summary (STYLE REFERENCE ONLY — do not copy structure or wording):\n"
            f"{ex.get('expert_summary', '')}"
        )
    return "\n\n".join(blocks)


def _format_abstract_sentences(sentences: list[str]) -> str:
    lines = []
    for i, sentence in enumerate(sentences, start=1):
        text = str(sentence or "").strip()
        if text:
            lines.append(f"[S{i}] {text}")
    return "\n".join(lines) if lines else ""


def _group_final_keep_af(afs: list[dict[str, Any]], abstract_sentences: list[str]) -> str:
    by_idx: dict[int | str, list[dict[str, Any]]] = defaultdict(list)
    for af in afs:
        idx = af.get("abstract_sentence_idx")
        if idx in {None, "", "null", "None"}:
            by_idx["unmapped"].append(af)
        else:
            by_idx[int(idx)].append(af)

    blocks: list[str] = []
    for i in range(1, len(abstract_sentences) + 1):
        sentence = str(abstract_sentences[i - 1]).strip()
        group = by_idx.get(i, [])
        if not sentence and not group:
            continue
        lines = [f"[S{i}] {sentence}" if sentence else f"[S{i}]"]
        for af in group:
            lines.append(f"  - AF {af['af_id']}: {af['fact']}")
        blocks.append("\n".join(lines))

    unmapped = by_idx.get("unmapped", [])
    if unmapped:
        lines = ["[S?] AFs without abstract sentence mapping:"]
        for af in unmapped:
            lines.append(f"  - AF {af['af_id']}: {af['fact']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "(no final keep AFs)"


def _validate_generation(
    obj: dict[str, Any],
    *,
    valid_af_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[dict[str, str]]]:
    errors: list[str] = []
    warnings: list[dict[str, str]] = []
    raw_plan = obj.get("abstract_claim_plan", [])
    raw_sentences = obj.get("sentences", [])
    if not isinstance(raw_plan, list):
        errors.append("abstract_claim_plan must be a list")
        raw_plan = []
    if not isinstance(raw_sentences, list) or not raw_sentences:
        errors.append("sentences must be a non-empty list")

    plan_rows: list[dict[str, Any]] = []
    for entry in raw_plan:
        if not isinstance(entry, dict):
            errors.append("each abstract_claim_plan entry must be an object")
            continue
        af_ids = _normalize_af_ids(entry.get("af_ids", entry.get("af_id")), valid_af_ids, errors, label="abstract_claim_plan")
        idx = entry.get("abstract_sentence_idx")
        if idx in {"", "null", "None"}:
            idx = None
        plan_rows.append(
            {
                "abstract_sentence_idx": idx,
                "core_claim": str(entry.get("core_claim", "")).strip(),
                "af_ids": af_ids,
                "include_in_summary": bool(entry.get("include_in_summary", True)),
            }
        )

    sentence_rows: list[dict[str, Any]] = []
    for entry in raw_sentences if isinstance(raw_sentences, list) else []:
        if not isinstance(entry, dict):
            errors.append("each sentence must be an object")
            continue
        text = str(entry.get("text", "")).strip()
        af_ids = _normalize_af_ids(entry.get("af_ids", entry.get("af_id")), valid_af_ids, errors, label="sentence")
        section = str(entry.get("section", "")).strip().lower()
        if not text:
            errors.append("sentence text is empty")
            continue
        if not af_ids:
            warnings.append({"text": text, "reason": "missing_af_ids"})
            continue
        if section not in VALID_SECTIONS:
            errors.append(f"invalid section: {section!r}")
            continue
        sentence_rows.append({"text": text, "af_ids": af_ids, "section": section})

    if isinstance(raw_sentences, list) and raw_sentences and not sentence_rows:
        errors.append("no sentences with valid af_ids remained after validation")

    return plan_rows, sentence_rows, errors, warnings


def _normalize_af_ids(
    raw_af_ids: Any,
    valid_af_ids: set[str],
    errors: list[str],
    *,
    label: str,
) -> list[str]:
    if isinstance(raw_af_ids, str):
        values = [raw_af_ids.strip()] if raw_af_ids.strip() else []
    elif isinstance(raw_af_ids, list):
        values = [str(x).strip() for x in raw_af_ids if str(x).strip()]
    else:
        values = []
    out: list[str] = []
    for af_id in values:
        if af_id not in valid_af_ids:
            errors.append(f"{label} cites unknown af_id: {af_id}")
        else:
            out.append(af_id)
    return list(dict.fromkeys(out))


def generate_summaries(
    *,
    run_name: str,
    model_key: str,
    max_workers: int,
    resume: bool,
    limit_articles: int | None = None,
) -> list[dict[str, Any]]:
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_key {model_key!r}")

    articles = load_articles(run_name)
    if limit_articles is not None:
        articles = articles[:limit_articles]

    final_af_path = _module2_dir(run_name, model_key) / "final_keep_af.jsonl"
    if not final_af_path.exists():
        raise FileNotFoundError(f"Missing Module 2 output: {final_af_path}")

    final_af = load_jsonl(final_af_path)
    af_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_af:
        af_by_article[str(row["article_id"])].append(row)

    prompt_path = PROMPTS_DIR / "generate_lay_summary.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")
    few_shot_block = _format_few_shot_examples(_load_few_shot_examples())

    out_dir = _summaries_dir(run_name, model_key)
    out_path = out_dir / "generated_summaries.jsonl"

    usage = usage_tracker(run_name, resume=resume)
    client = ProviderClient(model_key, usage)
    existing = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("article_id")) for r in existing}

    def worker(article: dict[str, Any]) -> dict[str, Any]:
        article_id = str(article["id"])
        afs = af_by_article.get(article_id, [])
        valid_af_ids = {str(af["af_id"]) for af in afs}
        expert_wc = int(article.get("expert_summary_word_count") or word_count(article.get("expert_summary", "")))
        min_word_count = expert_wc
        abstract_sentences = article.get("abstract_sentences") or []
        grouped_afs = _group_final_keep_af(afs, abstract_sentences)

        raw = ""
        parse_error = False
        validation_errors: list[str] = []
        skipped_sentences: list[dict[str, str]] = []
        abstract_claim_plan: list[dict[str, Any]] = []
        sentences: list[dict[str, Any]] = []
        generated_summary = ""

        try:
            prompt = (
                prompt_template.replace("{few_shot_examples}", few_shot_block)
                .replace("{title}", str(article.get("title", "")))
                .replace("{abstract}", str(article.get("abstract", "")))
                .replace("{grouped_afs}", grouped_afs)
                .replace("{expert_summary_word_count}", str(expert_wc))
            )
            raw = client.chat(
                stage=f"phase_b.generate_summary:{model_key}",
                item_id=article_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = json.loads(raw)
            abstract_claim_plan, sentences, validation_errors, skipped_sentences = _validate_generation(
                obj,
                valid_af_ids=valid_af_ids,
            )
            if validation_errors:
                raise ValueError("; ".join(validation_errors))
            generated_summary = " ".join(row["text"] for row in sentences if row["text"])
            if not generated_summary.strip():
                raise ValueError("generated_summary is empty after joining sentences")
        except Exception as exc:
            parse_error = True
            skipped_sentences = []
            if not validation_errors:
                validation_errors = [str(exc)]

        af_ids_used = sorted({af_id for row in sentences for af_id in row.get("af_ids", [])})
        generated_wc = word_count(generated_summary)
        below_min_length = bool(generated_summary) and generated_wc < min_word_count

        return {
            "article_id": article_id,
            "source_dataset": article.get("source_dataset"),
            "original_index": article.get("original_index"),
            "model_key": model_key,
            "generator_provider": MODEL_CONFIGS[model_key]["provider"],
            "generator_model": MODEL_CONFIGS[model_key]["model"],
            "prompt_version": PROMPT_VERSION,
            "prompt_file": str(prompt_path),
            "prompt_sha256": sha256_text(prompt_template),
            "expert_summary_word_count": expert_wc,
            "min_word_count": min_word_count,
            "target_word_count": expert_wc,
            "abstract_claim_plan": abstract_claim_plan,
            "sentences": sentences,
            "generated_summary": generated_summary,
            "af_ids_used": af_ids_used,
            "af_count_input": len(afs),
            "af_count_cited": len(af_ids_used),
            "generated_word_count": generated_wc,
            "below_min_length": below_min_length,
            "parse_error": parse_error,
            "validation_errors": validation_errors,
            "skipped_sentences": skipped_sentences,
            "raw_response": raw if parse_error else None,
        }

    todo = [art for art in articles if str(art["id"]) not in done]
    new_rows = run_parallel(
        todo,
        worker,
        max_workers=max_workers,
        desc=f"phase_b.generate_summary | {model_key}",
    )
    all_rows = existing + new_rows
    all_rows.sort(key=lambda r: str(r["article_id"]))
    save_jsonl(all_rows, out_path)
    save_usage(run_name, usage)

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "stage": "04_summaries",
            "model_key": model_key,
            "provider": MODEL_CONFIGS[model_key]["provider"],
            "model": MODEL_CONFIGS[model_key]["model"],
            "max_workers": max_workers,
            "articles_limit": limit_articles,
            "prompt_version": PROMPT_VERSION,
            "prompt_file": str(prompt_path),
            "prompt_sha256": sha256_text(prompt_template),
            "few_shot_file": str(PROMPTS_DIR / "few_shot_summary_examples.json"),
            "length_policy": {
                "min_word_count": "expert_summary_word_count",
                "max_word_count": "none (longer allowed)",
            },
            "summaries_total": len(all_rows),
            "parse_error_count": sum(1 for r in all_rows if r.get("parse_error")),
            "below_min_length_count": sum(1 for r in all_rows if r.get("below_min_length")),
            "skipped_sentence_count": sum(len(r.get("skipped_sentences") or []) for r in all_rows),
            "generated_word_count_stats": _word_stats([int(r.get("generated_word_count") or 0) for r in all_rows]),
            "af_cited_stats": _word_stats([int(r.get("af_count_cited") or 0) for r in all_rows]),
            "summaries_by_source": dict(Counter(str(r.get("source_dataset")) for r in all_rows)),
            "usage_summary": usage.summarize(),
        },
        out_dir / "generated_summaries_metadata.json",
    )

    print(
        f"[phase_b] generate_summary {model_key} summaries={len(all_rows)} "
        f"parse_errors={sum(1 for r in all_rows if r.get('parse_error'))} "
        f"below_min_length={sum(1 for r in all_rows if r.get('below_min_length'))}"
    )
    print(f"[phase_b] output -> {out_path}")
    return all_rows


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _questions_dir(run_name: str, model_key: str) -> Path:
    return run_data_dir(run_name) / "03_questions" / model_key


def _answers_dir(run_name: str, model_key: str) -> Path:
    path = run_data_dir(run_name) / "05_module3_answers" / model_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_options(question: dict[str, Any]) -> dict[str, str]:
    opts = question.get("options") or {}
    return {letter: str(opts.get(letter, "")).strip() for letter in "ABCDE"}


def _parse_option_eval(obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = obj.get("option_evaluation", {})
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return out
    for letter in "ABCD":
        entry = raw.get(letter)
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status", "")).strip().lower()
        if status not in {"supported", "contradicted", "insufficient"}:
            status = ""
        out[letter] = {
            "status": status,
            "negation_flag": _coerce_bool(entry.get("negation_flag", False)),
            "evidence": str(entry.get("evidence", "")).strip(),
            "strictness_mismatch": _coerce_bool(entry.get("strictness_mismatch", False)),
        }
    return out


def _decide_setup_c_answer(option_eval: dict[str, dict[str, Any]], obj: dict[str, Any]) -> str:
    direct = str(obj.get("final_answer", "")).strip().upper()
    if direct in {"A", "B", "C", "D", "E"}:
        return direct
    supported = []
    for letter in "ABCD":
        entry = option_eval.get(letter, {})
        if (
            str(entry.get("status", "")).lower() == "supported"
            and not _coerce_bool(entry.get("negation_flag", False))
            and not _coerce_bool(entry.get("strictness_mismatch", False))
        ):
            supported.append(letter)
    if not supported:
        return "E"
    if len(supported) == 1:
        return supported[0]
    return max(supported, key=lambda x: len(str(option_eval.get(x, {}).get("evidence", ""))))


def _loads_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty response")
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = text.find("{")
        if start < 0:
            raise
        obj, _end = decoder.raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("response JSON must be an object")
    return obj


def _parse_answer_from_raw(raw: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str, str]:
    obj = _loads_json_object(raw)
    option_eval = _parse_option_eval(obj)
    pred_letter = _decide_setup_c_answer(option_eval, obj)
    reasoning = str(obj.get("reasoning", "")).strip()
    pred_letter = pred_letter if pred_letter in {"A", "B", "C", "D", "E"} else "E"
    return obj, option_eval, pred_letter, reasoning


def _answer_row_from_parsed(
    *,
    question: dict[str, Any],
    summary_row: dict[str, Any],
    model_key: str,
    opts: dict[str, str],
    correct_letter: str,
    option_eval: dict[str, dict[str, Any]],
    pred_letter: str,
    reasoning: str,
    system_prompt_path: Path,
    user_prompt_path: Path,
    system_prompt: str,
    user_template: str,
    parse_error: bool,
    raw_response: str | None,
    json_repaired: bool = False,
) -> dict[str, Any]:
    correct = (not parse_error) and pred_letter == correct_letter
    return {
        "question_id": question["question_id"],
        "af_id": question["af_id"],
        "article_id": str(question["article_id"]),
        "source_dataset": question.get("source_dataset"),
        "original_index": question.get("original_index"),
        "model_key": model_key,
        "answer_model": MODEL_CONFIGS[model_key]["model"],
        "answer_provider": MODEL_CONFIGS[model_key]["provider"],
        "fact": question.get("fact", ""),
        "abstract_sentence_idx": question.get("abstract_sentence_idx"),
        "abstract_sentence": question.get("abstract_sentence"),
        "generated_summary_word_count": int(summary_row.get("generated_word_count") or 0),
        "options": opts,
        "correct_letter": correct_letter,
        "predicted_letter": pred_letter,
        "correct": correct,
        "correct_answer": opts.get(correct_letter, ""),
        "predicted_answer": opts.get(pred_letter, ""),
        "option_evaluation": option_eval,
        "reasoning": reasoning,
        "parse_error": parse_error,
        "json_repaired": json_repaired,
        "raw_response": raw_response,
        "context_mode": "full_generated_summary",
        "prompt_file_system": str(system_prompt_path),
        "prompt_file_user": str(user_prompt_path),
        "prompt_sha256_system": sha256_text(system_prompt),
        "prompt_sha256_user": sha256_text(user_template),
    }


def _try_repair_answer_row(row: dict[str, Any]) -> dict[str, Any] | None:
    raw = row.get("raw_response")
    if not raw or not row.get("parse_error"):
        return None
    try:
        _obj, option_eval, pred_letter, reasoning = _parse_answer_from_raw(str(raw))
    except Exception:
        return None
    correct_letter = str(row["correct_letter"]).strip().upper()
    opts = row.get("options") or {}
    repaired = dict(row)
    repaired.update(
        {
            "predicted_letter": pred_letter,
            "predicted_answer": opts.get(pred_letter, ""),
            "option_evaluation": option_eval,
            "reasoning": reasoning,
            "parse_error": False,
            "json_repaired": True,
            "correct": pred_letter == correct_letter,
            "raw_response": None,
        }
    )
    return repaired


def _repair_answer_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    repaired = 0
    out: list[dict[str, Any]] = []
    for row in rows:
        fixed = _try_repair_answer_row(row)
        if fixed is not None:
            out.append(fixed)
            repaired += 1
        else:
            out.append(row)
    return out, repaired


def _save_answer_outputs(
    *,
    run_name: str,
    model_key: str,
    out_dir: Path,
    all_rows: list[dict[str, Any]],
    questions_expected: int,
    max_workers: int,
    limit_articles: int | None,
    system_prompt_path: Path,
    user_prompt_path: Path,
    system_prompt: str,
    user_template: str,
    usage_summary: dict[str, Any],
    json_repaired_count: int = 0,
) -> None:
    all_rows.sort(key=lambda r: (str(r["article_id"]), str(r["af_id"])))
    wrong_rows = [r for r in all_rows if r.get("parse_error") or not r.get("correct")]
    save_jsonl(all_rows, out_dir / "module3_answers.jsonl")
    save_jsonl(wrong_rows, out_dir / "wrong_answers.jsonl")
    scored = [r for r in all_rows if not r.get("parse_error")]
    accuracy = (sum(1 for r in scored if r.get("correct")) / len(scored)) if scored else 0.0
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "stage": "05_module3_answers",
            "model_key": model_key,
            "provider": MODEL_CONFIGS[model_key]["provider"],
            "model": MODEL_CONFIGS[model_key]["model"],
            "max_workers": max_workers,
            "articles_limit": limit_articles,
            "context_mode": "full_generated_summary",
            "prompt_file_system": str(system_prompt_path),
            "prompt_file_user": str(user_prompt_path),
            "prompt_sha256_system": sha256_text(system_prompt),
            "prompt_sha256_user": sha256_text(user_template),
            "questions_total": len(all_rows),
            "questions_expected": questions_expected,
            "parse_error_count": sum(1 for r in all_rows if r.get("parse_error")),
            "json_repaired_count": json_repaired_count,
            "correct_count": sum(1 for r in all_rows if r.get("correct")),
            "wrong_count": len(wrong_rows),
            "accuracy_excluding_parse_errors": round(accuracy, 4),
            "accuracy_all_rows": round(sum(1 for r in all_rows if r.get("correct")) / len(all_rows), 4)
            if all_rows
            else 0.0,
            "predicted_letter_counts": dict(Counter(str(r.get("predicted_letter")) for r in all_rows)),
            "wrong_by_article": dict(Counter(str(r["article_id"]) for r in wrong_rows)),
            "usage_summary": usage_summary,
        },
        out_dir / "module3_answers_metadata.json",
    )


def _build_answer_user_message(*, generated_summary: str, question: dict[str, Any], template: str) -> str:
    opts = _normalize_options(question)
    return (
        template.replace("{generated_summary}", generated_summary)
        .replace("{option_a}", opts["A"])
        .replace("{option_b}", opts["B"])
        .replace("{option_c}", opts["C"])
        .replace("{option_d}", opts["D"])
        .replace("{option_e}", opts["E"])
    )


def answer_questions(
    *,
    run_name: str,
    model_key: str,
    max_workers: int,
    resume: bool,
    limit_articles: int | None = None,
) -> list[dict[str, Any]]:
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_key {model_key!r}")

    questions_path = _questions_dir(run_name, model_key) / "questions_1t3f_nota.jsonl"
    if not questions_path.exists():
        raise FileNotFoundError(f"Missing Module 3 questions: {questions_path}")

    summaries_path = _summaries_dir(run_name, model_key) / "generated_summaries.jsonl"
    if not summaries_path.exists():
        raise FileNotFoundError(f"Missing generated summaries: {summaries_path}")

    questions = load_jsonl(questions_path)
    summaries = {
        str(row["article_id"]): row
        for row in load_jsonl(summaries_path)
        if not row.get("parse_error") and str(row.get("generated_summary", "")).strip()
    }
    if not summaries:
        raise RuntimeError("No valid generated summaries available for answering.")

    if limit_articles is not None:
        allowed_ids = {str(r["id"]) for r in load_articles(run_name)[:limit_articles]}
        questions = [q for q in questions if str(q["article_id"]) in allowed_ids]

    missing_summary = sorted({str(q["article_id"]) for q in questions if str(q["article_id"]) not in summaries})
    if missing_summary:
        raise RuntimeError(f"Missing generated summary for articles: {missing_summary}")

    system_prompt_path = PROMPTS_DIR / "answer_1t3f_setup_c.txt"
    user_prompt_path = PROMPTS_DIR / "answer_1t3f_user.txt"
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    user_template = user_prompt_path.read_text(encoding="utf-8")

    out_dir = _answers_dir(run_name, model_key)

    usage = usage_tracker(run_name, resume=resume)
    client = ProviderClient(model_key, usage)
    existing = load_jsonl(out_dir / "module3_answers.jsonl") if resume and (out_dir / "module3_answers.jsonl").exists() else []
    done = {str(r.get("question_id")) for r in existing}

    def worker(question: dict[str, Any]) -> dict[str, Any]:
        article_id = str(question["article_id"])
        summary_row = summaries[article_id]
        generated_summary = str(summary_row["generated_summary"]).strip()
        opts = _normalize_options(question)
        correct_letter = str(question["correct_letter"]).strip().upper()

        option_eval: dict[str, dict[str, Any]] = {}
        pred_letter = "E"
        reasoning = ""
        parse_error = False
        json_repaired = False
        raw = ""

        try:
            raw = client.chat(
                stage=f"phase_b.answer_questions:{model_key}",
                item_id=str(question["question_id"]),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": _build_answer_user_message(
                            generated_summary=generated_summary,
                            question=question,
                            template=user_template,
                        ),
                    },
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            try:
                json.loads(str(raw).strip())
            except json.JSONDecodeError:
                json_repaired = True
            _obj, option_eval, pred_letter, reasoning = _parse_answer_from_raw(raw)
        except Exception as exc:
            parse_error = True
            reasoning = f"parse_or_runtime_error: {exc}"

        return _answer_row_from_parsed(
            question=question,
            summary_row=summary_row,
            model_key=model_key,
            opts=opts,
            correct_letter=correct_letter,
            option_eval=option_eval,
            pred_letter=pred_letter,
            reasoning=reasoning,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            system_prompt=system_prompt,
            user_template=user_template,
            parse_error=parse_error,
            raw_response=raw if parse_error else None,
            json_repaired=json_repaired,
        )

    todo = [q for q in questions if str(q["question_id"]) not in done]
    new_rows = run_parallel(
        todo,
        worker,
        max_workers=max_workers,
        desc=f"phase_b.answer_questions | {model_key}",
    )
    all_rows = existing + new_rows
    all_rows, repaired_count = _repair_answer_rows(all_rows)
    save_usage(run_name, usage)
    _save_answer_outputs(
        run_name=run_name,
        model_key=model_key,
        out_dir=out_dir,
        all_rows=all_rows,
        questions_expected=len(questions),
        max_workers=max_workers,
        limit_articles=limit_articles,
        system_prompt_path=system_prompt_path,
        user_prompt_path=user_prompt_path,
        system_prompt=system_prompt,
        user_template=user_template,
        usage_summary=usage.summarize(),
        json_repaired_count=sum(1 for r in all_rows if r.get("json_repaired")),
    )

    wrong_count = sum(1 for r in all_rows if r.get("parse_error") or not r.get("correct"))
    print(
        f"[phase_b] answer_questions {model_key} answers={len(all_rows)} "
        f"correct={sum(1 for r in all_rows if r.get('correct'))} "
        f"wrong={wrong_count} parse_errors={sum(1 for r in all_rows if r.get('parse_error'))} "
        f"json_repaired={sum(1 for r in all_rows if r.get('json_repaired'))}"
    )
    print(f"[phase_b] output -> {out_dir / 'module3_answers.jsonl'}")
    print(f"[phase_b] wrong_answers -> {out_dir / 'wrong_answers.jsonl'}")
    return all_rows


def repair_answers(*, run_name: str, model_key: str) -> list[dict[str, Any]]:
    out_dir = _answers_dir(run_name, model_key)
    out_path = out_dir / "module3_answers.jsonl"
    rows = load_jsonl(out_path)
    repaired_rows, repaired_count = _repair_answer_rows(rows)
    meta_path = out_dir / "module3_answers_metadata.json"
    meta = load_json(meta_path) if meta_path.exists() else {}
    system_prompt_path = Path(meta.get("prompt_file_system") or PROMPTS_DIR / "answer_1t3f_setup_c.txt")
    user_prompt_path = Path(meta.get("prompt_file_user") or PROMPTS_DIR / "answer_1t3f_user.txt")
    system_prompt = system_prompt_path.read_text(encoding="utf-8") if system_prompt_path.exists() else ""
    user_template = user_prompt_path.read_text(encoding="utf-8") if user_prompt_path.exists() else ""
    _save_answer_outputs(
        run_name=run_name,
        model_key=model_key,
        out_dir=out_dir,
        all_rows=repaired_rows,
        questions_expected=int(meta.get("questions_expected") or len(repaired_rows)),
        max_workers=int(meta.get("max_workers") or 1),
        limit_articles=meta.get("articles_limit"),
        system_prompt_path=system_prompt_path,
        user_prompt_path=user_prompt_path,
        system_prompt=system_prompt,
        user_template=user_template,
        usage_summary=meta.get("usage_summary") or {},
        json_repaired_count=repaired_count,
    )
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "stage": "05_module3_answers_repair",
            "model_key": model_key,
            "repaired_count": repaired_count,
            "parse_error_count": sum(1 for r in repaired_rows if r.get("parse_error")),
        },
        out_dir / "module3_answers_repair_metadata.json",
    )
    print(
        f"[phase_b] repair_answers repaired={repaired_count} "
        f"remaining_parse_errors={sum(1 for r in repaired_rows if r.get('parse_error'))}"
    )
    return repaired_rows


def _rewritten_dir(run_name: str, model_key: str) -> Path:
    path = run_data_dir(run_name) / "06_rewritten" / model_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _format_rewrite_feedback(wrong_rows: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for i, row in enumerate(wrong_rows, start=1):
        fact = str(row.get("fact", "")).strip()
        correct_letter = str(row.get("correct_letter", "")).strip().upper()
        correct_answer = str(row.get("correct_answer", "")).strip()
        predicted_letter = str(row.get("predicted_letter", "")).strip().upper()
        reasoning = str(row.get("reasoning", "")).strip()
        af_id = str(row.get("af_id", "")).strip()
        lines = [f"- AF {af_id}: {fact}"]
        if row.get("parse_error"):
            lines.append("  Quiz checker: parse error (fact still must be covered).")
        else:
            lines.append(f'  Quiz: correct={correct_letter} "{correct_answer}"')
            lines.append(f"  Checker chose: {predicted_letter} — {reasoning}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "(no feedback)"


def rewrite_summaries(
    *,
    run_name: str,
    model_key: str,
    max_workers: int,
    resume: bool,
    limit_articles: int | None = None,
) -> list[dict[str, Any]]:
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_key {model_key!r}")

    articles = load_articles(run_name)
    if limit_articles is not None:
        articles = articles[:limit_articles]
    articles_by_id = {str(a["id"]): a for a in articles}

    summaries_path = _summaries_dir(run_name, model_key) / "generated_summaries.jsonl"
    if not summaries_path.exists():
        raise FileNotFoundError(f"Missing generated summaries: {summaries_path}")

    wrong_path = _answers_dir(run_name, model_key) / "wrong_answers.jsonl"
    if not wrong_path.exists():
        raise FileNotFoundError(f"Missing wrong answers: {wrong_path}")

    generated_rows = {
        str(r["article_id"]): r
        for r in load_jsonl(summaries_path)
        if not r.get("parse_error") and str(r.get("generated_summary", "")).strip()
    }
    wrong_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(wrong_path):
        aid = str(row["article_id"])
        if aid in articles_by_id:
            wrong_by_article[aid].append(row)

    prompt_path = PROMPTS_DIR / "rewrite_lay_summary.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")

    out_dir = _rewritten_dir(run_name, model_key)
    out_path = out_dir / "rewritten_summaries.jsonl"

    usage = usage_tracker(run_name, resume=resume)
    client = ProviderClient(model_key, usage)
    existing = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("article_id")) for r in existing}

    def worker(article: dict[str, Any]) -> dict[str, Any]:
        article_id = str(article["id"])
        gen_row = generated_rows.get(article_id)
        if gen_row is None:
            raise RuntimeError(f"No valid generated summary for {article_id}")

        generated_summary = str(gen_row["generated_summary"]).strip()
        expert_wc = int(article.get("expert_summary_word_count") or word_count(article.get("expert_summary", "")))
        generated_wc = int(gen_row.get("generated_word_count") or word_count(generated_summary))
        min_word_count = max(expert_wc, generated_wc)

        wrong_rows = wrong_by_article.get(article_id, [])
        feedback_items = [
            {
                "af_id": row.get("af_id"),
                "fact": row.get("fact", ""),
                "correct_letter": row.get("correct_letter"),
                "predicted_letter": row.get("predicted_letter"),
                "reasoning": row.get("reasoning", ""),
                "parse_error": bool(row.get("parse_error")),
            }
            for row in wrong_rows
        ]
        wrong_af_ids = sorted({str(r.get("af_id")) for r in wrong_rows if r.get("af_id")})

        if not wrong_rows:
            rewritten_summary = generated_summary
            return {
                "article_id": article_id,
                "source_dataset": article.get("source_dataset"),
                "original_index": article.get("original_index"),
                "model_key": model_key,
                "rewriter_model": MODEL_CONFIGS[model_key]["model"],
                "rewriter_provider": MODEL_CONFIGS[model_key]["provider"],
                "prompt_version": REWRITE_PROMPT_VERSION,
                "prompt_file": str(prompt_path),
                "prompt_sha256": sha256_text(prompt_template),
                "generated_summary": generated_summary,
                "rewritten_summary": rewritten_summary,
                "rewrite_skipped_no_errors": True,
                "wrong_feedback_count": 0,
                "wrong_af_ids": [],
                "feedback_items": [],
                "revision_notes": ["no wrong quiz answers; copied generated summary"],
                "expert_summary_word_count": expert_wc,
                "generated_word_count": generated_wc,
                "rewritten_word_count": word_count(rewritten_summary),
                "min_word_count": min_word_count,
                "below_min_length": word_count(rewritten_summary) < min_word_count,
                "parse_error": False,
                "raw_response": None,
            }

        feedback = _format_rewrite_feedback(wrong_rows)
        raw = ""
        rewritten_summary = generated_summary
        revision_notes: list[str] = []
        parse_error = False

        try:
            prompt = (
                prompt_template.replace("{abstract}", str(article.get("abstract", "")))
                .replace("{generated_summary}", generated_summary)
                .replace("{feedback}", feedback)
                .replace("{min_word_count}", str(min_word_count))
            )
            raw = client.chat(
                stage=f"phase_b.rewrite_summary:{model_key}",
                item_id=article_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = _loads_json_object(raw)
            rewritten_summary = str(obj.get("rewritten_summary", obj.get("summary", ""))).strip()
            raw_notes = obj.get("revision_notes", [])
            if isinstance(raw_notes, list):
                revision_notes = [str(x).strip() for x in raw_notes if str(x).strip()]
            elif raw_notes:
                revision_notes = [str(raw_notes).strip()]
            if not rewritten_summary:
                raise ValueError("empty rewritten_summary")
        except Exception as exc:
            parse_error = True
            revision_notes = [f"parse_or_runtime_error: {exc}"]
            rewritten_summary = generated_summary

        rewritten_wc = word_count(rewritten_summary)
        return {
            "article_id": article_id,
            "source_dataset": article.get("source_dataset"),
            "original_index": article.get("original_index"),
            "model_key": model_key,
            "rewriter_model": MODEL_CONFIGS[model_key]["model"],
            "rewriter_provider": MODEL_CONFIGS[model_key]["provider"],
            "prompt_version": REWRITE_PROMPT_VERSION,
            "prompt_file": str(prompt_path),
            "prompt_sha256": sha256_text(prompt_template),
            "generated_summary": generated_summary,
            "rewritten_summary": rewritten_summary,
            "rewrite_skipped_no_errors": False,
            "wrong_feedback_count": len(wrong_rows),
            "wrong_af_ids": wrong_af_ids,
            "feedback_items": feedback_items,
            "revision_notes": revision_notes,
            "expert_summary_word_count": expert_wc,
            "generated_word_count": generated_wc,
            "rewritten_word_count": rewritten_wc,
            "min_word_count": min_word_count,
            "below_min_length": rewritten_wc < min_word_count,
            "parse_error": parse_error,
            "raw_response": raw if parse_error else None,
        }

    missing = [str(a["id"]) for a in articles if str(a["id"]) not in generated_rows]
    if missing:
        raise RuntimeError(f"Missing generated summaries for articles: {missing}")

    todo = [art for art in articles if str(art["id"]) not in done]
    new_rows = run_parallel(
        todo,
        worker,
        max_workers=max_workers,
        desc=f"phase_b.rewrite_summary | {model_key}",
    )
    all_rows = existing + new_rows
    all_rows.sort(key=lambda r: str(r["article_id"]))
    save_jsonl(all_rows, out_path)
    save_usage(run_name, usage)

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "stage": "06_rewritten",
            "model_key": model_key,
            "provider": MODEL_CONFIGS[model_key]["provider"],
            "model": MODEL_CONFIGS[model_key]["model"],
            "max_workers": max_workers,
            "articles_limit": limit_articles,
            "prompt_version": REWRITE_PROMPT_VERSION,
            "prompt_file": str(prompt_path),
            "prompt_sha256": sha256_text(prompt_template),
            "rewritten_total": len(all_rows),
            "skipped_no_errors_count": sum(1 for r in all_rows if r.get("rewrite_skipped_no_errors")),
            "parse_error_count": sum(1 for r in all_rows if r.get("parse_error")),
            "below_min_length_count": sum(1 for r in all_rows if r.get("below_min_length")),
            "wrong_feedback_stats": _word_stats([int(r.get("wrong_feedback_count") or 0) for r in all_rows]),
            "rewritten_word_count_stats": _word_stats([int(r.get("rewritten_word_count") or 0) for r in all_rows]),
            "usage_summary": usage.summarize(),
        },
        out_dir / "rewritten_summaries_metadata.json",
    )

    print(
        f"[phase_b] rewrite_summary {model_key} rewritten={len(all_rows)} "
        f"skipped={sum(1 for r in all_rows if r.get('rewrite_skipped_no_errors'))} "
        f"parse_errors={sum(1 for r in all_rows if r.get('parse_error'))} "
        f"below_min_length={sum(1 for r in all_rows if r.get('below_min_length'))}"
    )
    print(f"[phase_b] output -> {out_path}")
    return all_rows


def _word_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 1),
    }


def repair_summaries(*, run_name: str, model_key: str) -> list[dict[str, Any]]:
    final_af_path = _module2_dir(run_name, model_key) / "final_keep_af.jsonl"
    final_af = load_jsonl(final_af_path)
    af_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_af:
        af_by_article[str(row["article_id"])].append(row)

    out_dir = _summaries_dir(run_name, model_key)
    out_path = out_dir / "generated_summaries.jsonl"
    rows = load_jsonl(out_path)
    repaired = 0
    for row in rows:
        if not row.get("parse_error") or not row.get("raw_response"):
            continue
        article_id = str(row["article_id"])
        valid_af_ids = {str(af["af_id"]) for af in af_by_article.get(article_id, [])}
        try:
            obj = json.loads(str(row["raw_response"]))
            abstract_claim_plan, sentences, validation_errors, skipped_sentences = _validate_generation(
                obj,
                valid_af_ids=valid_af_ids,
            )
            if validation_errors:
                continue
            generated_summary = " ".join(s["text"] for s in sentences if s["text"])
            if not generated_summary.strip():
                continue
            af_ids_used = sorted({af_id for s in sentences for af_id in s.get("af_ids", [])})
            generated_wc = word_count(generated_summary)
            min_word_count = int(row.get("min_word_count") or row.get("expert_summary_word_count") or 0)
            row.update(
                {
                    "abstract_claim_plan": abstract_claim_plan,
                    "sentences": sentences,
                    "generated_summary": generated_summary,
                    "af_ids_used": af_ids_used,
                    "af_count_cited": len(af_ids_used),
                    "generated_word_count": generated_wc,
                    "below_min_length": generated_wc < min_word_count,
                    "parse_error": False,
                    "validation_errors": [],
                    "skipped_sentences": skipped_sentences,
                    "raw_response": None,
                }
            )
            repaired += 1
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    rows.sort(key=lambda r: str(r["article_id"]))
    save_jsonl(rows, out_path)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "stage": "04_summaries_repair",
            "model_key": model_key,
            "repaired_count": repaired,
            "parse_error_count": sum(1 for r in rows if r.get("parse_error")),
            "skipped_sentence_count": sum(len(r.get("skipped_sentences") or []) for r in rows),
        },
        out_dir / "generated_summaries_repair_metadata.json",
    )
    print(f"[phase_b] repair_summaries repaired={repaired} remaining_parse_errors={sum(1 for r in rows if r.get('parse_error'))}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Thesis lay-summary pipeline — Phase B (steps 4–6)."
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--model-key", type=str, default=DEFAULT_MODEL_KEY)
    parser.add_argument(
        "--step",
        type=str,
        default="generate-summary",
        choices=["generate-summary", "repair-summaries", "answer-questions", "repair-answers", "rewrite-summaries"],
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit-articles", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.step == "generate-summary":
        generate_summaries(
            run_name=args.run_name,
            model_key=args.model_key,
            max_workers=args.max_workers,
            resume=not args.no_resume,
            limit_articles=args.limit_articles,
        )
    elif args.step == "repair-summaries":
        repair_summaries(run_name=args.run_name, model_key=args.model_key)
    elif args.step == "answer-questions":
        answer_questions(
            run_name=args.run_name,
            model_key=args.model_key,
            max_workers=args.max_workers,
            resume=not args.no_resume,
            limit_articles=args.limit_articles,
        )
    elif args.step == "repair-answers":
        repair_answers(run_name=args.run_name, model_key=args.model_key)
    elif args.step == "rewrite-summaries":
        rewrite_summaries(
            run_name=args.run_name,
            model_key=args.model_key,
            max_workers=args.max_workers,
            resume=not args.no_resume,
            limit_articles=args.limit_articles,
        )
    else:
        raise ValueError(f"Unsupported step: {args.step}")


if __name__ == "__main__":
    main()
