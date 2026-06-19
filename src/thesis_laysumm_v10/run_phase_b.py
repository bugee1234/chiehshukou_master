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
from src.thesis_laysumm.llm_utils import load_articles, run_parallel, save_usage, sha256_text, usage_tracker
from src.thesis_laysumm.paths import run_data_dir
from src.thesis_laysumm_v10.common import (
    BANNED_FREE_IMPLICATION_PHRASES,
    PROMPTS_DIR,
    compact_json,
    dataset_profile,
    bad_sentence_issues,
    find_mojibake,
    length_policy,
    loads_json_object,
    readability_policy,
    readability_proxy_score,
    readability_stats,
    require_mode,
    safe_word_count,
    sentence_risk_score,
    split_sentences,
)
from src.utils import load_jsonl, save_json, save_jsonl

PROMPTS_DIR_V2 = ROOT_DIR / "src" / "thesis_laysumm" / "prompts"
DEFAULT_MODEL_KEY = "gpt41_mini"


def _evidence_dir(run_name: str, model_key: str, mode: str) -> Path:
    return run_data_dir(run_name) / "02_5_evidence_table" / mode / model_key


def _questions_dir(run_name: str, model_key: str, mode: str) -> Path:
    return run_data_dir(run_name) / "03_questions_v10" / mode / model_key


def _summaries_dir(run_name: str, model_key: str) -> Path:
    path = run_data_dir(run_name) / "04_summaries" / model_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _answers_dir(run_name: str, model_key: str, mode: str) -> Path:
    path = run_data_dir(run_name) / "05_module3_answers_v10" / mode / model_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rewritten_dir(run_name: str, model_key: str) -> Path:
    path = run_data_dir(run_name) / "06_rewritten" / model_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_evidence_by_article(run_name: str, model_key: str, mode: str) -> dict[str, dict[str, Any]]:
    path = _evidence_dir(run_name, model_key, mode) / "evidence_table.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing evidence table: {path}")
    return {str(r["article_id"]): r for r in load_jsonl(path) if not r.get("parse_error")}


def _format_evidence_table(article_row: dict[str, Any]) -> str:
    rows = []
    for row in article_row.get("evidence_rows") or []:
        rows.append(
            {
                "evidence_row_id": row.get("evidence_row_id"),
                "abstract_sentence_idx": row.get("abstract_sentence_idx"),
                "abstract_claim": row.get("abstract_claim"),
                "core_keep_af": row.get("core_keep_af"),
                "expert_summary_hint": row.get("expert_summary_hint"),
                "lay_context": row.get("lay_context"),
                "story_role": row.get("story_role"),
                "must_use_terms": row.get("must_use_terms"),
                "allowed_evidence_spans": row.get("allowed_evidence_spans"),
                "forbidden_overgeneralizations": row.get("forbidden_overgeneralizations"),
            }
        )
    return compact_json(rows)


def _slot_sentence(slot: dict[str, Any]) -> str:
    parts = [
        str(slot.get("claim_sentence", "")).strip(),
        str(slot.get("clarification_sentence", "")).strip(),
    ]
    return " ".join(part for part in parts if part)


def _summary_from_slots(slots: list[dict[str, Any]]) -> str:
    return " ".join(_slot_sentence(slot) for slot in slots if slot.get("used_for_summary", True)).strip()


def _format_slots(slots: list[dict[str, Any]]) -> str:
    return compact_json(
        [
            {
                "slot_id": slot.get("slot_id"),
                "role": slot.get("role"),
                "evidence_row_id": slot.get("evidence_row_id"),
                "claim_sentence": slot.get("claim_sentence"),
                "clarification_sentence": slot.get("clarification_sentence", ""),
                "used_for_summary": slot.get("used_for_summary", True),
            }
            for slot in slots
        ]
    )


def _contains_banned_phrase(text: str) -> list[str]:
    lower = str(text or "").lower()
    return [phrase for phrase in BANNED_FREE_IMPLICATION_PHRASES if phrase in lower]


def _normalize_slots(obj: dict[str, Any], valid_row_ids: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    raw_slots = obj.get("slots", [])
    if not isinstance(raw_slots, list) or not raw_slots:
        return [], ["slots must be a non-empty list"]

    slots: list[dict[str, Any]] = []
    seen_slot_ids: set[str] = set()
    allowed_roles = {"background", "motivation", "method", "mechanism", "finding", "implication", "limitation"}
    for i, raw in enumerate(raw_slots, start=1):
        if not isinstance(raw, dict):
            errors.append("each slot must be an object")
            continue
        slot_id = str(raw.get("slot_id") or f"slot_{i:04d}").strip()
        if slot_id in seen_slot_ids:
            slot_id = f"slot_{i:04d}"
        seen_slot_ids.add(slot_id)
        row_id = str(raw.get("evidence_row_id", "")).strip()
        if row_id not in valid_row_ids:
            errors.append(f"{slot_id}: unknown evidence_row_id: {row_id}")
            continue
        role = str(raw.get("role", "finding")).strip().lower() or "finding"
        if role not in allowed_roles:
            role = "finding"
        claim = str(raw.get("claim_sentence", "")).strip()
        clarification = str(raw.get("clarification_sentence", "")).strip()
        used = raw.get("used_for_summary", True)
        used = used if isinstance(used, bool) else str(used).strip().lower() not in {"false", "0", "no"}
        if used and not claim and not clarification:
            errors.append(f"{slot_id}: used slot missing claim_sentence or clarification_sentence")
            continue
        banned = _contains_banned_phrase(f"{claim} {clarification}")
        slots.append(
            {
                "slot_id": slot_id,
                "role": role,
                "evidence_row_id": row_id,
                "claim_sentence": claim,
                "clarification_sentence": clarification,
                "used_for_summary": used,
                "banned_phrase_warnings": banned,
            }
        )
    return slots, errors


def _slots_to_sentences(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sentences: list[dict[str, Any]] = []
    for slot in slots:
        if not slot.get("used_for_summary", True):
            continue
        for sentence_type, text in (
            ("claim", str(slot.get("claim_sentence", "")).strip()),
            ("clarification", str(slot.get("clarification_sentence", "")).strip()),
        ):
            if text:
                sentences.append(
                    {
                        "text": text,
                        "evidence_row_id": slot["evidence_row_id"],
                        "section": slot.get("role", "finding"),
                        "slot_id": slot.get("slot_id"),
                        "sentence_type": sentence_type,
                    }
                )
    return sentences


def _fit_slots_to_length(slots: list[dict[str, Any]], policy: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    fitted = [dict(slot) for slot in slots]
    notes: list[str] = []
    max_words = int(policy["max_word_count"])
    min_words = int(policy["min_word_count"])
    if safe_word_count(_summary_from_slots(fitted)) <= max_words:
        return fitted, notes

    for slot in reversed(fitted):
        old = str(slot.get("clarification_sentence", "")).strip()
        if not old:
            continue
        slot["clarification_sentence"] = ""
        wc = safe_word_count(_summary_from_slots(fitted))
        notes.append(f"removed clarification from {slot.get('slot_id')} to fit length; wc={wc}")
        if wc <= max_words:
            return fitted, notes

    role_drop_order = {
        "limitation": 0,
        "background": 1,
        "implication": 2,
        "motivation": 3,
        "mechanism": 4,
        "method": 5,
        "finding": 6,
    }
    candidates = sorted(
        [
            (role_drop_order.get(str(slot.get("role")), 3), idx, slot)
            for idx, slot in enumerate(fitted)
            if slot.get("used_for_summary", True)
        ],
        key=lambda item: (item[0], -item[1]),
    )
    for _, _, slot in candidates:
        if sum(1 for s in fitted if s.get("used_for_summary", True)) <= 5:
            break
        old_used = slot.get("used_for_summary", True)
        slot["used_for_summary"] = False
        wc = safe_word_count(_summary_from_slots(fitted))
        if wc >= min_words:
            notes.append(f"disabled {slot.get('slot_id')} to fit length; wc={wc}")
            if wc <= max_words:
                return fitted, notes
        else:
            slot["used_for_summary"] = old_used
    return fitted, notes


def _merge_slots(current_slots: list[dict[str, Any]], revised_slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not current_slots:
        return revised_slots
    if len(revised_slots) >= len(current_slots):
        return revised_slots

    by_slot_id = {str(slot.get("slot_id")): slot for slot in revised_slots if slot.get("slot_id")}
    by_row_id = {str(slot.get("evidence_row_id")): slot for slot in revised_slots if slot.get("evidence_row_id")}
    merged: list[dict[str, Any]] = []
    used_revised_ids: set[int] = set()
    for slot in current_slots:
        replacement = by_slot_id.get(str(slot.get("slot_id"))) or by_row_id.get(str(slot.get("evidence_row_id")))
        if replacement is None:
            merged.append(slot)
            continue
        patched = dict(slot)
        for key in ("role", "evidence_row_id", "used_for_summary"):
            if replacement.get(key) not in {None, ""}:
                patched[key] = replacement[key]
        for key in ("claim_sentence", "clarification_sentence"):
            value = str(replacement.get(key, "")).strip()
            if value:
                patched[key] = value
        patched["banned_phrase_warnings"] = replacement.get("banned_phrase_warnings", [])
        merged.append(patched)
        used_revised_ids.add(id(replacement))
    for slot in revised_slots:
        if id(slot) not in used_revised_ids:
            merged.append(slot)
    return merged


def _expand_if_needed(
    *,
    client: ProviderClient,
    stage_prefix: str,
    item_id: str,
    mode: str,
    source_dataset: str,
    policy: dict[str, Any],
    evidence: dict[str, Any],
    slots: list[dict[str, Any]],
    summary: str,
    max_attempts: int = 2,
) -> tuple[list[dict[str, Any]], str, list[str], list[str]]:
    attempts: list[str] = []
    errors: list[str] = []
    valid_row_ids = {str(r["evidence_row_id"]) for r in evidence.get("evidence_rows") or []}
    if safe_word_count(summary) >= int(policy["min_word_count"]):
        return slots, summary, attempts, errors

    prompt_path = PROMPTS_DIR / "expand_summary_v10.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")
    current_slots = slots
    current_summary = summary
    for attempt in range(1, max_attempts + 1):
        attempts.append(f"attempt_{attempt}: before_wc={safe_word_count(current_summary)}")
        try:
            raw = client.chat(
                stage=f"{stage_prefix}.expand_length.{mode}",
                item_id=f"{item_id}:expand:{attempt}",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            prompt_template.replace("{mode}", mode)
                            .replace("{dataset_profile}", dataset_profile(source_dataset))
                            .replace("{style_profile}", str(policy.get("style_profile", "compact_factual")))
                            .replace("{target_word_count}", str(policy["target_word_count"]))
                            .replace("{min_word_count}", str(policy["min_word_count"]))
                            .replace("{max_word_count}", str(policy["max_word_count"]))
                            .replace("{current_slots}", _format_slots(current_slots))
                            .replace("{current_summary}", current_summary)
                            .replace("{evidence_table}", _format_evidence_table(evidence))
                        ),
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = loads_json_object(raw)
            expanded_slots, validation_errors = _normalize_slots(obj, valid_row_ids)
            expanded_summary = str(obj.get("summary", "")).strip() or _summary_from_slots(expanded_slots)
        except Exception as exc:
            errors.append(f"expand_{attempt}: {exc}")
            continue
        if validation_errors:
            errors.extend(f"expand_{attempt}: {err}" for err in validation_errors)
            continue
        if expanded_summary:
            current_slots = expanded_slots
            current_summary = expanded_summary
        attempts[-1] += f" after_wc={safe_word_count(current_summary)}"
        if safe_word_count(current_summary) >= int(policy["min_word_count"]):
            break
    return current_slots, current_summary, attempts, errors


def generate_summaries(
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
    articles = load_articles(run_name)
    if limit_articles is not None:
        articles = articles[:limit_articles]
    evidence_by_article = _load_evidence_by_article(run_name, model_key, mode)

    prompt_path = PROMPTS_DIR / "generate_summary_v10.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")
    out_dir = _summaries_dir(run_name, model_key)
    out_path = out_dir / "generated_summaries.jsonl"

    usage = usage_tracker(run_name, resume=resume)
    client = ProviderClient(model_key, usage)
    existing = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("article_id")) for r in existing}

    def worker(article: dict[str, Any]) -> dict[str, Any]:
        aid = str(article["id"])
        evidence = evidence_by_article.get(aid)
        if evidence is None:
            raise RuntimeError(f"Missing valid evidence table for {aid}")
        policy = length_policy(
            mode=mode,
            expert_word_count=int(article.get("expert_summary_word_count") or 0),
            source_dataset=str(article.get("source_dataset", "")),
        )
        valid_row_ids = {str(r["evidence_row_id"]) for r in evidence.get("evidence_rows") or []}

        raw = ""
        parse_error = False
        validation_errors: list[str] = []
        slots: list[dict[str, Any]] = []
        sentences: list[dict[str, Any]] = []
        generated_summary = ""
        expansion_attempts: list[str] = []
        expansion_errors: list[str] = []
        length_fit_notes: list[str] = []
        try:
            prompt = (
                prompt_template.replace("{mode}", mode)
                .replace("{title}", str(article.get("title", "")))
                .replace("{abstract}", str(article.get("abstract", "")))
                .replace("{evidence_table}", _format_evidence_table(evidence))
                .replace("{dataset_profile}", dataset_profile(str(article.get("source_dataset", ""))))
                .replace("{target_word_count}", str(policy["target_word_count"]))
                .replace("{min_word_count}", str(policy["min_word_count"]))
                .replace("{max_word_count}", str(policy["max_word_count"]))
                .replace("{style_profile}", str(policy.get("style_profile", "compact_factual")))
                .replace(
                    "{readability_policy}",
                    compact_json(
                        readability_policy(
                            source_dataset=str(article.get("source_dataset", "")),
                            expert_word_count=int(article.get("expert_summary_word_count") or 0),
                        )
                    ),
                )
            )
            raw = client.chat(
                stage=f"V10.generate_summary.{mode}:{model_key}",
                item_id=aid,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = loads_json_object(raw)
            slots, validation_errors = _normalize_slots(obj, valid_row_ids)
            generated_summary = str(obj.get("summary", "")).strip()
            if not generated_summary:
                generated_summary = _summary_from_slots(slots)
            if not validation_errors:
                slots, generated_summary, expansion_attempts, expand_errors = _expand_if_needed(
                    client=client,
                    stage_prefix=f"V10.generate_summary.{model_key}",
                    item_id=aid,
                    mode=mode,
                    source_dataset=str(article.get("source_dataset", "")),
                    policy=policy,
                    evidence=evidence,
                    slots=slots,
                    summary=generated_summary,
                )
                expansion_errors.extend(expand_errors)
            slots, length_fit_notes = _fit_slots_to_length(slots, policy)
            generated_summary = _summary_from_slots(slots)
            sentences = _slots_to_sentences(slots)
            if validation_errors or not generated_summary:
                raise ValueError("; ".join(validation_errors) or "empty generated summary")
        except Exception as exc:
            parse_error = True
            validation_errors = validation_errors or [str(exc)]

        wc = safe_word_count(generated_summary)
        return {
            "article_id": aid,
            "source_dataset": article.get("source_dataset"),
            "original_index": article.get("original_index"),
            "model_key": model_key,
            "mode": mode,
            "prompt_version": f"V10_generate_{mode}",
            "prompt_file": str(prompt_path),
            "prompt_sha256": sha256_text(prompt_template),
            "dataset_profile": dataset_profile(str(article.get("source_dataset", ""))),
            "length_policy": policy,
            "expert_summary_word_count": int(article.get("expert_summary_word_count") or 0),
            "target_word_count": policy["target_word_count"],
            "min_word_count": policy["min_word_count"],
            "max_word_count": policy["max_word_count"],
            "slots": slots,
            "sentences": sentences,
            "generated_summary": generated_summary,
            "generated_word_count": wc,
            "readability_stats": readability_stats(generated_summary),
            "readability_proxy_score": readability_proxy_score(generated_summary),
            "mojibake_warnings": find_mojibake(generated_summary),
            "below_min_length": bool(generated_summary) and wc < policy["min_word_count"],
            "above_max_length": bool(generated_summary) and wc > policy["max_word_count"],
            "parse_error": parse_error,
            "validation_errors": validation_errors,
            "length_expansion_attempts": expansion_attempts,
            "length_expansion_errors": expansion_errors,
            "length_fit_notes": length_fit_notes,
            "raw_response": raw if parse_error else None,
        }

    todo = [a for a in articles if str(a["id"]) not in done]
    new_rows = run_parallel(todo, worker, max_workers=max_workers, desc=f"V10 generate | {mode} | {model_key}")
    all_rows = existing + new_rows
    all_rows.sort(key=lambda r: str(r["article_id"]))
    save_jsonl(all_rows, out_path)
    save_usage(run_name, usage)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "stage": "V10_04_summaries",
            "mode": mode,
            "model_key": model_key,
            "prompt_file": str(prompt_path),
            "prompt_sha256": sha256_text(prompt_template),
            "summaries_total": len(all_rows),
            "parse_error_count": sum(1 for r in all_rows if r.get("parse_error")),
            "below_min_length_count": sum(1 for r in all_rows if r.get("below_min_length")),
            "above_max_length_count": sum(1 for r in all_rows if r.get("above_max_length")),
            "usage_summary": usage.summarize(),
        },
        out_dir / "generated_summaries_metadata.json",
    )
    print(f"[V10 generate] {mode} {model_key} summaries={len(all_rows)} -> {out_path}")
    return all_rows


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


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


def _decide_answer(option_eval: dict[str, dict[str, Any]], obj: dict[str, Any]) -> str:
    direct = str(obj.get("final_answer", "")).strip().upper()
    if direct in {"A", "B", "C", "D", "E"}:
        return direct
    supported = [
        letter
        for letter in "ABCD"
        if str((option_eval.get(letter) or {}).get("status")) == "supported"
        and not _coerce_bool((option_eval.get(letter) or {}).get("negation_flag"))
        and not _coerce_bool((option_eval.get(letter) or {}).get("strictness_mismatch"))
    ]
    if len(supported) == 1:
        return supported[0]
    if not supported:
        return "E"
    return max(supported, key=lambda x: len(str((option_eval.get(x) or {}).get("evidence", ""))))


def answer_questions(
    *,
    run_name: str,
    model_key: str,
    mode: str,
    max_workers: int,
    resume: bool,
    limit_articles: int | None = None,
) -> list[dict[str, Any]]:
    require_mode(mode)
    questions_path = _questions_dir(run_name, model_key, mode) / "questions_1t3f_nota.jsonl"
    summaries_path = _summaries_dir(run_name, model_key) / "generated_summaries.jsonl"
    if not questions_path.exists():
        raise FileNotFoundError(f"Missing V10 questions: {questions_path}")
    if not summaries_path.exists():
        raise FileNotFoundError(f"Missing V10 generated summaries: {summaries_path}")

    questions = load_jsonl(questions_path)
    if limit_articles is not None:
        allowed = sorted({str(r["article_id"]) for r in questions})[:limit_articles]
        questions = [r for r in questions if str(r["article_id"]) in set(allowed)]
    summaries = {
        str(r["article_id"]): r
        for r in load_jsonl(summaries_path)
        if not r.get("parse_error") and str(r.get("generated_summary", "")).strip()
    }

    out_dir = _answers_dir(run_name, model_key, mode)
    out_path = out_dir / "module3_answers.jsonl"
    wrong_path = out_dir / "wrong_answers.jsonl"
    system_prompt_path = PROMPTS_DIR_V2 / "answer_1t3f_setup_c.txt"
    user_prompt_path = PROMPTS_DIR_V2 / "answer_1t3f_user.txt"
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    user_template = user_prompt_path.read_text(encoding="utf-8")

    usage = usage_tracker(run_name, resume=resume)
    client = ProviderClient(model_key, usage)
    existing = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("question_id")) for r in existing}

    def worker(question: dict[str, Any]) -> dict[str, Any]:
        aid = str(question["article_id"])
        summary_row = summaries.get(aid)
        if summary_row is None:
            raise RuntimeError(f"Missing generated summary for {aid}")
        opts = {letter: str((question.get("options") or {}).get(letter, "")).strip() for letter in "ABCDE"}
        raw = ""
        parse_error = False
        obj: dict[str, Any] = {}
        option_eval: dict[str, dict[str, Any]] = {}
        pred = "E"
        reasoning = ""
        try:
            user_prompt = (
                user_template.replace("{generated_summary}", str(summary_row["generated_summary"]))
                .replace("{option_a}", opts["A"])
                .replace("{option_b}", opts["B"])
                .replace("{option_c}", opts["C"])
                .replace("{option_d}", opts["D"])
                .replace("{option_e}", opts["E"])
            )
            raw = client.chat(
                stage=f"V10.answer_questions.{mode}:{model_key}",
                item_id=str(question["question_id"]),
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = loads_json_object(raw)
            option_eval = _parse_option_eval(obj)
            pred = _decide_answer(option_eval, obj)
            reasoning = str(obj.get("reasoning", "")).strip()
        except Exception as exc:
            parse_error = True
            reasoning = f"parse_or_runtime_error: {exc}"

        correct = (not parse_error) and pred == str(question.get("correct_letter", "")).strip().upper()
        return {
            "question_id": question["question_id"],
            "af_id": question["af_id"],
            "article_id": aid,
            "source_dataset": question.get("source_dataset"),
            "original_index": question.get("original_index"),
            "model_key": model_key,
            "mode": mode,
            "evidence_row_id": question.get("evidence_row_id"),
            "check_type": question.get("check_type"),
            "slot_type": question.get("slot_type"),
            "fact": question.get("fact", ""),
            "source_span": question.get("source_span", ""),
            "generated_summary_word_count": int(summary_row.get("generated_word_count") or 0),
            "options": opts,
            "correct_letter": question.get("correct_letter"),
            "predicted_letter": pred,
            "correct": correct,
            "correct_answer": opts.get(str(question.get("correct_letter", "")).strip().upper(), ""),
            "predicted_answer": opts.get(pred, ""),
            "option_evaluation": option_eval,
            "reasoning": reasoning,
            "parse_error": parse_error,
            "raw_response": raw if parse_error else None,
            "context_mode": "V10_full_generated_summary",
        }

    todo = [q for q in questions if str(q.get("question_id")) not in done]
    new_rows = run_parallel(todo, worker, max_workers=max_workers, desc=f"V10 answer | {mode} | {model_key}")
    all_rows = existing + new_rows
    all_rows.sort(key=lambda r: (str(r["article_id"]), str(r["question_id"])))
    wrong_rows = [r for r in all_rows if not r.get("correct")]
    save_jsonl(all_rows, out_path)
    save_jsonl(wrong_rows, wrong_path)
    save_usage(run_name, usage)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "stage": "V10_05_answers",
            "mode": mode,
            "model_key": model_key,
            "questions_total": len(questions),
            "answers_total": len(all_rows),
            "wrong_total": len(wrong_rows),
            "accuracy": round((len(all_rows) - len(wrong_rows)) / len(all_rows), 4) if all_rows else 0.0,
            "wrong_check_type_counts": dict(Counter(str(r.get("check_type")) for r in wrong_rows)),
            "wrong_slot_type_counts": dict(Counter(str(r.get("slot_type")) for r in wrong_rows)),
            "usage_summary": usage.summarize(),
        },
        out_dir / "module3_answers_metadata.json",
    )
    print(f"[V10 answer] {mode} {model_key} answers={len(all_rows)} wrong={len(wrong_rows)}")
    return all_rows


def _feedback_for_article(wrong_rows: list[dict[str, Any]]) -> str:
    if not wrong_rows:
        return "(no failed checks)"
    items = []
    for i, row in enumerate(wrong_rows, start=1):
        items.append(
            {
                "idx": i,
                "evidence_row_id": row.get("evidence_row_id"),
                "check_type": row.get("check_type"),
                "slot_type": row.get("slot_type"),
                "expected_fact": row.get("fact"),
                "correct_answer": row.get("correct_answer"),
                "predicted_answer": row.get("predicted_answer"),
                "reasoning": row.get("reasoning"),
            }
        )
    return compact_json(items)


def _evidence_by_row_id(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("evidence_row_id")): row for row in evidence.get("evidence_rows") or []}


def _first_sentence(text: str, *, max_words: int = 18) -> str:
    sentence = (split_sentences(text) or [str(text or "").strip()])[0].strip()
    words = sentence.split()
    if len(words) <= max_words:
        return sentence
    return ""


def _clean_extra_sentence(text: str, *, max_words: int = 16) -> str:
    raw_sentences = split_sentences(str(text or "").strip())
    candidates = raw_sentences or [str(text or "").strip()]
    sentence = ""
    for raw_sentence in candidates:
        candidate = str(raw_sentence or "").strip().strip("\"'")
        if not candidate:
            continue
        wc = safe_word_count(candidate)
        if wc <= max_words:
            sentence = candidate
            break
    if not sentence:
        return ""
    if not sentence.endswith((".", "!", "?")):
        sentence = sentence.rstrip(" ,;:") + "."
    if find_mojibake(sentence):
        return ""
    if safe_word_count(sentence) < 5 or safe_word_count(sentence) > max_words + 1:
        return ""
    lower = sentence.lower()
    if lower.startswith(("this ", "these ", "it ", "they ", "this change", "this finding", "this means", "this shows")):
        return ""
    if any(term in lower for term in ("important for", "helps scientists", "could lead to", "may help")):
        return ""
    if bad_sentence_issues(sentence):
        return ""
    return sentence


def _normalized_sentence_key(text: str) -> str:
    return " ".join(str(text or "").lower().replace("-", " ").split())


def _summary_sentence_keys(text: str) -> set[str]:
    return {_normalized_sentence_key(sentence) for sentence in split_sentences(text)}


def _extra_sentences_for_row(erow: dict[str, Any], existing_summary: str) -> list[str]:
    existing_lower = str(existing_summary or "").lower()
    out: list[str] = []
    for key, max_words in (
        ("lay_context", 15),
        ("expert_summary_hint", 14),
        ("abstract_claim", 16),
        ("core_keep_af", 16),
    ):
        candidate = _clean_extra_sentence(str(erow.get(key, "")), max_words=max_words)
        if not candidate:
            continue
        if candidate.lower() in existing_lower:
            continue
        out.append(candidate)
    for span in erow.get("allowed_evidence_spans") or []:
        candidate = _clean_extra_sentence(str(span), max_words=15)
        if not candidate:
            continue
        if candidate.lower() in existing_lower:
            continue
        out.append(candidate)
    deduped: list[str] = []
    seen: set[str] = set()
    for sentence in out:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sentence)
    return deduped


def _force_min_length_from_evidence(
    *,
    evidence: dict[str, Any],
    slots: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, list[str]]:
    repaired = [dict(slot) for slot in slots]
    notes: list[str] = []
    min_words = int(policy["min_word_count"])
    max_words = int(policy["max_word_count"])
    if safe_word_count(_summary_from_slots(repaired)) >= min_words:
        return repaired, _summary_from_slots(repaired), notes

    rows_by_id = _evidence_by_row_id(evidence)
    next_idx = len(repaired) + 1

    def current_summary() -> str:
        return _summary_from_slots(repaired)

    def sentence_pool(erow: dict[str, Any], summary_text: str) -> list[str]:
        pool: list[str] = []
        existing_keys = _summary_sentence_keys(summary_text)
        raw_items: list[str] = []
        for key in ("abstract_claim", "core_keep_af", "expert_summary_hint", "lay_context"):
            raw_items.append(str(erow.get(key, "")))
        raw_items.extend(str(x) for x in (erow.get("allowed_evidence_spans") or []))
        for raw in raw_items:
            for sentence in split_sentences(raw):
                cleaned = _clean_extra_sentence(sentence, max_words=20)
                if not cleaned:
                    continue
                key = _normalized_sentence_key(cleaned)
                if key in existing_keys:
                    continue
                if any(key and (key in old or old in key) for old in existing_keys if len(old) > 30):
                    continue
                pool.append(cleaned)
                existing_keys.add(key)
        return pool

    # First, extend slots that are already part of the summary.
    for slot in repaired:
        if safe_word_count(current_summary()) >= min_words:
            break
        if not slot.get("used_for_summary", True):
            continue
        erow = rows_by_id.get(str(slot.get("evidence_row_id")))
        if not erow:
            continue
        for sentence in sentence_pool(erow, current_summary()):
            old = str(slot.get("clarification_sentence", "")).strip()
            slot["clarification_sentence"] = f"{old} {sentence}".strip()
            wc = safe_word_count(current_summary())
            if wc <= max_words:
                notes.append(f"v10 floor added evidence sentence to {slot.get('slot_id')}; wc={wc}")
                break
            slot["clarification_sentence"] = old
        if safe_word_count(current_summary()) >= min_words:
            break

    # Then add unused central evidence rows if the existing slots are still short.
    used_rows = {str(slot.get("evidence_row_id")) for slot in repaired if slot.get("used_for_summary", True)}
    for erow in evidence.get("evidence_rows") or []:
        if safe_word_count(current_summary()) >= min_words:
            break
        row_id = str(erow.get("evidence_row_id"))
        if row_id in used_rows:
            continue
        role = str(erow.get("story_role") or "finding").strip().lower()
        if role not in {"method", "mechanism", "finding", "motivation"}:
            continue
        pool = sentence_pool(erow, current_summary())
        if not pool:
            continue
        candidate_slot = {
            "slot_id": f"slot_floor_{next_idx:04d}",
            "role": role if role in {"background", "motivation", "method", "mechanism", "finding", "implication", "limitation"} else "finding",
            "evidence_row_id": row_id,
            "claim_sentence": pool[0],
            "clarification_sentence": "",
            "used_for_summary": True,
            "banned_phrase_warnings": _contains_banned_phrase(pool[0]),
        }
        repaired.append(candidate_slot)
        wc = safe_word_count(current_summary())
        if wc <= max_words:
            notes.append(f"v10 floor added unused evidence row {row_id}; wc={wc}")
            used_rows.add(row_id)
            next_idx += 1
            continue
        repaired.pop()

    return repaired, current_summary(), notes


def _sentence_align_risk_score(sentence: str) -> int:
    text = str(sentence or "").strip()
    lower = text.lower()
    score = 0
    if lower.startswith(("this ", "these ", "it ", "they ", "this change", "this finding", "this means", "this shows")):
        score += 3
    if any(term in lower for term in ("means that", "shows that", "confirms", "ensures", "causes", "allows", "makes", "drives", "explains")):
        score += 2
    if any(term in lower for term in ("like ", "as if", "similar to", "magnet", "sponge", "machine")):
        score += 2
    if any(term in lower for term in ("most common", "currently too", "helps scientists", "important for", "better repair")):
        score += 2
    if safe_word_count(text) > 22:
        score += 1
    return score


def _deterministic_expand_slots(
    *,
    evidence: dict[str, Any],
    slots: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, list[str]]:
    expanded = [dict(slot) for slot in slots]
    notes: list[str] = []
    min_words = int(policy["min_word_count"])
    max_words = int(policy["max_word_count"])
    rows_by_id = _evidence_by_row_id(evidence)

    def summary() -> str:
        return _summary_from_slots(expanded)

    if safe_word_count(summary()) >= min_words:
        return expanded, summary(), notes

    for slot in expanded:
        if safe_word_count(summary()) >= min_words:
            break
        if str(slot.get("clarification_sentence", "")).strip():
            continue
        erow = rows_by_id.get(str(slot.get("evidence_row_id")))
        if not erow:
            continue
        for key in ("abstract_claim", "core_keep_af", "expert_summary_hint", "lay_context"):
            candidate = _first_sentence(str(erow.get(key, "")).strip(), max_words=12)
            if not candidate or candidate in summary():
                continue
            if key == "lay_context" and safe_word_count(candidate) > 10:
                continue
            old = slot.get("clarification_sentence", "")
            slot["clarification_sentence"] = candidate
            if safe_word_count(summary()) <= max_words:
                notes.append(f"filled clarification for {slot.get('slot_id')} from {key}")
                break
            slot["clarification_sentence"] = old

    used_rows = {str(slot.get("evidence_row_id")) for slot in expanded if slot.get("used_for_summary", True)}
    next_idx = len(expanded) + 1
    for erow in evidence.get("evidence_rows") or []:
        if safe_word_count(summary()) >= min_words:
            break
        row_id = str(erow.get("evidence_row_id"))
        if row_id in used_rows:
            continue
        role = str(erow.get("story_role") or "finding").strip().lower()
        if role not in {"method", "mechanism", "finding"}:
            continue
        claim = _first_sentence(str(erow.get("core_keep_af") or erow.get("abstract_claim") or "").strip(), max_words=14)
        if not claim:
            continue
        candidate_slot = {
            "slot_id": f"slot_auto_{next_idx:04d}",
            "role": role,
            "evidence_row_id": row_id,
            "claim_sentence": claim,
            "clarification_sentence": "",
            "used_for_summary": True,
            "banned_phrase_warnings": _contains_banned_phrase(claim),
        }
        expanded.append(candidate_slot)
        if safe_word_count(summary()) <= max_words:
            notes.append(f"added unused evidence row {row_id} for length")
            used_rows.add(row_id)
            next_idx += 1
            continue
        expanded.pop()

    # Last-resort length floor: add one short evidence-local sentence to existing
    # slots. This prevents invalid below-min finals without letting the model
    # invent broad background or unsupported implications.
    for slot in expanded:
        if safe_word_count(summary()) >= min_words:
            break
        if not slot.get("used_for_summary", True):
            continue
        erow = rows_by_id.get(str(slot.get("evidence_row_id")))
        if not erow:
            continue
        for candidate in _extra_sentences_for_row(erow, summary()):
            old = str(slot.get("clarification_sentence", "")).strip()
            if candidate in old or candidate in str(slot.get("claim_sentence", "")):
                continue
            slot["clarification_sentence"] = f"{old} {candidate}".strip()
            if safe_word_count(summary()) <= max_words:
                notes.append(f"added short evidence-local sentence to {slot.get('slot_id')} for length")
                break
            slot["clarification_sentence"] = old

    return expanded, summary(), notes


def _deterministic_trim_slots(
    *,
    slots: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, list[str]]:
    trimmed = [dict(slot) for slot in slots]
    notes: list[str] = []
    min_words = int(policy["min_word_count"])

    def summary() -> str:
        return _summary_from_slots(trimmed)

    if safe_word_count(summary()) <= min_words:
        return trimmed, summary(), notes

    role_drop_order = {
        "implication": 0,
        "limitation": 1,
        "background": 2,
        "motivation": 3,
        "method": 4,
        "mechanism": 5,
        "finding": 6,
    }
    candidates = sorted(
        [
            (
                -_sentence_align_risk_score(str(slot.get("clarification_sentence", ""))),
                role_drop_order.get(str(slot.get("role")), 4),
                -idx,
                slot,
            )
            for idx, slot in enumerate(trimmed)
            if str(slot.get("clarification_sentence", "")).strip()
        ],
        key=lambda item: (item[0], item[1], item[2]),
    )
    before_score = readability_proxy_score(summary())
    for _, _, _, slot in candidates:
        current_summary = summary()
        if safe_word_count(current_summary) <= min_words:
            break
        old = str(slot.get("clarification_sentence", "")).strip()
        slot["clarification_sentence"] = ""
        new_summary = summary()
        if safe_word_count(new_summary) < min_words:
            slot["clarification_sentence"] = old
            continue
        old_risk = _sentence_align_risk_score(old)
        new_score = readability_proxy_score(new_summary)
        if new_score <= before_score or old_risk >= 2:
            notes.append(f"removed non-essential/risky clarification from {slot.get('slot_id')}")
            before_score = readability_proxy_score(new_summary)
            continue
        slot["clarification_sentence"] = old

    return trimmed, summary(), notes


def _valid_candidate(text: str, policy: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    wc = safe_word_count(text)
    if not text.strip():
        reasons.append("empty")
    if find_mojibake(text):
        reasons.append("mojibake")
    if wc < int(policy["min_word_count"]):
        reasons.append("below_min")
    if wc > int(policy["max_word_count"]):
        reasons.append("above_max")
    stats = readability_stats(text)
    if int(stats["max_sentence_words"]) > 40:
        reasons.append("very_long_sentence")
    bad_issues = bad_sentence_issues(text)
    if bad_issues:
        reasons.extend(f"bad_sentence:{issue}" for issue in bad_issues[:5])
    return not reasons, reasons


def _candidate_by_variant(candidates: list[dict[str, Any]], variant: str) -> dict[str, Any] | None:
    return next((c for c in candidates if c.get("variant") == variant), None)


def _estimate_repair_wrong_count(wrong_count: int, edits: list[dict[str, Any]], parse_error: bool) -> int:
    if parse_error:
        return wrong_count
    if wrong_count <= 0:
        return 0
    material_edits = [
        edit
        for edit in edits
        if str(edit.get("op", "")).strip().lower() in {"replace", "delete", "insert_after"}
    ]
    return max(0, wrong_count - len(material_edits))


def _annotate_candidates(
    candidates: list[dict[str, Any]],
    *,
    wrong_count: int,
    repair_wrong_count_proxy: int | None,
    policy: dict[str, Any],
) -> None:
    for candidate in candidates:
        text = str(candidate.get("summary", ""))
        ok, reasons = _valid_candidate(text, policy)
        variant = str(candidate.get("variant", ""))
        candidate["valid"] = ok
        candidate["invalid_reasons"] = reasons
        candidate["readability_stats"] = readability_stats(text)
        candidate["readability_proxy_score"] = readability_proxy_score(text)
        candidate["word_count"] = safe_word_count(text)
        if variant == "factual_repair":
            candidate["wrong_count_proxy"] = int(repair_wrong_count_proxy if repair_wrong_count_proxy is not None else wrong_count)
        else:
            candidate["wrong_count_proxy"] = int(wrong_count)


def _repair_is_too_costly(repair: dict[str, Any], baseline: dict[str, Any], *, wrong_count: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    repair_wrong = int(repair.get("wrong_count_proxy") or 0)
    baseline_wrong = int(baseline.get("wrong_count_proxy") or wrong_count)
    wrong_drop = baseline_wrong - repair_wrong
    if wrong_count < 4:
        reasons.append("wrong_feedback_not_severe_enough_for_rewrite")
    if wrong_drop < 2:
        reasons.append("wrong_count_proxy_drop_too_small")
    if repair_wrong >= baseline_wrong:
        reasons.append("wrong_count_not_reduced")

    repair_score = float(repair.get("readability_proxy_score", 999.0))
    baseline_score = float(baseline.get("readability_proxy_score", 999.0))
    stats = repair.get("readability_stats") if isinstance(repair.get("readability_stats"), dict) else {}
    baseline_stats = baseline.get("readability_stats") if isinstance(baseline.get("readability_stats"), dict) else {}

    readability_slack = 0.0 if wrong_count < 4 else 0.75
    if repair_score > baseline_score + readability_slack:
        reasons.append("readability_proxy_worse")
    hard_word_slack = 0.0 if wrong_count < 4 else 0.01
    if float(stats.get("long_word_ratio", 0.0)) > float(baseline_stats.get("long_word_ratio", 0.0)) + hard_word_slack:
        reasons.append("long_word_density_worse")
    if float(stats.get("hard_word_ratio", 0.0)) > float(baseline_stats.get("hard_word_ratio", 0.0)) + hard_word_slack:
        reasons.append("hard_word_density_worse")
    if int(stats.get("high_risk_sentence_count", 0)) > int(baseline_stats.get("high_risk_sentence_count", 0)):
        reasons.append("more_high_risk_sentences")
    if int(stats.get("very_long_sentence_count", 0)) > int(baseline_stats.get("very_long_sentence_count", 0)):
        reasons.append("more_very_long_sentences")
    if int(stats.get("max_sentence_words", 0)) > int(baseline_stats.get("max_sentence_words", 0)) + 8:
        reasons.append("max_sentence_longer")

    return bool(reasons), reasons


def _is_safe_readability_gain(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not candidate.get("valid"):
        reasons.append("candidate_invalid")
    if int(candidate.get("wrong_count_proxy") or 0) > int(baseline.get("wrong_count_proxy") or 0):
        reasons.append("wrong_count_proxy_worse")
    cand_score = float(candidate.get("readability_proxy_score", 999.0))
    base_score = float(baseline.get("readability_proxy_score", 999.0))
    if cand_score > base_score - 0.35:
        reasons.append("readability_gain_too_small")
    cand_wc = int(candidate.get("word_count") or 0)
    base_wc = int(baseline.get("word_count") or 0)
    if cand_wc < int(base_wc * 0.9):
        reasons.append("too_much_content_removed")
    stats = candidate.get("readability_stats") if isinstance(candidate.get("readability_stats"), dict) else {}
    base_stats = baseline.get("readability_stats") if isinstance(baseline.get("readability_stats"), dict) else {}
    if float(stats.get("long_word_ratio", 0.0)) > float(base_stats.get("long_word_ratio", 0.0)):
        reasons.append("long_word_density_worse")
    if float(stats.get("hard_word_ratio", 0.0)) > float(base_stats.get("hard_word_ratio", 0.0)):
        reasons.append("hard_word_density_worse")
    if int(stats.get("high_risk_sentence_count", 0)) > int(base_stats.get("high_risk_sentence_count", 0)):
        reasons.append("more_high_risk_sentences")
    return not reasons, reasons


def _is_safe_expansion(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not candidate.get("valid"):
        reasons.append("candidate_invalid")
    if int(candidate.get("wrong_count_proxy") or 0) > int(baseline.get("wrong_count_proxy") or 0):
        reasons.append("wrong_count_proxy_worse")
    stats = candidate.get("readability_stats") if isinstance(candidate.get("readability_stats"), dict) else {}
    base_stats = baseline.get("readability_stats") if isinstance(baseline.get("readability_stats"), dict) else {}
    if float(stats.get("long_word_ratio", 0.0)) > float(base_stats.get("long_word_ratio", 0.0)) + 0.005:
        reasons.append("long_word_density_worse")
    if float(stats.get("hard_word_ratio", 0.0)) > float(base_stats.get("hard_word_ratio", 0.0)) + 0.005:
        reasons.append("hard_word_density_worse")
    if int(stats.get("long_sentence_count", 0)) > int(base_stats.get("long_sentence_count", 0)):
        reasons.append("more_long_sentences")
    if int(stats.get("high_risk_sentence_count", 0)) > int(base_stats.get("high_risk_sentence_count", 0)):
        reasons.append("more_high_risk_sentences")
    cand_score = float(candidate.get("readability_proxy_score", 999.0))
    base_score = float(baseline.get("readability_proxy_score", 999.0))
    if cand_score > base_score + 0.75:
        reasons.append("readability_proxy_much_worse")
    return not reasons, reasons


def _v10_rewrite_reject_reasons(rewrite: dict[str, Any], generated: dict[str, Any] | None, *, wrong_count: int) -> list[str]:
    reasons: list[str] = []
    if not rewrite.get("valid"):
        reasons.append("rewrite_invalid")
    if generated is None:
        return reasons
    if int(rewrite.get("wrong_count_proxy") or 0) > int(generated.get("wrong_count_proxy") or wrong_count):
        reasons.append("wrong_count_proxy_worse")
    rewrite_stats = rewrite.get("readability_stats") if isinstance(rewrite.get("readability_stats"), dict) else {}
    gen_stats = generated.get("readability_stats") if isinstance(generated.get("readability_stats"), dict) else {}
    if int(rewrite_stats.get("very_long_sentence_count", 0)) > int(gen_stats.get("very_long_sentence_count", 0)):
        reasons.append("more_very_long_sentences")
    if int(rewrite_stats.get("high_risk_sentence_count", 0)) > int(gen_stats.get("high_risk_sentence_count", 0)) + 1:
        reasons.append("too_many_more_high_risk_sentences")
    if float(rewrite_stats.get("hard_word_ratio", 0.0)) > float(gen_stats.get("hard_word_ratio", 0.0)) + 0.025:
        reasons.append("hard_word_density_much_worse")
    if float(rewrite_stats.get("long_word_ratio", 0.0)) > float(gen_stats.get("long_word_ratio", 0.0)) + 0.025:
        reasons.append("long_word_density_much_worse")

    rewrite_score = float(rewrite.get("readability_proxy_score", 999.0))
    gen_score = float(generated.get("readability_proxy_score", 999.0))
    rewrite_wc = int(rewrite.get("word_count") or 0)
    gen_wc = int(generated.get("word_count") or 0)
    relevance_proxy_gain = rewrite_wc >= gen_wc + 8
    readability_gain = rewrite_score <= gen_score - 0.25
    factual_repair_gain = int(rewrite.get("wrong_count_proxy") or 0) < int(generated.get("wrong_count_proxy") or wrong_count)
    validity_gain = "below_min" in generated.get("invalid_reasons", []) and "below_min" not in rewrite.get("invalid_reasons", [])
    rewrite["v10_internal_gain_flags"] = {
        "relevance_proxy_gain": relevance_proxy_gain,
        "readability_gain": readability_gain,
        "factual_repair_gain": factual_repair_gain,
        "validity_gain": validity_gain,
    }
    return reasons


def _select_final_candidate(
    candidates: list[dict[str, Any]],
    *,
    wrong_count: int,
    repair_wrong_count_proxy: int | None,
    policy: dict[str, Any],
) -> dict[str, Any]:
    _annotate_candidates(
        candidates,
        wrong_count=wrong_count,
        repair_wrong_count_proxy=repair_wrong_count_proxy,
        policy=policy,
    )

    valid = [c for c in candidates if c.get("valid")]
    if not valid:
        min_words = int(policy["min_word_count"])
        selected = min(
            candidates,
            key=lambda c: (
                "mojibake" in c.get("invalid_reasons", []),
                "above_max" in c.get("invalid_reasons", []),
                max(0, min_words - int(c.get("word_count") or 0)),
                len(c.get("invalid_reasons", [])),
                float(c.get("readability_proxy_score", 999.0)),
            ),
        )
        selected["selection_reason"] = "least_invalid_candidate_closest_to_min_length"
        return selected

    generated_any = _candidate_by_variant(candidates, "generated")
    generated = _candidate_by_variant(valid, "generated")
    expanded = _candidate_by_variant(valid, "expanded_generated")
    readability_trim = _candidate_by_variant(valid, "readability_trim")
    repair = _candidate_by_variant(valid, "factual_repair")

    if repair:
        rewrite_reject_reasons = _v10_rewrite_reject_reasons(repair, generated_any, wrong_count=wrong_count)
        repair["v10_rewrite_reject_reasons"] = rewrite_reject_reasons
        if not rewrite_reject_reasons:
            repair["selection_reason"] = "v10_1t3f_rewrite_passed_gain_gate"
            return repair

    if generated and wrong_count == 0:
        if readability_trim:
            ok, reject_reasons = _is_safe_readability_gain(readability_trim, generated)
            readability_trim["readability_trim_reject_reasons"] = reject_reasons
            if ok:
                readability_trim["selection_reason"] = "safe_readability_gain_without_factual_feedback"
                return readability_trim
        generated["selection_reason"] = "generated_valid_no_wrong_feedback"
        return generated

    if generated and "below_min" not in generated.get("invalid_reasons", []):
        baseline = generated
    else:
        baseline = generated_any or expanded or generated or min(valid, key=lambda c: len(c.get("invalid_reasons", [])))

    if repair:
        too_costly, repair_reject_reasons = _repair_is_too_costly(repair, baseline, wrong_count=wrong_count)
        repair["repair_reject_reasons"] = repair_reject_reasons
        if not too_costly:
            repair["selection_reason"] = "repair_reduced_wrong_proxy_without_readability_cost"
            return repair

    if generated and generated.get("valid") and "below_min" not in generated.get("invalid_reasons", []):
        generated["selection_reason"] = "kept_generated_conservative_factuality_gate"
        return generated

    if expanded and expanded.get("valid"):
        expansion_baseline = generated_any or baseline
        ok, reject_reasons = _is_safe_expansion(expanded, expansion_baseline)
        expanded["expanded_reject_reasons"] = reject_reasons
        if ok:
            expanded["selection_reason"] = "expanded_generated_fixed_length_or_validity"
            return expanded
        alternatives = [c for c in valid if c is not expanded]
        if alternatives:
            selected = min(
                alternatives,
                key=lambda c: (
                    int(c.get("wrong_count_proxy") or 0),
                    float(c.get("readability_proxy_score", 999.0)),
                    -int(c.get("word_count", 0)),
                ),
            )
            selected["selection_reason"] = "expanded_rejected_best_quality_alternative"
            return selected
        expanded["selection_reason"] = "expanded_generated_only_valid_candidate"
        return expanded

    selected = min(
        valid,
        key=lambda c: (
            int(c.get("wrong_count_proxy") or 0),
            float(c.get("readability_proxy_score", 999.0)),
            -int(c.get("word_count", 0)),
        ),
    )
    selected["selection_reason"] = "best_valid_readability_proxy"
    return selected


def rewrite_summaries(
    *,
    run_name: str,
    model_key: str,
    mode: str,
    max_workers: int,
    resume: bool,
    limit_articles: int | None = None,
) -> list[dict[str, Any]]:
    require_mode(mode)
    articles = load_articles(run_name)
    if limit_articles is not None:
        articles = articles[:limit_articles]
    evidence_by_article = _load_evidence_by_article(run_name, model_key, mode)
    summaries_path = _summaries_dir(run_name, model_key) / "generated_summaries.jsonl"
    wrong_path = _answers_dir(run_name, model_key, mode) / "wrong_answers.jsonl"
    summaries = {
        str(r["article_id"]): r
        for r in load_jsonl(summaries_path)
        if not r.get("parse_error") and str(r.get("generated_summary", "")).strip()
    }
    wrong_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if wrong_path.exists():
        for row in load_jsonl(wrong_path):
            wrong_by_article[str(row["article_id"])].append(row)

    prompt_path = PROMPTS_DIR / "rewrite_summary_v10.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")
    out_dir = _rewritten_dir(run_name, model_key)
    out_path = out_dir / "rewritten_summaries.jsonl"

    usage = usage_tracker(run_name, resume=resume)
    client = ProviderClient(model_key, usage)
    existing = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("article_id")) for r in existing}

    def worker(article: dict[str, Any]) -> dict[str, Any]:
        aid = str(article["id"])
        summary_row = summaries.get(aid)
        evidence = evidence_by_article.get(aid)
        if summary_row is None or evidence is None:
            raise RuntimeError(f"Missing summary/evidence for {aid}")
        current = str(summary_row["generated_summary"]).strip()
        current_slots = summary_row.get("slots") if isinstance(summary_row.get("slots"), list) else []
        policy = length_policy(
            mode=mode,
            expert_word_count=int(article.get("expert_summary_word_count") or 0),
            source_dataset=str(article.get("source_dataset", "")),
        )
        if current_slots:
            current_slots, skip_fit_notes = _fit_slots_to_length(current_slots, policy)
            current = _summary_from_slots(current_slots) or current
        else:
            skip_fit_notes = []
        wrong_rows = wrong_by_article.get(aid, [])
        original_generated = current
        current_slots, current, deterministic_notes = _deterministic_expand_slots(
            evidence=evidence,
            slots=current_slots,
            policy=policy,
        )
        current_slots, generated_fit_notes = _fit_slots_to_length(current_slots, policy)
        current = _summary_from_slots(current_slots) or current
        trim_slots, trim_summary, trim_notes = _deterministic_trim_slots(
            slots=current_slots,
            policy=policy,
        )
        needs_factual_repair = True
        needs_length_repair = safe_word_count(original_generated) < int(policy["min_word_count"])

        raw = ""
        parse_error = False
        repair_summary = current
        repair_slots = current_slots
        edits: list[dict[str, Any]] = []
        notes: list[str] = []
        expansion_attempts: list[str] = []
        expansion_errors: list[str] = []
        length_fit_notes: list[str] = []
        rewrite_invoked = True
        if needs_factual_repair:
            try:
                prompt = (
                    prompt_template.replace("{mode}", mode)
                    .replace("{dataset_profile}", dataset_profile(str(article.get("source_dataset", ""))))
                    .replace("{current_summary}", current)
                    .replace("{current_slots}", _format_slots(current_slots))
                    .replace("{evidence_table}", _format_evidence_table(evidence))
                    .replace("{feedback}", _feedback_for_article(wrong_rows))
                    .replace("{min_word_count}", str(policy["min_word_count"]))
                    .replace("{max_word_count}", str(policy["max_word_count"]))
                    .replace("{target_word_count}", str(policy["target_word_count"]))
                    .replace("{style_profile}", str(policy.get("style_profile", "compact_factual")))
                    .replace(
                        "{readability_policy}",
                        compact_json(
                            readability_policy(
                                source_dataset=str(article.get("source_dataset", "")),
                                expert_word_count=int(article.get("expert_summary_word_count") or 0),
                            )
                        ),
                    )
                    .replace("{current_readability_stats}", compact_json(readability_stats(current)))
                )
                raw = client.chat(
                    stage=f"V10.rewrite_summary.{mode}:{model_key}",
                    item_id=aid,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                obj = loads_json_object(raw)
                valid_row_ids = {str(r["evidence_row_id"]) for r in evidence.get("evidence_rows") or []}
                revised_slots, validation_errors = _normalize_slots(obj, valid_row_ids)
                if validation_errors:
                    raise ValueError("; ".join(validation_errors))
                repair_slots = _merge_slots(current_slots, revised_slots)
                repair_summary = str(obj.get("rewritten_summary", "")).strip() or _summary_from_slots(repair_slots)
                if not repair_summary:
                    raise ValueError("empty rewritten_summary")
                repair_slots, repair_summary, expansion_attempts, expand_errors = _expand_if_needed(
                    client=client,
                    stage_prefix=f"V10.rewrite_summary.{model_key}",
                    item_id=aid,
                    mode=mode,
                    source_dataset=str(article.get("source_dataset", "")),
                    policy=policy,
                    evidence=evidence,
                    slots=repair_slots,
                    summary=repair_summary,
                )
                expansion_errors.extend(expand_errors)
                repair_slots, repair_summary, repair_det_notes = _deterministic_expand_slots(
                    evidence=evidence,
                    slots=repair_slots,
                    policy=policy,
                )
                repair_slots, length_fit_notes = _fit_slots_to_length(repair_slots, policy)
                repair_summary = _summary_from_slots(repair_slots) or repair_summary
                length_fit_notes.extend(repair_det_notes)
                raw_edits = obj.get("edits", [])
                if isinstance(raw_edits, list):
                    edits = [e for e in raw_edits if isinstance(e, dict)]
                raw_notes = obj.get("revision_notes", [])
                if isinstance(raw_notes, list):
                    notes = [str(x).strip() for x in raw_notes if str(x).strip()]
                elif raw_notes:
                    notes = [str(raw_notes).strip()]
            except Exception as exc:
                parse_error = True
                notes = [f"parse_or_runtime_error: {exc}"]
                repair_summary = current
                repair_slots = current_slots

        repair_wrong_count_proxy = _estimate_repair_wrong_count(len(wrong_rows), edits, parse_error)
        candidates = [
            {
                "variant": "generated",
                "summary": original_generated,
                "slots": summary_row.get("slots") if isinstance(summary_row.get("slots"), list) else current_slots,
            },
            {
                "variant": "expanded_generated",
                "summary": current,
                "slots": current_slots,
                "expansion_needed": needs_length_repair,
            },
            {
                "variant": "readability_trim",
                "summary": trim_summary,
                "slots": trim_slots,
                "trim_notes": trim_notes,
            },
        ]
        candidates.append({"variant": "factual_repair", "summary": repair_summary, "slots": repair_slots})
        floor_notes: list[str] = []
        for candidate in candidates:
            if safe_word_count(str(candidate.get("summary", ""))) >= int(policy["min_word_count"]):
                continue
            candidate_slots = candidate.get("slots") if isinstance(candidate.get("slots"), list) else current_slots
            floor_slots, floor_summary, notes_for_candidate = _force_min_length_from_evidence(
                evidence=evidence,
                slots=candidate_slots,
                policy=policy,
            )
            if notes_for_candidate:
                candidate["slots"] = floor_slots
                candidate["summary"] = floor_summary
                floor_notes.extend(f"{candidate.get('variant')}: {note}" for note in notes_for_candidate)
        selected = _select_final_candidate(
            candidates,
            wrong_count=len(wrong_rows),
            repair_wrong_count_proxy=repair_wrong_count_proxy,
            policy=policy,
        )
        rewritten = str(selected.get("summary", "")).strip()
        rewritten_slots = selected.get("slots") if isinstance(selected.get("slots"), list) else current_slots

        wc = safe_word_count(rewritten)
        return {
            "article_id": aid,
            "source_dataset": article.get("source_dataset"),
            "original_index": article.get("original_index"),
            "model_key": model_key,
            "mode": mode,
            "prompt_version": f"V10_rewrite_{mode}",
            "prompt_file": str(prompt_path),
            "prompt_sha256": sha256_text(prompt_template),
            "dataset_profile": dataset_profile(str(article.get("source_dataset", ""))),
            "generated_summary": original_generated,
            "expanded_generated_summary": current,
            "rewritten_summary": rewritten,
            "slots": rewritten_slots,
            "sentences": _slots_to_sentences(rewritten_slots),
            "selected_variant": selected.get("variant"),
            "selection_reason": selected.get("selection_reason"),
            "candidate_summaries": [
                {
                    "variant": c.get("variant"),
                    "summary": c.get("summary"),
                    "word_count": c.get("word_count"),
                    "readability_stats": c.get("readability_stats"),
                    "valid": c.get("valid"),
                    "invalid_reasons": c.get("invalid_reasons"),
                    "wrong_count_proxy": c.get("wrong_count_proxy"),
                    "readability_proxy_score": c.get("readability_proxy_score"),
                    "repair_reject_reasons": c.get("repair_reject_reasons", []),
                    "v10_rewrite_reject_reasons": c.get("v10_rewrite_reject_reasons", []),
                    "v10_internal_gain_flags": c.get("v10_internal_gain_flags", {}),
                    "expanded_reject_reasons": c.get("expanded_reject_reasons", []),
                    "readability_trim_reject_reasons": c.get("readability_trim_reject_reasons", []),
                    "trim_notes": c.get("trim_notes", []),
                }
                for c in candidates
            ],
            "rewrite_invoked": rewrite_invoked,
            "rewrite_skipped_no_errors": not rewrite_invoked,
            "wrong_feedback_count": len(wrong_rows),
            "repair_wrong_count_proxy": repair_wrong_count_proxy,
            "feedback_items": wrong_rows,
            "edits": edits,
            "revision_notes": notes + skip_fit_notes + deterministic_notes + generated_fit_notes + trim_notes + floor_notes,
            "expert_summary_word_count": int(article.get("expert_summary_word_count") or 0),
            "generated_word_count": int(summary_row.get("generated_word_count") or safe_word_count(original_generated)),
            "expanded_generated_word_count": safe_word_count(current),
            "rewritten_word_count": wc,
            "generated_readability_stats": readability_stats(original_generated),
            "expanded_generated_readability_stats": readability_stats(current),
            "rewritten_readability_stats": readability_stats(rewritten),
            "generated_readability_proxy_score": readability_proxy_score(original_generated),
            "expanded_generated_readability_proxy_score": readability_proxy_score(current),
            "rewritten_readability_proxy_score": readability_proxy_score(rewritten),
            "readability_proxy_delta": round(readability_proxy_score(rewritten) - readability_proxy_score(original_generated), 4),
            "mojibake_warnings": find_mojibake(rewritten),
            "min_word_count": policy["min_word_count"],
            "max_word_count": policy["max_word_count"],
            "below_min_length": wc < policy["min_word_count"],
            "above_max_length": wc > policy["max_word_count"],
            "parse_error": parse_error,
            "length_expansion_attempts": expansion_attempts,
            "length_expansion_errors": expansion_errors,
            "length_fit_notes": length_fit_notes,
            "raw_response": raw if parse_error else None,
        }

    todo = [a for a in articles if str(a["id"]) not in done]
    new_rows = run_parallel(todo, worker, max_workers=max_workers, desc=f"V10 rewrite | {mode} | {model_key}")
    all_rows = existing + new_rows
    all_rows.sort(key=lambda r: str(r["article_id"]))
    save_jsonl(all_rows, out_path)
    save_usage(run_name, usage)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "stage": "V10_06_rewritten",
            "mode": mode,
            "model_key": model_key,
            "rewritten_total": len(all_rows),
            "skipped_no_errors_count": sum(1 for r in all_rows if r.get("rewrite_skipped_no_errors")),
            "rewrite_invoked_count": sum(1 for r in all_rows if r.get("rewrite_invoked")),
            "selected_variant_counts": dict(Counter(str(r.get("selected_variant")) for r in all_rows)),
            "parse_error_count": sum(1 for r in all_rows if r.get("parse_error")),
            "below_min_length_count": sum(1 for r in all_rows if r.get("below_min_length")),
            "above_max_length_count": sum(1 for r in all_rows if r.get("above_max_length")),
            "usage_summary": usage.summarize(),
        },
        out_dir / "rewritten_summaries_metadata.json",
    )
    print(f"[V10 rewrite] {mode} {model_key} rewritten={len(all_rows)} -> {out_path}")
    return all_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V10 Phase B: generate, answer 1T3F, and rewrite.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--mode", choices=["balanced", "factuality_chase"], required=True)
    parser.add_argument("--step", choices=["generate-summary", "answer-questions", "rewrite-summaries"], required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit-articles", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    kwargs = {
        "run_name": args.run_name,
        "model_key": args.model_key,
        "mode": args.mode,
        "max_workers": args.max_workers,
        "resume": not args.no_resume,
        "limit_articles": args.limit_articles,
    }
    if args.step == "generate-summary":
        generate_summaries(**kwargs)
    elif args.step == "answer-questions":
        answer_questions(**kwargs)
    elif args.step == "rewrite-summaries":
        rewrite_summaries(**kwargs)
    else:
        raise ValueError(args.step)


if __name__ == "__main__":
    main()

