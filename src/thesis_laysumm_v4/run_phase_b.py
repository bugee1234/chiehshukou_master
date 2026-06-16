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
from src.thesis_laysumm_v4.common import (
    PROMPTS_DIR,
    compact_json,
    length_policy,
    loads_json_object,
    require_mode,
    safe_word_count,
    split_sentences,
)
from src.utils import load_jsonl, save_json, save_jsonl

PROMPTS_DIR_V2 = ROOT_DIR / "src" / "thesis_laysumm" / "prompts"
DEFAULT_MODEL_KEY = "gpt41_mini"


def _evidence_dir(run_name: str, model_key: str, mode: str) -> Path:
    return run_data_dir(run_name) / "02_5_evidence_table" / mode / model_key


def _questions_dir(run_name: str, model_key: str, mode: str) -> Path:
    return run_data_dir(run_name) / "03_questions_v4" / mode / model_key


def _summaries_dir(run_name: str, model_key: str) -> Path:
    path = run_data_dir(run_name) / "04_summaries" / model_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _answers_dir(run_name: str, model_key: str, mode: str) -> Path:
    path = run_data_dir(run_name) / "05_module3_answers_v4" / mode / model_key
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

    prompt_path = PROMPTS_DIR / "generate_summary_v4.txt"
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
        policy = length_policy(mode=mode, expert_word_count=int(article.get("expert_summary_word_count") or 0))
        valid_row_ids = {str(r["evidence_row_id"]) for r in evidence.get("evidence_rows") or []}

        raw = ""
        parse_error = False
        validation_errors: list[str] = []
        sentences: list[dict[str, Any]] = []
        generated_summary = ""
        try:
            prompt = (
                prompt_template.replace("{mode}", mode)
                .replace("{title}", str(article.get("title", "")))
                .replace("{abstract}", str(article.get("abstract", "")))
                .replace("{evidence_table}", _format_evidence_table(evidence))
                .replace("{target_word_count}", str(policy["target_word_count"]))
                .replace("{min_word_count}", str(policy["min_word_count"]))
                .replace("{max_word_count}", str(policy["max_word_count"]))
                .replace("{style_profile}", str(policy.get("style_profile", "compact_factual")))
            )
            raw = client.chat(
                stage=f"v4.generate_summary.{mode}:{model_key}",
                item_id=aid,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = loads_json_object(raw)
            raw_sentences = obj.get("sentences", [])
            if not isinstance(raw_sentences, list) or not raw_sentences:
                validation_errors.append("sentences must be a non-empty list")
            for entry in raw_sentences if isinstance(raw_sentences, list) else []:
                if not isinstance(entry, dict):
                    continue
                text = str(entry.get("text", "")).strip()
                row_id = str(entry.get("evidence_row_id", "")).strip()
                if not text:
                    continue
                if row_id not in valid_row_ids:
                    validation_errors.append(f"unknown evidence_row_id: {row_id}")
                    continue
                sentences.append(
                    {
                        "text": text,
                        "evidence_row_id": row_id,
                        "section": str(entry.get("section", "results")).strip() or "results",
                    }
                )
            generated_summary = str(obj.get("summary", "")).strip()
            if not generated_summary:
                generated_summary = " ".join(s["text"] for s in sentences)
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
            "prompt_version": f"v4_generate_{mode}",
            "prompt_file": str(prompt_path),
            "prompt_sha256": sha256_text(prompt_template),
            "length_policy": policy,
            "expert_summary_word_count": int(article.get("expert_summary_word_count") or 0),
            "target_word_count": policy["target_word_count"],
            "min_word_count": policy["min_word_count"],
            "max_word_count": policy["max_word_count"],
            "sentences": sentences,
            "generated_summary": generated_summary,
            "generated_word_count": wc,
            "below_min_length": bool(generated_summary) and wc < policy["min_word_count"],
            "above_max_length": bool(generated_summary) and wc > policy["max_word_count"],
            "parse_error": parse_error,
            "validation_errors": validation_errors,
            "raw_response": raw if parse_error else None,
        }

    todo = [a for a in articles if str(a["id"]) not in done]
    new_rows = run_parallel(todo, worker, max_workers=max_workers, desc=f"v4 generate | {mode} | {model_key}")
    all_rows = existing + new_rows
    all_rows.sort(key=lambda r: str(r["article_id"]))
    save_jsonl(all_rows, out_path)
    save_usage(run_name, usage)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "stage": "v4_04_summaries",
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
    print(f"[v4 generate] {mode} {model_key} summaries={len(all_rows)} -> {out_path}")
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
        raise FileNotFoundError(f"Missing v4 questions: {questions_path}")
    if not summaries_path.exists():
        raise FileNotFoundError(f"Missing v4 generated summaries: {summaries_path}")

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
                stage=f"v4.answer_questions.{mode}:{model_key}",
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
            "context_mode": "v4_full_generated_summary",
        }

    todo = [q for q in questions if str(q.get("question_id")) not in done]
    new_rows = run_parallel(todo, worker, max_workers=max_workers, desc=f"v4 answer | {mode} | {model_key}")
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
            "stage": "v4_05_answers",
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
    print(f"[v4 answer] {mode} {model_key} answers={len(all_rows)} wrong={len(wrong_rows)}")
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

    prompt_path = PROMPTS_DIR / "rewrite_summary_v4.txt"
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
        policy = length_policy(mode=mode, expert_word_count=int(article.get("expert_summary_word_count") or 0))
        wrong_rows = wrong_by_article.get(aid, [])
        current_wc = safe_word_count(current)
        if not wrong_rows and policy["min_word_count"] <= current_wc <= policy["max_word_count"]:
            return {
                "article_id": aid,
                "source_dataset": article.get("source_dataset"),
                "original_index": article.get("original_index"),
                "model_key": model_key,
                "mode": mode,
                "prompt_version": f"v4_rewrite_{mode}",
                "prompt_file": str(prompt_path),
                "prompt_sha256": sha256_text(prompt_template),
                "generated_summary": current,
                "rewritten_summary": current,
                "rewrite_skipped_no_errors": True,
                "wrong_feedback_count": 0,
                "feedback_items": [],
                "edits": [],
                "revision_notes": ["no failed 1T3F checks; copied generated summary"],
                "expert_summary_word_count": int(article.get("expert_summary_word_count") or 0),
                "generated_word_count": int(summary_row.get("generated_word_count") or safe_word_count(current)),
                "rewritten_word_count": current_wc,
                "min_word_count": policy["min_word_count"],
                "max_word_count": policy["max_word_count"],
                "below_min_length": False,
                "above_max_length": False,
                "parse_error": False,
                "raw_response": None,
            }

        raw = ""
        parse_error = False
        rewritten = current
        edits: list[dict[str, Any]] = []
        notes: list[str] = []
        try:
            prompt = (
                prompt_template.replace("{mode}", mode)
                .replace("{current_summary}", current)
                .replace("{evidence_table}", _format_evidence_table(evidence))
                .replace("{feedback}", _feedback_for_article(wrong_rows))
                .replace("{min_word_count}", str(policy["min_word_count"]))
                .replace("{max_word_count}", str(policy["max_word_count"]))
                .replace("{target_word_count}", str(policy["target_word_count"]))
                .replace("{style_profile}", str(policy.get("style_profile", "compact_factual")))
            )
            raw = client.chat(
                stage=f"v4.rewrite_summary.{mode}:{model_key}",
                item_id=aid,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = loads_json_object(raw)
            rewritten = str(obj.get("rewritten_summary", "")).strip()
            if not rewritten:
                raise ValueError("empty rewritten_summary")
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
            rewritten = current

        wc = safe_word_count(rewritten)
        return {
            "article_id": aid,
            "source_dataset": article.get("source_dataset"),
            "original_index": article.get("original_index"),
            "model_key": model_key,
            "mode": mode,
            "prompt_version": f"v4_rewrite_{mode}",
            "prompt_file": str(prompt_path),
            "prompt_sha256": sha256_text(prompt_template),
            "generated_summary": current,
            "rewritten_summary": rewritten,
            "rewrite_skipped_no_errors": False,
            "wrong_feedback_count": len(wrong_rows),
            "feedback_items": wrong_rows,
            "edits": edits,
            "revision_notes": notes,
            "expert_summary_word_count": int(article.get("expert_summary_word_count") or 0),
            "generated_word_count": int(summary_row.get("generated_word_count") or safe_word_count(current)),
            "rewritten_word_count": wc,
            "min_word_count": policy["min_word_count"],
            "max_word_count": policy["max_word_count"],
            "below_min_length": wc < policy["min_word_count"],
            "above_max_length": wc > policy["max_word_count"],
            "parse_error": parse_error,
            "raw_response": raw if parse_error else None,
        }

    todo = [a for a in articles if str(a["id"]) not in done]
    new_rows = run_parallel(todo, worker, max_workers=max_workers, desc=f"v4 rewrite | {mode} | {model_key}")
    all_rows = existing + new_rows
    all_rows.sort(key=lambda r: str(r["article_id"]))
    save_jsonl(all_rows, out_path)
    save_usage(run_name, usage)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "stage": "v4_06_rewritten",
            "mode": mode,
            "model_key": model_key,
            "rewritten_total": len(all_rows),
            "skipped_no_errors_count": sum(1 for r in all_rows if r.get("rewrite_skipped_no_errors")),
            "parse_error_count": sum(1 for r in all_rows if r.get("parse_error")),
            "below_min_length_count": sum(1 for r in all_rows if r.get("below_min_length")),
            "above_max_length_count": sum(1 for r in all_rows if r.get("above_max_length")),
            "usage_summary": usage.summarize(),
        },
        out_dir / "rewritten_summaries_metadata.json",
    )
    print(f"[v4 rewrite] {mode} {model_key} rewritten={len(all_rows)} -> {out_path}")
    return all_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V4 Phase B: generate, answer 1T3F, and rewrite.")
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
