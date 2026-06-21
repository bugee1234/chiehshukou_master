from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.thesis_laysumm.paths import run_data_dir, run_results_dir
from src.thesis_laysumm_v11.common import bad_sentence_issues, find_mojibake, length_policy, readability_stats, safe_word_count
from src.utils import load_json, load_jsonl


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def _article_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("id") or row.get("article_id")) for row in rows}


def _summary_from_slots(slots: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for slot in slots:
        if not slot.get("used_for_summary", True):
            continue
        claim = str(slot.get("claim_sentence", "")).strip()
        clarification = str(slot.get("clarification_sentence", "")).strip()
        if claim:
            parts.append(claim)
        if clarification:
            parts.append(clarification)
    return " ".join(parts).strip()


def _norm(text: str) -> str:
    return " ".join(str(text or "").split())


def _validate_summary_rows(
    *,
    label: str,
    rows: list[dict[str, Any]],
    text_key: str,
    articles_by_id: dict[str, dict[str, Any]],
    mode: str,
) -> None:
    if _article_ids(rows) != set(articles_by_id):
        raise ValueError(f"{label} article IDs do not match articles.jsonl")
    for row in rows:
        aid = str(row["article_id"])
        if row.get("parse_error"):
            raise ValueError(f"{label} parse_error: {aid}")
        text = str(row.get(text_key, "")).strip()
        if not text:
            raise ValueError(f"{label} summary is empty: {aid}")
        mojibake = find_mojibake(text)
        if mojibake:
            raise ValueError(f"{label} summary has mojibake-like text: {aid} hits={mojibake}")
        bad_sentences = bad_sentence_issues(text)
        if bad_sentences and label != "generated":
            raise ValueError(f"{label} summary has bad sentence issues: {aid} hits={bad_sentences}")
        stats = readability_stats(text)
        if int(stats["max_sentence_words"]) > 65:
            raise ValueError(f"{label} summary has a very long sentence: {aid} stats={stats}")
        slots = row.get("slots")
        if not isinstance(slots, list) or not slots:
            raise ValueError(f"{label} slots are missing/empty: {aid}")
        for slot in slots:
            if not isinstance(slot, dict):
                raise ValueError(f"{label} slot is not an object: {aid}")
            if not slot.get("evidence_row_id"):
                raise ValueError(f"{label} slot missing evidence_row_id: {aid}")
            if slot.get("used_for_summary", True) and not slot.get("claim_sentence") and not slot.get("clarification_sentence"):
                raise ValueError(f"{label} used slot missing claim_sentence or clarification_sentence: {aid}")
        slot_summary = _summary_from_slots(slots)
        if slot_summary and _norm(slot_summary) != _norm(text):
            raise ValueError(f"{label} summary does not match joined slots: {aid}")
        policy = length_policy(
            mode=mode,
            expert_word_count=int(articles_by_id[aid].get("expert_summary_word_count") or 0),
            source_dataset=str(articles_by_id[aid].get("source_dataset", "")),
        )
        wc = safe_word_count(text)
        if label == "rewritten":
            if wc < int(policy["min_word_count"]):
                raise ValueError(
                    f"{label} summary below minimum length: {aid} wc={wc} min={policy['min_word_count']}"
                )
            if wc > int(policy["max_word_count"]):
                raise ValueError(
                    f"{label} summary above maximum length: {aid} wc={wc} max={policy['max_word_count']}"
                )
            if not row.get("selected_variant"):
                raise ValueError(f"{label} missing selected_variant: {aid}")
            selected = str(row.get("selected_variant"))
            candidates = row.get("candidate_summaries")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"{label} missing candidate_summaries: {aid}")
            selected_candidates = [c for c in candidates if str(c.get("variant")) == selected]
            if selected_candidates and selected_candidates[0].get("valid") is False:
                hard_reasons = [
                    reason
                    for reason in (selected_candidates[0].get("invalid_reasons") or [])
                    if reason not in {"too_many_high_risk_sentences"}
                ]
                if not hard_reasons:
                    continue
                raise ValueError(
                    f"{label} selected invalid candidate: {aid} reasons={selected_candidates[0].get('invalid_reasons')}"
                )


def validate_run(*, run_name: str, model_key: str, mode: str, stage: str, require_eval: bool) -> None:
    data_dir = run_data_dir(run_name)
    results_dir = run_results_dir(run_name) / model_key

    articles = load_jsonl(_require(data_dir / "00_articles" / "articles.jsonl"))
    if not articles:
        raise ValueError("articles.jsonl is empty")
    articles_by_id = {str(row["id"]): row for row in articles}

    evidence_rows = load_jsonl(
        _require(data_dir / "02_5_evidence_table" / mode / model_key / "evidence_table.jsonl")
    )
    if _article_ids(evidence_rows) != set(articles_by_id):
        raise ValueError("Evidence article IDs do not match articles.jsonl")
    for row in evidence_rows:
        if row.get("parse_error"):
            raise ValueError(f"Evidence parse_error: {row.get('article_id')}")
        inner = row.get("evidence_rows") or []
        if not inner:
            raise ValueError(f"No evidence rows for {row.get('article_id')}")
        for erow in inner:
            for key in ("evidence_row_id", "core_keep_af", "allowed_evidence_spans", "story_role"):
                if not erow.get(key):
                    raise ValueError(f"Evidence row missing {key}: {row.get('article_id')} {erow}")
            if "lay_context" not in erow:
                raise ValueError(f"Evidence row missing lay_context key: {row.get('article_id')} {erow}")
    if stage == "evidence":
        print(f"[V11 validate] ok evidence run={run_name} model={model_key} articles={len(articles)}")
        return

    questions = load_jsonl(_require(data_dir / "03_questions_v11" / mode / model_key / "questions_1t3f_nota.jsonl"))
    if not questions:
        raise ValueError("questions_1t3f_nota.jsonl is empty")
    question_ids = {str(row.get("question_id")) for row in questions}
    if len(question_ids) != len(questions):
        raise ValueError("Duplicate question_id values found")
    if stage == "questions":
        print(f"[V11 validate] ok questions run={run_name} model={model_key} questions={len(questions)}")
        return

    generated = load_jsonl(_require(data_dir / "04_summaries" / model_key / "generated_summaries.jsonl"))
    _validate_summary_rows(
        label="generated",
        rows=generated,
        text_key="generated_summary",
        articles_by_id=articles_by_id,
        mode=mode,
    )
    if stage == "summaries":
        print(f"[V11 validate] ok generated summaries run={run_name} model={model_key} articles={len(generated)}")
        return

    answers = load_jsonl(_require(data_dir / "05_module3_answers_v11" / mode / model_key / "module3_answers.jsonl"))
    if len(answers) != len(questions):
        raise ValueError(f"Answer/question count mismatch: answers={len(answers)} questions={len(questions)}")
    if any(row.get("parse_error") for row in answers):
        bad = [row.get("question_id") for row in answers if row.get("parse_error")]
        raise ValueError(f"Answer parse_error rows: {bad[:10]}")
    if stage == "answers":
        print(f"[V11 validate] ok answers run={run_name} model={model_key} answers={len(answers)}")
        return

    rewritten = load_jsonl(_require(data_dir / "06_rewritten" / model_key / "rewritten_summaries.jsonl"))
    _validate_summary_rows(
        label="rewritten",
        rows=rewritten,
        text_key="rewritten_summary",
        articles_by_id=articles_by_id,
        mode=mode,
    )
    if stage == "rewritten":
        improved = 0
        comparable = 0
        below_min = 0
        for row in rewritten:
            if row.get("below_min_length"):
                below_min += 1
            candidates = row.get("candidate_summaries")
            if not isinstance(candidates, list) or len(candidates) < 2:
                raise ValueError(f"rewritten candidate_summaries missing/too small: {row.get('article_id')}")
            variants = {str(c.get("variant")) for c in candidates if isinstance(c, dict)}
            if "generated" not in variants or "expanded_generated" not in variants:
                raise ValueError(f"rewritten candidates missing generated/expanded_generated: {row.get('article_id')}")
            if "factual_repair" not in variants:
                raise ValueError(f"rewritten candidates missing mandatory V11 factual_repair: {row.get('article_id')}")
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    raise ValueError(f"rewritten candidate is not an object: {row.get('article_id')}")
                for key in ("summary", "word_count", "readability_stats", "invalid_reasons", "wrong_count_proxy"):
                    if key not in candidate:
                        raise ValueError(f"rewritten candidate missing {key}: {row.get('article_id')}")
            if not row.get("rewrite_invoked") and not row.get("rewrite_skip_reasons"):
                raise ValueError(f"rewritten skipped rewrite without skip reasons in V11: {row.get('article_id')}")
            if not row.get("selected_variant") or not row.get("selection_reason"):
                raise ValueError(f"rewritten selection metadata missing: {row.get('article_id')}")
            before = row.get("generated_readability_proxy_score")
            after = row.get("rewritten_readability_proxy_score")
            if before is None or after is None:
                continue
            comparable += 1
            if float(after) <= float(before):
                improved += 1
        print(
            f"[V11 validate] ok rewritten run={run_name} model={model_key} "
            f"articles={len(rewritten)} below_min={below_min} "
            f"readability_proxy_improved={improved}/{comparable}"
        )
        return

    if require_eval or stage == "eval":
        for name in ("generated_scores.json", "rewritten_scores.json", "leaderboard_comparison.json", "official_style_rank.json"):
            _require(results_dir / name)
        generated_scores = load_json(results_dir / "generated_scores.json")
        rewritten_scores = load_json(results_dir / "rewritten_scores.json")
        for label, payload in (("generated", generated_scores), ("rewritten", rewritten_scores)):
            if int(payload.get("article_count") or 0) != len(articles):
                raise ValueError(f"{label} evaluated article_count does not match articles")
            for metric in ("ROUGE", "BLEU", "METEOR", "BERTScore", "FKGL", "DCRS", "CLI", "LENS", "AlignScore", "SummaC"):
                if metric not in (payload.get("overall") or {}):
                    raise ValueError(f"{label} scores missing metric {metric}")
    if stage == "eval":
        print(f"[V11 validate] ok eval run={run_name} model={model_key}")
        return

    print(
        f"[V11 validate] ok run={run_name} model={model_key} "
        f"articles={len(articles)} evidence_articles={len(evidence_rows)} questions={len(questions)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate V11 lay-summary pipeline outputs.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--mode", choices=["balanced", "factuality_chase"], default="factuality_chase")
    parser.add_argument(
        "--stage",
        choices=["all", "evidence", "questions", "summaries", "answers", "rewritten", "eval"],
        default="all",
    )
    parser.add_argument("--require-eval", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_run(
        run_name=args.run_name,
        model_key=args.model_key,
        mode=args.mode,
        stage=args.stage,
        require_eval=args.require_eval,
    )


if __name__ == "__main__":
    main()


