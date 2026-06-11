from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.utils import OpenAIClient, load_jsonl, save_json, save_jsonl


EXP3_DATA_DIR = ROOT_DIR / "data" / "experiment_3"
EXP3_RESULTS_DIR = ROOT_DIR / "results" / "experiment_3"
PROMPTS_EXP2_DIR = ROOT_DIR / "src" / "experiment_2" / "prompts"
PROMPTS_EXP3_DIR = ROOT_DIR / "src" / "experiment_3" / "prompts"
SETUP_C_PROMPT = ROOT_DIR / "src" / "experiment_1" / "prompts_1a_v2_fair_ac" / "setup_c_system.txt"

DATASETS = {
    "PLOS": ("BioLaySumm/BioLaySumm2025-PLOS", "plos"),
    "eLife": ("BioLaySumm/BioLaySumm2025-eLife", "elife"),
}

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TOP_K = 5
DEFAULT_SUMMARY_CHUNK_WORDS = 120
DEFAULT_SUMMARY_OVERLAP_WORDS = 30
DEFAULT_AF_CHUNK_WORDS = 1200
DEFAULT_AF_OVERLAP_WORDS = 120
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def _run_dir(run_name: str) -> Path:
    return EXP3_DATA_DIR / "runs" / run_name


def _stage_dir(run_name: str, stage: str) -> Path:
    d = _run_dir(run_name) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _word_count(text: str | None) -> int:
    return len(str(text or "").split())


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v or "").strip().lower() in {"1", "true", "yes", "y"}


def _chunk_words(text: str, chunk_words: int, overlap_words: int = 0) -> list[dict[str, Any]]:
    words = re.findall(r"\S+", str(text or ""))
    step = max(1, chunk_words - max(0, overlap_words))
    chunks: list[dict[str, Any]] = []
    for i in range(0, len(words), step):
        seg = words[i : i + chunk_words]
        if not seg:
            continue
        chunks.append(
            {
                "chunk_idx": len(chunks),
                "start_word": i,
                "end_word": i + len(seg),
                "chunk_text": " ".join(seg),
            }
        )
    if not chunks and str(text or "").strip():
        chunks.append({"chunk_idx": 0, "start_word": 0, "end_word": 0, "chunk_text": str(text).strip()})
    return chunks


def _mutate_sentence(sentence: str) -> str:
    t = str(sentence or "").strip()
    if not t:
        return "This statement is not supported."
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", t)
    if m:
        v = float(m.group(1))
        new_v = v + 5 if v < 90 else v - 5
        rep = str(int(new_v)) if float(new_v).is_integer() else str(round(new_v, 1))
        return t[: m.start(1)] + rep + t[m.end(1) :]
    rules = [
        (r"\bincrease(d|s)?\b", "decreased"),
        (r"\bdecrease(d|s)?\b", "increased"),
        (r"\bhigher\b", "lower"),
        (r"\blower\b", "higher"),
        (r"\bmore\b", "less"),
        (r"\bless\b", "more"),
        (r"\bcan\b", "cannot"),
        (r"\bcannot\b", "can"),
    ]
    for pat, rep in rules:
        if re.search(pat, t, flags=re.IGNORECASE):
            return re.sub(pat, rep, t, count=1, flags=re.IGNORECASE)
    return "It is not true that " + t[:1].lower() + t[1:]


def _parse_option_eval(obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {
        k: {"status": "", "negation_flag": False, "evidence": "", "strictness_mismatch": False}
        for k in ["A", "B", "C", "D"]
    }
    raw = obj.get("option_evaluation", {})
    if not isinstance(raw, dict):
        return out
    for letter in ["A", "B", "C", "D"]:
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
    for letter in ["A", "B", "C", "D"]:
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


def _embed_texts(client: OpenAI, texts: list[str], model: str, batch_size: int = 64) -> list[list[float]]:
    embs: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        embs.extend([d.embedding for d in resp.data])
    return embs


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / nb / na


def _load_articles(run_name: str) -> list[dict[str, Any]]:
    return load_jsonl(_stage_dir(run_name, "00_inputs") / "articles.jsonl")


def stage00_prepare_inputs(
    run_name: str,
    split: str,
    n_per_source: int | None,
    seed: int,
) -> None:
    out_dir = _stage_dir(run_name, "00_inputs")
    out_articles = out_dir / "articles.jsonl"
    out_truth_elife = out_dir / "eLife_test.jsonl"
    out_truth_plos = out_dir / "PLOS_test.jsonl"

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    truth_by_source: dict[str, list[dict[str, Any]]] = {"PLOS": [], "eLife": []}
    meta: dict[str, Any] = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "split": split,
        "seed": seed,
        "n_per_source": n_per_source,
        "datasets": {},
        "paper_alignment": "BioLaySumm 2025 Task 1.1 test split: PLOS 142 and eLife 142 when n_per_source is omitted.",
    }

    for source, (dataset_name, prefix) in DATASETS.items():
        ds = load_dataset(dataset_name, split=split, token=config.HF_TOKEN or None)
        if n_per_source is None and split == "test" and len(ds) != 142:
            raise RuntimeError(
                f"{source}: expected BioLaySumm 2025 Task 1.1 test split size 142, got {len(ds)}. "
                "Check dataset version/split before running the full comparison."
            )
        cols = set(ds.column_names)
        article_col = "article" if "article" in cols else ("document" if "document" in cols else None)
        lay_col = "lay_summary" if "lay_summary" in cols else ("summary" if "summary" in cols else ("reference" if "reference" in cols else None))
        abstract_col = "abstract" if "abstract" in cols else None
        if article_col is None or lay_col is None:
            raise RuntimeError(f"{source}: cannot identify article/reference columns: {ds.column_names}")

        indices = list(range(len(ds)))
        if n_per_source is not None:
            if len(indices) < n_per_source:
                raise RuntimeError(f"{source}: split has {len(indices)} rows, need {n_per_source}")
            indices = sorted(rng.sample(indices, n_per_source))

        meta["datasets"][source] = {
            "dataset_name": dataset_name,
            "split_size": len(ds),
            "selected_count": len(indices),
            "selected_indices": indices,
            "columns": ds.column_names,
        }
        for j, idx in enumerate(indices, start=1):
            item = ds[int(idx)]
            article = str(item.get(article_col, ""))
            reference = str(item.get(lay_col, ""))
            abstract = str(item.get(abstract_col, "")) if abstract_col else ""
            document = f"{abstract}\n{article}".strip() if abstract else article
            article_id = f"exp3_{prefix}_{idx:04d}"
            row = {
                "id": article_id,
                "source_dataset": source,
                "dataset_name": dataset_name,
                "split": split,
                "original_index": int(idx),
                "article": article,
                "abstract": abstract,
                "document": document,
                "reference": reference,
                "article_word_count": _word_count(article),
                "reference_word_count": _word_count(reference),
                "run_order": j,
            }
            rows.append(row)
            truth_by_source[source].append({"document": document, "reference": reference})

    save_jsonl(rows, out_articles)
    save_jsonl(truth_by_source["eLife"], out_truth_elife)
    save_jsonl(truth_by_source["PLOS"], out_truth_plos)
    meta["total_articles"] = len(rows)
    meta["by_source"] = dict(Counter(r["source_dataset"] for r in rows))
    meta["empty_reference_count"] = sum(1 for r in rows if not str(r.get("reference", "")).strip())
    if meta["empty_reference_count"]:
        meta["reference_note"] = (
            "The HuggingFace test split can contain empty summary/reference fields. "
            "Use these rows for generation/submission alignment; use official non-empty reference JSONL files "
            "for final metric evaluation."
        )
    save_json(meta, out_dir / "input_metadata.json")
    print(f"[stage00] articles={len(rows)} -> {out_articles}")
    if meta["empty_reference_count"]:
        print("[stage00] WARNING: empty references detected; final evaluation needs official reference JSONL files.")


def stage01_extract_keep_af(
    run_name: str,
    model: str,
    chunk_words: int,
    overlap_words: int,
    resume: bool,
) -> None:
    articles = _load_articles(run_name)
    out_dir = _stage_dir(run_name, "01_selected_keep_af")
    out_path = out_dir / "selected_keep_af.jsonl"
    prompt = (PROMPTS_EXP2_DIR / "selective_keep_af_ultra_recall.txt").read_text(encoding="utf-8")
    client = OpenAIClient()

    rows = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("article_id")) for r in rows}
    failed: list[dict[str, Any]] = []
    for art in tqdm(articles, desc="stage01 | Part C selected keep AF", total=len(articles)):
        article_id = str(art["id"])
        if resume and article_id in done:
            continue
        seen: set[str] = set()
        cnt = 0
        for ch in _chunk_words(str(art["article"]), chunk_words, overlap_words):
            raw = ""
            try:
                resp = client.chat(
                    messages=[{"role": "user", "content": prompt.replace("{source_text}", ch["chunk_text"])}],
                    model=model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                raw = str(resp.get("content", ""))
                obj = json.loads(raw)
                facts = obj.get("keep_atomic_facts", [])
                if not isinstance(facts, list):
                    facts = []
                for fact_obj in facts:
                    if not isinstance(fact_obj, dict):
                        continue
                    fact = str(fact_obj.get("fact", "")).strip()
                    source_span = str(fact_obj.get("source_span", "")).strip()
                    reasoning = str(fact_obj.get("reasoning", "")).strip()
                    n = _norm(fact)
                    if not fact or n in seen:
                        continue
                    seen.add(n)
                    cnt += 1
                    rows.append(
                        {
                            "af_id": f"{article_id}_keep_af_{cnt:04d}",
                            "article_id": article_id,
                            "source_dataset": art["source_dataset"],
                            "original_index": art["original_index"],
                            "fact": fact,
                            "source_span": source_span,
                            "keep": True,
                            "keep_reasoning": reasoning,
                            "article_chunk_idx": ch["chunk_idx"],
                            "chunk_start_word": ch["start_word"],
                            "chunk_end_word": ch["end_word"],
                        }
                    )
            except Exception as exc:
                failed.append({"article_id": article_id, "chunk_idx": ch["chunk_idx"], "error": str(exc), "raw_preview": raw[:300]})
        done.add(article_id)
        save_jsonl(rows, out_path)

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "method": "Experiment 2 Part C selected keep AF; Module 2 human review skipped.",
            "chunk_words": chunk_words,
            "overlap_words": overlap_words,
            "selected_keep_af_total": len(rows),
            "selected_keep_af_by_source": dict(Counter(str(r["source_dataset"]) for r in rows)),
            "selected_keep_af_by_article": dict(Counter(str(r["article_id"]) for r in rows)),
            "failed_chunks": failed,
        },
        out_dir / "selected_keep_af_metadata.json",
    )
    print(f"[stage01] selected_keep_AF={len(rows)} failed_chunks={len(failed)} -> {out_dir}")


def _generate_question(client: OpenAIClient, af: dict[str, Any], model: str, prompt: str) -> dict[str, Any]:
    try:
        resp = client.chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt.replace("{atomic_fact}", str(af["fact"])).replace("{source_span}", str(af.get("source_span", ""))),
                }
            ],
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        obj = json.loads(str(resp.get("content", "")))
        true_statement = str(obj.get("true_statement", af["fact"])).strip() or str(af["fact"])
        false_items = obj.get("false_statements", [])
        if not isinstance(false_items, list):
            false_items = []
        false_texts = [str(x.get("text", "")).strip() for x in false_items if isinstance(x, dict) and str(x.get("text", "")).strip()]
    except Exception:
        true_statement = str(af["fact"])
        false_texts = []
    while len(false_texts) < 3:
        cand = _mutate_sentence(true_statement if len(false_texts) % 2 == 0 else str(af["fact"]))
        false_texts.append(cand if _norm(cand) != _norm(true_statement) else f"{af['fact']} (not supported detail {len(false_texts) + 1})")

    options_with_label = [("TRUE", true_statement)] + [("FALSE", x) for x in false_texts[:3]]
    seed = int(hashlib.md5(str(af["af_id"]).encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    rng.shuffle(options_with_label)
    correct_idx = next(i for i, (lab, _) in enumerate(options_with_label) if lab == "TRUE")
    return {
        "af_id": af["af_id"],
        "article_id": af["article_id"],
        "source_dataset": af["source_dataset"],
        "question_id": f"{af['af_id']}_q",
        "options": [x for _, x in options_with_label],
        "correct_letter": "ABCD"[correct_idx],
    }


def stage02_generate_questions(run_name: str, model: str, resume: bool) -> None:
    out_dir = _stage_dir(run_name, "02_questions")
    out_path = out_dir / "questions_1t3f_keep_af.jsonl"
    af_rows = load_jsonl(_stage_dir(run_name, "01_selected_keep_af") / "selected_keep_af.jsonl")
    existing = load_jsonl(out_path) if resume and out_path.exists() else []
    questions = {str(q["af_id"]): q for q in existing if str(q.get("af_id", ""))}
    client = OpenAIClient()
    prompt = (PROMPTS_EXP2_DIR / "coverage_question_generation.txt").read_text(encoding="utf-8")

    for af in tqdm(af_rows, desc="stage02 | build 1T3F questions", total=len(af_rows)):
        af_id = str(af["af_id"])
        if resume and af_id in questions:
            continue
        questions[af_id] = _generate_question(client, af, model, prompt)
        save_jsonl(list(questions.values()), out_path)

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "prompt_source": str(PROMPTS_EXP2_DIR / "coverage_question_generation.txt"),
            "af_total": len(af_rows),
            "questions": len(questions),
        },
        out_dir / "question_metadata.json",
    )
    print(f"[stage02] questions={len(questions)} -> {out_dir}")


def stage03_generate_initial_summaries(run_name: str, model: str, resume: bool) -> None:
    articles = _load_articles(run_name)
    out_dir = _stage_dir(run_name, "03_summaries")
    out_path = out_dir / "summaries_iter0.jsonl"
    prompt = (PROMPTS_EXP3_DIR / "summary_generation.txt").read_text(encoding="utf-8")
    client = OpenAIClient()
    rows = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("article_id")) for r in rows}
    failed: list[dict[str, Any]] = []

    for art in tqdm(articles, desc="stage03 | initial summaries", total=len(articles)):
        article_id = str(art["id"])
        if resume and article_id in done:
            continue
        raw = ""
        summary = ""
        parse_error = False
        try:
            resp = client.chat(
                messages=[{"role": "user", "content": prompt.replace("{article}", str(art["document"]))}],
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = str(resp.get("content", ""))
            obj = json.loads(raw)
            summary = str(obj.get("summary", "")).strip()
            if not summary:
                raise ValueError("empty summary")
        except Exception as exc:
            parse_error = True
            failed.append({"article_id": article_id, "error": str(exc), "raw_preview": raw[:300]})
            summary = raw.strip()
        rows.append(
            {
                "article_id": article_id,
                "source_dataset": art["source_dataset"],
                "original_index": art["original_index"],
                "iteration": 0,
                "summary": summary,
                "parse_error": parse_error,
            }
        )
        done.add(article_id)
        save_jsonl(rows, out_path)

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "summaries": len(rows),
            "failed": failed,
        },
        out_dir / "summaries_iter0_metadata.json",
    )
    print(f"[stage03] summaries={len(rows)} failed={len(failed)} -> {out_dir}")


def _retrieve_summary_chunks(
    *,
    article_id: str,
    summary: str,
    facts: list[dict[str, Any]],
    top_k: int,
    chunk_words: int,
    overlap_words: int,
    embed_model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunks = _chunk_words(summary, chunk_words, overlap_words)
    for ch in chunks:
        ch["article_id"] = article_id
        ch["chunk_id"] = f"{article_id}_summary_c{int(ch['chunk_idx']):03d}"
    if not chunks or not facts:
        return chunks, []
    oai = OpenAI(api_key=config.OPENAI_API_KEY)
    chunk_embs = _embed_texts(oai, [c["chunk_text"] for c in chunks], model=embed_model)
    fact_embs = _embed_texts(oai, [str(f["fact"]) for f in facts], model=embed_model)
    retrieval_rows: list[dict[str, Any]] = []
    for i, af in enumerate(facts):
        scored = [(_cosine(fact_embs[i], chunk_embs[j]), chunks[j]) for j in range(len(chunks))]
        scored.sort(key=lambda x: x[0], reverse=True)
        retrieval_rows.append(
            {
                "af_id": af["af_id"],
                "article_id": article_id,
                "top_k": top_k,
                "retrieved_chunks": [
                    {
                        "rank": rank + 1,
                        "chunk_id": ch["chunk_id"],
                        "score": round(float(score), 6),
                        "chunk_text": ch["chunk_text"],
                    }
                    for rank, (score, ch) in enumerate(scored[:top_k])
                ],
            }
        )
    return chunks, retrieval_rows


def _summaries_for_iteration(run_name: str, iteration: int) -> dict[str, dict[str, Any]]:
    path = _stage_dir(run_name, "03_summaries") / f"summaries_iter{iteration}.jsonl"
    if not path.exists():
        return {}
    return {str(r["article_id"]): r for r in load_jsonl(path)}


def stage04_check_iteration(
    run_name: str,
    model: str,
    iteration: int,
    top_k: int,
    chunk_words: int,
    overlap_words: int,
    embed_model: str,
    resume: bool,
) -> None:
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing. Required for summary chunk retrieval and LLM checking.")
    out_dir = _stage_dir(run_name, f"04_checks_iter{iteration}")
    out_path = out_dir / "coverage_predictions.jsonl"
    out_retrieval = out_dir / "retrieval_topk.jsonl"
    out_chunks = out_dir / "summary_chunks.jsonl"
    af_rows = load_jsonl(_stage_dir(run_name, "01_selected_keep_af") / "selected_keep_af.jsonl")
    questions = {str(q["af_id"]): q for q in load_jsonl(_stage_dir(run_name, "02_questions") / "questions_1t3f_keep_af.jsonl")}
    summaries = _summaries_for_iteration(run_name, iteration)
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for af in af_rows:
        by_article[str(af["article_id"])].append(af)

    rows = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("af_id")) for r in rows}
    all_chunks: list[dict[str, Any]] = []
    all_retrieval: list[dict[str, Any]] = []
    setup_c_system = SETUP_C_PROMPT.read_text(encoding="utf-8")
    client = OpenAIClient()
    parse_errors = 0

    for article_id, summ in tqdm(summaries.items(), desc=f"stage04 | check iter {iteration}", total=len(summaries)):
        facts = by_article.get(article_id, [])
        chunks, retrieval = _retrieve_summary_chunks(
            article_id=article_id,
            summary=str(summ["summary"]),
            facts=facts,
            top_k=top_k,
            chunk_words=chunk_words,
            overlap_words=overlap_words,
            embed_model=embed_model,
        )
        all_chunks.extend(chunks)
        all_retrieval.extend(retrieval)
        retr_map = {str(r["af_id"]): r for r in retrieval}
        for af in facts:
            af_id = str(af["af_id"])
            if resume and af_id in done:
                continue
            q = questions[af_id]
            opts = q["options"]
            chunks_txt = "\n\n".join([f"[{c['rank']}] {c['chunk_text']}" for c in retr_map.get(af_id, {}).get("retrieved_chunks", [])])
            user_msg = (
                "Context Chunks:\n"
                f"{chunks_txt}\n\n"
                "Question: Which option is supported by the context under the SAME strict policy as direct claim checking?\n"
                f"A. {opts[0]}\n"
                f"B. {opts[1]}\n"
                f"C. {opts[2]}\n"
                f"D. {opts[3]}\n"
                "E. None of the above\n"
            )
            option_eval: dict[str, dict[str, Any]] = {}
            pred_letter = "E"
            reason = ""
            parse_error = False
            try:
                resp = client.chat(
                    messages=[
                        {"role": "system", "content": setup_c_system},
                        {"role": "user", "content": user_msg},
                    ],
                    model=model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                obj = json.loads(str(resp.get("content", "")))
                option_eval = _parse_option_eval(obj)
                pred_letter = _decide_setup_c_answer(option_eval, obj)
                reason = str(obj.get("reasoning", "")).strip()
            except Exception as exc:
                parse_error = True
                reason = f"parse_or_runtime_error: {exc}"
            if parse_error:
                parse_errors += 1
            covered = (not parse_error) and pred_letter == q["correct_letter"]
            true_eval = option_eval.get(q["correct_letter"], {})
            rows.append(
                {
                    "af_id": af_id,
                    "article_id": article_id,
                    "source_dataset": af["source_dataset"],
                    "iteration": iteration,
                    "fact": af["fact"],
                    "question_id": q["question_id"],
                    "options": q["options"] + ["None of the above"],
                    "correct_letter": q["correct_letter"],
                    "predicted_letter": pred_letter,
                    "covered": covered,
                    "error_type": "none" if covered else ("parse_error" if parse_error else "missing_or_incorrect"),
                    "supporting_span": true_eval.get("evidence", "Quote: NONE") if covered else "Quote: NONE",
                    "true_option_status": true_eval.get("status", ""),
                    "retrieved_chunks": retr_map.get(af_id, {}).get("retrieved_chunks", []),
                    "option_evaluation": option_eval,
                    "reason": reason,
                    "parse_error": parse_error,
                }
            )
            done.add(af_id)
            save_jsonl(rows, out_path)

    save_jsonl(all_chunks, out_chunks)
    save_jsonl(all_retrieval, out_retrieval)
    total = len(rows)
    covered_count = sum(1 for r in rows if r.get("covered"))
    by_article_metrics = {}
    for article_id in sorted({str(r["article_id"]) for r in rows}):
        sub = [r for r in rows if str(r["article_id"]) == article_id]
        cov = sum(1 for r in sub if r.get("covered"))
        by_article_metrics[article_id] = {
            "af_total": len(sub),
            "covered": cov,
            "errors": len(sub) - cov,
            "coverage_rate": round(cov / len(sub), 4) if sub else 0.0,
        }
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "iteration": iteration,
            "setup": "Experiment 1 Setup C 1T3F + E(None), fair strict policy",
            "top_k": top_k,
            "summary_chunk_words": chunk_words,
            "summary_overlap_words": overlap_words,
            "embedding_model": embed_model,
            "af_total": total,
            "covered": covered_count,
            "errors": total - covered_count,
            "coverage_rate": round(covered_count / total, 4) if total else 0.0,
            "parse_error_count": sum(1 for r in rows if r.get("parse_error")),
            "parse_errors_this_run": parse_errors,
            "by_article": by_article_metrics,
        },
        out_dir / "coverage_summary.json",
    )
    print(f"[stage04] iter={iteration} covered={covered_count}/{total} -> {out_dir}")


def stage05_rewrite_iteration(run_name: str, model: str, from_iteration: int, resume: bool) -> None:
    articles = {str(r["id"]): r for r in _load_articles(run_name)}
    current = _summaries_for_iteration(run_name, from_iteration)
    check_path = _stage_dir(run_name, f"04_checks_iter{from_iteration}") / "coverage_predictions.jsonl"
    checks = load_jsonl(check_path)
    by_article_errors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in checks:
        if not row.get("covered"):
            by_article_errors[str(row["article_id"])].append(row)

    out_dir = _stage_dir(run_name, "03_summaries")
    out_path = out_dir / f"summaries_iter{from_iteration + 1}.jsonl"
    rows = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("article_id")) for r in rows}
    prompt = (PROMPTS_EXP3_DIR / "rewrite_summary.txt").read_text(encoding="utf-8")
    client = OpenAIClient()

    for article_id, summ in tqdm(current.items(), desc=f"stage05 | rewrite iter {from_iteration + 1}", total=len(current)):
        if resume and article_id in done:
            continue
        errors = by_article_errors.get(article_id, [])
        if not errors:
            rows.append({**summ, "iteration": from_iteration + 1, "summary": summ["summary"], "rewrite_skipped_no_errors": True})
            done.add(article_id)
            save_jsonl(rows, out_path)
            continue
        feedback = "\n".join(
            f"- AF {i + 1}: {e['fact']}\n  Problem: predicted {e.get('predicted_letter')} instead of {e.get('correct_letter')}. Reason: {e.get('reason', '')}"
            for i, e in enumerate(errors)
        )
        raw = ""
        new_summary = ""
        revision_notes: list[str] = []
        parse_error = False
        try:
            resp = client.chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt.replace("{article}", str(articles[article_id]["document"]))
                        .replace("{summary}", str(summ["summary"]))
                        .replace("{feedback}", feedback),
                    }
                ],
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = str(resp.get("content", ""))
            obj = json.loads(raw)
            new_summary = str(obj.get("summary", "")).strip()
            raw_notes = obj.get("revision_notes", [])
            revision_notes = [str(x) for x in raw_notes] if isinstance(raw_notes, list) else [str(raw_notes)]
            if not new_summary:
                raise ValueError("empty rewritten summary")
        except Exception as exc:
            parse_error = True
            revision_notes = [f"parse_or_runtime_error: {exc}"]
            new_summary = raw.strip() or str(summ["summary"])
        rows.append(
            {
                "article_id": article_id,
                "source_dataset": summ["source_dataset"],
                "original_index": summ["original_index"],
                "iteration": from_iteration + 1,
                "summary": new_summary,
                "feedback_error_count": len(errors),
                "revision_notes": revision_notes,
                "parse_error": parse_error,
            }
        )
        done.add(article_id)
        save_jsonl(rows, out_path)
    print(f"[stage05] wrote summaries_iter{from_iteration + 1}.jsonl")


def stage06_build_submission_files(run_name: str, final_iteration: int) -> None:
    articles = _load_articles(run_name)
    initial = _summaries_for_iteration(run_name, 0)
    final = _summaries_for_iteration(run_name, final_iteration)
    out_dir = _stage_dir(run_name, "06_eval_inputs")
    rows_summary: list[dict[str, Any]] = []

    def write_pair(prefix: str, summ_map: dict[str, dict[str, Any]]) -> None:
        by_source = {"PLOS": [], "eLife": []}
        for art in articles:
            sid = str(art["id"])
            by_source[str(art["source_dataset"])].append(str(summ_map.get(sid, {}).get("summary", "")))
        (out_dir / f"plos_{prefix}.txt").write_text("\n".join(by_source["PLOS"]) + "\n", encoding="utf-8")
        (out_dir / f"elife_{prefix}.txt").write_text("\n".join(by_source["eLife"]) + "\n", encoding="utf-8")

    write_pair("initial", initial)
    write_pair("rewritten", final)
    for art in articles:
        article_id = str(art["id"])
        rows_summary.append(
            {
                "article_id": article_id,
                "source_dataset": art["source_dataset"],
                "original_index": art["original_index"],
                "initial_summary": initial.get(article_id, {}).get("summary", ""),
                "rewritten_summary": final.get(article_id, {}).get("summary", ""),
                "reference": art["reference"],
            }
        )
    save_jsonl(rows_summary, out_dir / "paired_summaries.jsonl")
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "final_iteration": final_iteration,
            "files": {
                "initial": ["plos_initial.txt", "elife_initial.txt"],
                "rewritten": ["plos_rewritten.txt", "elife_rewritten.txt"],
                "ground_truth": ["../00_inputs/PLOS_test.jsonl", "../00_inputs/eLife_test.jsonl"],
            },
            "evaluation_note": "For evaluation_final.evaluate_all, pass list[str] predictions and JSONL truth dicts with document/reference.",
        },
        out_dir / "eval_input_metadata.json",
    )
    print(f"[stage06] eval input files -> {out_dir}")


def stage07_summarize_run(run_name: str, max_iteration: int) -> None:
    run_dir = _run_dir(run_name)
    out_dir = EXP3_RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for i in range(max_iteration + 1):
        path = run_dir / f"04_checks_iter{i}" / "coverage_summary.json"
        if not path.exists():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "iteration": i,
                "af_total": obj.get("af_total", 0),
                "covered": obj.get("covered", 0),
                "errors": obj.get("errors", 0),
                "coverage_rate": obj.get("coverage_rate", 0),
                "parse_error_count": obj.get("parse_error_count", 0),
            }
        )
    if rows:
        with (out_dir / "iteration_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    save_json(rows, out_dir / "iteration_summary.json")
    print(f"[stage07] summary -> {out_dir}")


def run_all(args: argparse.Namespace) -> None:
    stage00_prepare_inputs(args.run_name, args.split, args.n_per_source, args.seed)
    stage01_extract_keep_af(args.run_name, args.model, args.af_chunk_words, args.af_overlap_words, resume=not args.no_resume)
    stage02_generate_questions(args.run_name, args.model, resume=not args.no_resume)
    stage03_generate_initial_summaries(args.run_name, args.model, resume=not args.no_resume)

    final_iteration = 0
    for iteration in range(args.max_rewrites + 1):
        stage04_check_iteration(
            args.run_name,
            args.model,
            iteration,
            args.top_k,
            args.summary_chunk_words,
            args.summary_overlap_words,
            args.embed_model,
            resume=not args.no_resume,
        )
        summary_path = _stage_dir(args.run_name, f"04_checks_iter{iteration}") / "coverage_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        final_iteration = iteration
        if int(summary.get("errors", 0)) == 0:
            break
        if iteration < args.max_rewrites:
            stage05_rewrite_iteration(args.run_name, args.model, iteration, resume=not args.no_resume)

    stage06_build_submission_files(args.run_name, final_iteration)
    stage07_summarize_run(args.run_name, final_iteration)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3 AF-guided iterative lay summarization pipeline.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run-name", type=str, default="pilot_n20_gpt41_mini")
        p.add_argument("--model", type=str, default=DEFAULT_MODEL)
        p.add_argument("--no-resume", action="store_true")

    p_all = sub.add_parser("run_all")
    add_common(p_all)
    p_all.add_argument("--split", type=str, default="test")
    p_all.add_argument("--n-per-source", type=int, default=10, help="Use 10 for a 20-article pilot; omit via stage00 for full 142+142.")
    p_all.add_argument("--seed", type=int, default=20260611)
    p_all.add_argument("--max-rewrites", type=int, default=2)
    p_all.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p_all.add_argument("--summary-chunk-words", type=int, default=DEFAULT_SUMMARY_CHUNK_WORDS)
    p_all.add_argument("--summary-overlap-words", type=int, default=DEFAULT_SUMMARY_OVERLAP_WORDS)
    p_all.add_argument("--af-chunk-words", type=int, default=DEFAULT_AF_CHUNK_WORDS)
    p_all.add_argument("--af-overlap-words", type=int, default=DEFAULT_AF_OVERLAP_WORDS)
    p_all.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL)

    p0 = sub.add_parser("stage00_prepare_inputs")
    p0.add_argument("--run-name", type=str, default="pilot_n20_gpt41_mini")
    p0.add_argument("--split", type=str, default="test")
    p0.add_argument("--n-per-source", type=int, default=10)
    p0.add_argument("--full-test", action="store_true")
    p0.add_argument("--seed", type=int, default=20260611)

    p1 = sub.add_parser("stage01_extract_keep_af")
    add_common(p1)
    p1.add_argument("--af-chunk-words", type=int, default=DEFAULT_AF_CHUNK_WORDS)
    p1.add_argument("--af-overlap-words", type=int, default=DEFAULT_AF_OVERLAP_WORDS)

    p2 = sub.add_parser("stage02_generate_questions")
    add_common(p2)

    p3 = sub.add_parser("stage03_generate_initial_summaries")
    add_common(p3)

    p4 = sub.add_parser("stage04_check_iteration")
    add_common(p4)
    p4.add_argument("--iteration", type=int, required=True)
    p4.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p4.add_argument("--summary-chunk-words", type=int, default=DEFAULT_SUMMARY_CHUNK_WORDS)
    p4.add_argument("--summary-overlap-words", type=int, default=DEFAULT_SUMMARY_OVERLAP_WORDS)
    p4.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL)

    p5 = sub.add_parser("stage05_rewrite_iteration")
    add_common(p5)
    p5.add_argument("--from-iteration", type=int, required=True)

    p6 = sub.add_parser("stage06_build_submission_files")
    p6.add_argument("--run-name", type=str, default="pilot_n10_gpt41_mini")
    p6.add_argument("--final-iteration", type=int, required=True)

    p7 = sub.add_parser("stage07_summarize_run")
    p7.add_argument("--run-name", type=str, default="pilot_n10_gpt41_mini")
    p7.add_argument("--max-iteration", type=int, required=True)

    args = parser.parse_args()
    if args.cmd == "run_all":
        run_all(args)
    elif args.cmd == "stage00_prepare_inputs":
        stage00_prepare_inputs(args.run_name, args.split, None if args.full_test else args.n_per_source, args.seed)
    elif args.cmd == "stage01_extract_keep_af":
        stage01_extract_keep_af(args.run_name, args.model, args.af_chunk_words, args.af_overlap_words, resume=not args.no_resume)
    elif args.cmd == "stage02_generate_questions":
        stage02_generate_questions(args.run_name, args.model, resume=not args.no_resume)
    elif args.cmd == "stage03_generate_initial_summaries":
        stage03_generate_initial_summaries(args.run_name, args.model, resume=not args.no_resume)
    elif args.cmd == "stage04_check_iteration":
        stage04_check_iteration(
            args.run_name,
            args.model,
            args.iteration,
            args.top_k,
            args.summary_chunk_words,
            args.summary_overlap_words,
            args.embed_model,
            resume=not args.no_resume,
        )
    elif args.cmd == "stage05_rewrite_iteration":
        stage05_rewrite_iteration(args.run_name, args.model, args.from_iteration, resume=not args.no_resume)
    elif args.cmd == "stage06_build_submission_files":
        stage06_build_submission_files(args.run_name, args.final_iteration)
    elif args.cmd == "stage07_summarize_run":
        stage07_summarize_run(args.run_name, args.max_iteration)


if __name__ == "__main__":
    main()
