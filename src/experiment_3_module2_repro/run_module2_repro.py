from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from datasets import load_dataset
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.experiment_2.run_three_model_exp2 import MODEL_CONFIGS, ProviderClient, UsageTracker
from src.utils import load_jsonl, save_json, save_jsonl


DATA_DIR = ROOT_DIR / "data" / "experiment_3_module2_repro"
RUNS_DIR = DATA_DIR / "runs"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

DATASETS = {
    "PLOS": ("BioLaySumm/BioLaySumm2025-PLOS", "plos"),
    "eLife": ("BioLaySumm/BioLaySumm2025-eLife", "elife"),
}

DEFAULT_MODELS = ["gpt41_mini", "gemini3_flash_preview_minimal", "gemini25_flash_non_thinking"]
DEFAULT_JUDGE_MODEL = "gemini31_flash_lite"
DEFAULT_CHUNK_WORDS = 1200
DEFAULT_OVERLAP_WORDS = 120


def _run_dir(run_name: str) -> Path:
    return RUNS_DIR / run_name


def _stage_dir(run_name: str, stage: str) -> Path:
    path = _run_dir(run_name) / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def _model_dir(run_name: str, stage: str, model_key: str) -> Path:
    path = _stage_dir(run_name, stage) / model_key
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def _chunk_words(text: str, chunk_words: int, overlap_words: int) -> list[dict[str, Any]]:
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
    return chunks


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt(path_name: str) -> str:
    return (PROMPTS_DIR / path_name).read_text(encoding="utf-8")


def _load_articles(run_name: str) -> list[dict[str, Any]]:
    return load_jsonl(_stage_dir(run_name, "00_inputs") / "articles.jsonl")


def _usage(run_name: str, resume: bool) -> UsageTracker:
    usage_path = _run_dir(run_name) / "api_usage_calls.csv"
    return UsageTracker.from_csv(usage_path) if resume else UsageTracker(rows=[])


def _save_usage(run_name: str, usage: UsageTracker) -> None:
    _run_dir(run_name).mkdir(parents=True, exist_ok=True)
    usage.save(_run_dir(run_name))


def _run_parallel(
    items: Iterable[Any],
    worker: Callable[[Any], Any],
    *,
    max_workers: int,
    desc: str,
) -> list[Any]:
    item_list = list(items)
    if max_workers <= 1:
        out = []
        for item in tqdm(item_list, desc=desc, total=len(item_list)):
            out.append(worker(item))
        return out
    out = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, item) for item in item_list]
        for fut in tqdm(as_completed(futures), desc=desc, total=len(futures)):
            out.append(fut.result())
    return out


def _mutate_sentence(sentence: str) -> str:
    text = str(sentence or "").strip()
    if not text:
        return "This statement is not supported."
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
        if re.search(pat, text, flags=re.IGNORECASE):
            return re.sub(pat, rep, text, count=1, flags=re.IGNORECASE)
    return "It is not true that " + text[:1].lower() + text[1:]


def stage00_prepare_inputs(run_name: str, split: str, n_per_source: int, seed: int, selection_mode: str) -> None:
    out_dir = _stage_dir(run_name, "00_inputs")
    out_articles = out_dir / "articles.jsonl"
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "split": split,
        "seed": seed,
        "n_per_source": n_per_source,
        "selection_mode": selection_mode,
        "datasets": {},
        "note": "Validation split is used because simulated Module 2 requires non-empty expert summaries.",
    }

    for source, (dataset_name, prefix) in DATASETS.items():
        ds = load_dataset(dataset_name, split=split, token=config.HF_TOKEN or None)
        cols = set(ds.column_names)
        article_col = "article" if "article" in cols else ("document" if "document" in cols else None)
        summary_col = "lay_summary" if "lay_summary" in cols else ("summary" if "summary" in cols else ("reference" if "reference" in cols else None))
        abstract_col = "abstract" if "abstract" in cols else None
        if article_col is None or summary_col is None:
            raise RuntimeError(f"{source}: cannot identify article/expert summary columns: {ds.column_names}")

        indices = list(range(len(ds)))
        if len(indices) < n_per_source:
            raise RuntimeError(f"{source}: split has {len(indices)} rows, need {n_per_source}")
        if selection_mode == "first":
            selected = indices[:n_per_source]
        elif selection_mode == "random":
            selected = sorted(rng.sample(indices, n_per_source))
        else:
            raise ValueError("selection_mode must be 'first' or 'random'")
        meta["datasets"][source] = {
            "dataset_name": dataset_name,
            "split_size": len(ds),
            "selected_count": len(selected),
            "selected_indices": selected,
            "columns": ds.column_names,
        }

        for order, idx in enumerate(selected, start=1):
            item = ds[int(idx)]
            article = str(item.get(article_col, "") or "")
            abstract = str(item.get(abstract_col, "") or "") if abstract_col else ""
            expert_summary = str(item.get(summary_col, "") or "")
            if not expert_summary.strip():
                raise RuntimeError(
                    f"{source} index {idx}: expert summary is empty. "
                    "Use validation or provide references before running Module 2."
                )
            document = f"{abstract}\n{article}".strip() if abstract else article
            article_id = f"m2_{prefix}_{idx:04d}"
            rows.append(
                {
                    "id": article_id,
                    "source_dataset": source,
                    "dataset_name": dataset_name,
                    "split": split,
                    "original_index": int(idx),
                    "article": article,
                    "abstract": abstract,
                    "document": document,
                    "expert_summary": expert_summary,
                    "article_word_count": _word_count(article),
                    "expert_summary_word_count": _word_count(expert_summary),
                    "run_order": order,
                }
            )

    save_jsonl(rows, out_articles)
    meta["total_articles"] = len(rows)
    meta["by_source"] = dict(Counter(str(r["source_dataset"]) for r in rows))
    save_json(meta, out_dir / "input_metadata.json")
    print(f"[stage00] articles={len(rows)} -> {out_articles}")


def stage01_extract_candidate_keep_af(
    run_name: str,
    model_key: str,
    *,
    chunk_words: int,
    overlap_words: int,
    max_workers: int,
    resume: bool,
    usage: UsageTracker,
) -> list[dict[str, Any]]:
    articles = _load_articles(run_name)
    out_dir = _model_dir(run_name, "01_candidate_keep_af", model_key)
    out_path = out_dir / "candidate_keep_af.jsonl"
    existing = load_jsonl(out_path) if resume and out_path.exists() else []
    done_articles = {str(r.get("article_id")) for r in existing}
    prompt = _prompt("extract_candidate_keep_af.txt")
    client = ProviderClient(model_key, usage)

    def worker(art: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        article_id = str(art["id"])
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        count = 0
        for chunk in _chunk_words(str(art["article"]), chunk_words, overlap_words):
            raw = ""
            try:
                raw = client.chat(
                    stage=f"stage01_extract:{model_key}",
                    item_id=f"{article_id}:chunk{chunk['chunk_idx']}",
                    messages=[{"role": "user", "content": prompt.replace("{source_text}", chunk["chunk_text"])}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                obj = json.loads(raw)
                facts = obj.get("keep_atomic_facts", [])
                if not isinstance(facts, list):
                    facts = []
                for fact_obj in facts:
                    if not isinstance(fact_obj, dict):
                        continue
                    fact = str(fact_obj.get("fact", "")).strip()
                    norm = _norm(fact)
                    if not fact or norm in seen:
                        continue
                    seen.add(norm)
                    count += 1
                    rows.append(
                        {
                            "af_id": f"{model_key}_{article_id}_keep_af_{count:04d}",
                            "model_key": model_key,
                            "article_id": article_id,
                            "source_dataset": art["source_dataset"],
                            "original_index": art["original_index"],
                            "fact": fact,
                            "source_span": str(fact_obj.get("source_span", "")).strip(),
                            "keep_reasoning": str(fact_obj.get("reasoning", "")).strip(),
                            "article_chunk_idx": chunk["chunk_idx"],
                            "chunk_start_word": chunk["start_word"],
                            "chunk_end_word": chunk["end_word"],
                        }
                    )
            except Exception as exc:
                failed.append(
                    {
                        "article_id": article_id,
                        "chunk_idx": chunk["chunk_idx"],
                        "error": str(exc),
                        "raw_preview": raw[:300],
                    }
                )
        return article_id, rows, failed

    todo = [a for a in articles if str(a["id"]) not in done_articles]
    new_rows: list[dict[str, Any]] = []
    failed_chunks: list[dict[str, Any]] = []
    for _, rows, failed in _run_parallel(todo, worker, max_workers=max_workers, desc=f"stage01 | {model_key}"):
        new_rows.extend(rows)
        failed_chunks.extend(failed)

    all_rows = existing + new_rows
    all_rows.sort(key=lambda r: (str(r["article_id"]), str(r["af_id"])))
    save_jsonl(all_rows, out_path)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model_key": model_key,
            "provider": MODEL_CONFIGS[model_key]["provider"],
            "model": MODEL_CONFIGS[model_key]["model"],
            "chunk_words": chunk_words,
            "overlap_words": overlap_words,
            "max_workers": max_workers,
            "prompt_sha256": _sha256_text(prompt),
            "articles_total": len(articles),
            "articles_done_before_resume": len(done_articles),
            "candidate_keep_af_total": len(all_rows),
            "candidate_keep_af_by_source": dict(Counter(str(r["source_dataset"]) for r in all_rows)),
            "candidate_keep_af_by_article": dict(Counter(str(r["article_id"]) for r in all_rows)),
            "failed_chunks": failed_chunks,
        },
        out_dir / "candidate_keep_af_metadata.json",
    )
    print(f"[stage01] {model_key} candidate_keep_af={len(all_rows)} failed_chunks={len(failed_chunks)}")
    return all_rows


def _is_final_keep(judgement: dict[str, Any]) -> bool:
    return (
        _coerce_bool(judgement.get("covered_by_expert_summary", False))
        and str(judgement.get("coverage_type", "")).strip() in {"exact", "paraphrase", "lay_generalization"}
        and str(judgement.get("confidence", "")).strip() in {"high", "medium"}
        and bool(str(judgement.get("supporting_span", "") or "").strip())
        and not _coerce_bool(judgement.get("parse_error", False))
    )


def stage02_module2_judge(
    run_name: str,
    model_key: str,
    judge_model_key: str,
    *,
    max_workers: int,
    resume: bool,
    usage: UsageTracker,
) -> list[dict[str, Any]]:
    articles = {str(r["id"]): r for r in _load_articles(run_name)}
    af_rows = load_jsonl(_model_dir(run_name, "01_candidate_keep_af", model_key) / "candidate_keep_af.jsonl")
    out_dir = _model_dir(run_name, "02_module2_judgements", model_key)
    out_path = out_dir / "module2_judgements.jsonl"
    existing = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("af_id")) for r in existing}
    prompt = _prompt("judge_against_expert_summary.txt")
    client = ProviderClient(judge_model_key, usage)

    def worker(af: dict[str, Any]) -> dict[str, Any]:
        raw = ""
        parse_error = False
        article = articles[str(af["article_id"])]
        try:
            raw = client.chat(
                stage=f"stage02_module2_judge:{judge_model_key}",
                item_id=str(af["af_id"]),
                messages=[
                    {
                        "role": "user",
                        "content": prompt.replace("{atomic_fact}", str(af["fact"])).replace(
                            "{expert_summary}", str(article["expert_summary"])
                        ),
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = json.loads(raw)
        except Exception as exc:
            parse_error = True
            obj = {
                "covered_by_expert_summary": False,
                "coverage_type": "absent",
                "confidence": "low",
                "supporting_span": None,
                "reasoning": f"parse_or_runtime_error: {exc}",
            }
        coverage_type = str(obj.get("coverage_type", "absent")).strip()
        if coverage_type not in {"exact", "paraphrase", "lay_generalization", "topic_only", "absent", "contradicted"}:
            coverage_type = "absent"
        confidence = str(obj.get("confidence", "low")).strip()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        supporting_span = obj.get("supporting_span", None)
        supporting_span = None if supporting_span is None else str(supporting_span).strip()
        if supporting_span in {"", "null", "None"}:
            supporting_span = None
        row = {
            "af_id": af["af_id"],
            "model_key": model_key,
            "article_id": af["article_id"],
            "source_dataset": af["source_dataset"],
            "original_index": af["original_index"],
            "fact": af["fact"],
            "source_span": af.get("source_span", ""),
            "judge_model_key": judge_model_key,
            "judge_provider": MODEL_CONFIGS[judge_model_key]["provider"],
            "judge_model": MODEL_CONFIGS[judge_model_key]["model"],
            "covered_by_expert_summary": _coerce_bool(obj.get("covered_by_expert_summary", False)),
            "coverage_type": coverage_type,
            "confidence": confidence,
            "supporting_span": supporting_span,
            "reasoning": str(obj.get("reasoning", "")).strip(),
            "parse_error": parse_error,
        }
        row["final_keep"] = _is_final_keep(row)
        return row

    todo = [r for r in af_rows if str(r["af_id"]) not in done]
    new_rows = _run_parallel(todo, worker, max_workers=max_workers, desc=f"stage02 | {model_key}")
    all_rows = existing + new_rows
    all_rows.sort(key=lambda r: (str(r["article_id"]), str(r["af_id"])))
    save_jsonl(all_rows, out_path)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model_key": model_key,
            "judge_model_key": judge_model_key,
            "judge_model": MODEL_CONFIGS[judge_model_key]["model"],
            "max_workers": max_workers,
            "prompt_sha256": _sha256_text(prompt),
            "candidate_keep_af_total": len(af_rows),
            "judgements_total": len(all_rows),
            "final_keep_total": sum(1 for r in all_rows if r.get("final_keep")),
            "coverage_type_counts": dict(Counter(str(r.get("coverage_type")) for r in all_rows)),
            "confidence_counts": dict(Counter(str(r.get("confidence")) for r in all_rows)),
            "parse_error_count": sum(1 for r in all_rows if r.get("parse_error")),
        },
        out_dir / "module2_judgements_metadata.json",
    )
    print(f"[stage02] {model_key} judgements={len(all_rows)} final_keep={sum(1 for r in all_rows if r.get('final_keep'))}")
    return all_rows


def stage03_build_final_keep_af(run_name: str, model_key: str) -> list[dict[str, Any]]:
    judgements = load_jsonl(_model_dir(run_name, "02_module2_judgements", model_key) / "module2_judgements.jsonl")
    out_dir = _model_dir(run_name, "03_final_keep_af", model_key)
    rows = []
    for row in judgements:
        if not row.get("final_keep"):
            continue
        rows.append(
            {
                "af_id": row["af_id"],
                "model_key": model_key,
                "article_id": row["article_id"],
                "source_dataset": row["source_dataset"],
                "original_index": row["original_index"],
                "fact": row["fact"],
                "source_span": row.get("source_span", ""),
                "module2_supporting_span": row.get("supporting_span"),
                "module2_coverage_type": row.get("coverage_type"),
                "module2_confidence": row.get("confidence"),
                "module2_reasoning": row.get("reasoning"),
                "judge_model_key": row.get("judge_model_key"),
                "judge_model": row.get("judge_model"),
            }
        )
    rows.sort(key=lambda r: (str(r["article_id"]), str(r["af_id"])))
    save_jsonl(rows, out_dir / "final_keep_af.jsonl")
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model_key": model_key,
            "final_keep_total": len(rows),
            "final_keep_by_source": dict(Counter(str(r["source_dataset"]) for r in rows)),
            "final_keep_by_article": dict(Counter(str(r["article_id"]) for r in rows)),
            "rule": {
                "covered_by_expert_summary": True,
                "coverage_type": ["exact", "paraphrase", "lay_generalization"],
                "confidence": ["high", "medium"],
                "supporting_span": "non-empty",
                "parse_error": False,
            },
        },
        out_dir / "final_keep_af_metadata.json",
    )
    print(f"[stage03] {model_key} final_keep_af={len(rows)}")
    return rows


def stage04_generate_questions(
    run_name: str,
    model_key: str,
    *,
    max_workers: int,
    resume: bool,
    usage: UsageTracker,
) -> list[dict[str, Any]]:
    final_af = load_jsonl(_model_dir(run_name, "03_final_keep_af", model_key) / "final_keep_af.jsonl")
    out_dir = _model_dir(run_name, "04_questions", model_key)
    out_path = out_dir / "questions_1t3f_nota.jsonl"
    existing = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("af_id")) for r in existing}
    prompt = _prompt("generate_1t3f_nota.txt")
    client = ProviderClient(model_key, usage)

    def worker(af: dict[str, Any]) -> dict[str, Any]:
        raw = ""
        parse_error = False
        try:
            raw = client.chat(
                stage=f"stage04_questions:{model_key}",
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
            true_statement = str(obj.get("true_statement", af["fact"])).strip() or str(af["fact"])
            false_items = obj.get("false_statements", [])
            if not isinstance(false_items, list):
                false_items = []
            false_texts = [
                str(x.get("text", "")).strip()
                for x in false_items
                if isinstance(x, dict) and str(x.get("text", "")).strip()
            ]
        except Exception:
            parse_error = True
            true_statement = str(af["fact"])
            false_texts = []
        while len(false_texts) < 3:
            cand = _mutate_sentence(true_statement if len(false_texts) % 2 == 0 else str(af["fact"]))
            false_texts.append(cand if _norm(cand) != _norm(true_statement) else f"{af['fact']} (not supported detail {len(false_texts) + 1})")

        options_with_label = [("TRUE", true_statement)] + [("FALSE", x) for x in false_texts[:3]]
        seed = int(hashlib.md5(str(af["af_id"]).encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        rng.shuffle(options_with_label)
        correct_idx = next(i for i, (label, _) in enumerate(options_with_label) if label == "TRUE")
        options_abcd = [text for _, text in options_with_label]
        options = {letter: options_abcd[i] for i, letter in enumerate("ABCD")}
        options["E"] = "None of the above"
        return {
            "question_id": f"{af['af_id']}_q",
            "af_id": af["af_id"],
            "model_key": model_key,
            "article_id": af["article_id"],
            "source_dataset": af["source_dataset"],
            "original_index": af["original_index"],
            "fact": af["fact"],
            "options": options,
            "correct_letter": "ABCD"[correct_idx],
            "true_statement": true_statement,
            "false_statements": false_texts[:3],
            "nota_included": True,
            "question_generator_model_key": model_key,
            "question_generator_provider": MODEL_CONFIGS[model_key]["provider"],
            "question_generator_model": MODEL_CONFIGS[model_key]["model"],
            "parse_error": parse_error,
        }

    todo = [r for r in final_af if str(r["af_id"]) not in done]
    new_rows = _run_parallel(todo, worker, max_workers=max_workers, desc=f"stage04 | {model_key}")
    all_rows = existing + new_rows
    all_rows.sort(key=lambda r: (str(r["article_id"]), str(r["af_id"])))
    save_jsonl(all_rows, out_path)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model_key": model_key,
            "provider": MODEL_CONFIGS[model_key]["provider"],
            "model": MODEL_CONFIGS[model_key]["model"],
            "max_workers": max_workers,
            "prompt_sha256": _sha256_text(prompt),
            "final_keep_af_total": len(final_af),
            "questions_total": len(all_rows),
            "questions_by_source": dict(Counter(str(r["source_dataset"]) for r in all_rows)),
            "questions_by_article": dict(Counter(str(r["article_id"]) for r in all_rows)),
            "parse_error_count": sum(1 for r in all_rows if r.get("parse_error")),
        },
        out_dir / "question_metadata.json",
    )
    print(f"[stage04] {model_key} questions={len(all_rows)}")
    return all_rows


def stage05_reports(run_name: str, models: list[str], judge_model_key: str, usage: UsageTracker) -> None:
    out_dir = _stage_dir(run_name, "05_reports")
    rows: list[dict[str, Any]] = []
    for model_key in models:
        cand_path = _model_dir(run_name, "01_candidate_keep_af", model_key) / "candidate_keep_af.jsonl"
        judge_path = _model_dir(run_name, "02_module2_judgements", model_key) / "module2_judgements.jsonl"
        final_path = _model_dir(run_name, "03_final_keep_af", model_key) / "final_keep_af.jsonl"
        q_path = _model_dir(run_name, "04_questions", model_key) / "questions_1t3f_nota.jsonl"
        candidate = load_jsonl(cand_path) if cand_path.exists() else []
        judged = load_jsonl(judge_path) if judge_path.exists() else []
        final = load_jsonl(final_path) if final_path.exists() else []
        questions = load_jsonl(q_path) if q_path.exists() else []
        rows.append(
            {
                "model_key": model_key,
                "provider": MODEL_CONFIGS[model_key]["provider"],
                "model": MODEL_CONFIGS[model_key]["model"],
                "judge_model_key": judge_model_key,
                "candidate_keep_af": len(candidate),
                "module2_judgements": len(judged),
                "final_keep_af": len(final),
                "module2_keep_rate": round(len(final) / len(judged), 4) if judged else 0.0,
                "questions": len(questions),
                "candidate_by_source": dict(Counter(str(r["source_dataset"]) for r in candidate)),
                "final_by_source": dict(Counter(str(r["source_dataset"]) for r in final)),
                "questions_by_source": dict(Counter(str(r["source_dataset"]) for r in questions)),
            }
        )
    save_json(rows, out_dir / "model_artifact_summary.json")
    usage.save(_run_dir(run_name))
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "models": models,
            "judge_model_key": judge_model_key,
            "judge_model": MODEL_CONFIGS[judge_model_key]["model"],
            "api_usage": usage.summarize(),
        },
        out_dir / "run_summary.json",
    )
    print(f"[stage05] reports -> {out_dir}")


def run_all(args: argparse.Namespace) -> None:
    usage = _usage(args.run_name, resume=not args.no_resume)
    stage00_prepare_inputs(args.run_name, args.split, args.n_per_source, args.seed, args.selection_mode)
    for model_key in args.models:
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {model_key}. Valid keys include: {sorted(MODEL_CONFIGS)}")
        stage01_extract_candidate_keep_af(
            args.run_name,
            model_key,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
            max_workers=args.max_workers,
            resume=not args.no_resume,
            usage=usage,
        )
        _save_usage(args.run_name, usage)
        stage02_module2_judge(
            args.run_name,
            model_key,
            args.judge_model,
            max_workers=args.max_workers,
            resume=not args.no_resume,
            usage=usage,
        )
        _save_usage(args.run_name, usage)
        stage03_build_final_keep_af(args.run_name, model_key)
        stage04_generate_questions(
            args.run_name,
            model_key,
            max_workers=args.max_workers,
            resume=not args.no_resume,
            usage=usage,
        )
        _save_usage(args.run_name, usage)
    stage05_reports(args.run_name, args.models, args.judge_model, usage)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible simulated Module 2 keep AFs and 1T3F+NOTA questions.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run-name", type=str, default="pilot_val_n10_judge_gemini31_flash_lite")
        p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
        p.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
        p.add_argument("--max-workers", type=int, default=8)
        p.add_argument("--no-resume", action="store_true")

    p_all = sub.add_parser("run_all")
    add_common(p_all)
    p_all.add_argument("--split", type=str, default="validation")
    p_all.add_argument("--n-per-source", type=int, default=5)
    p_all.add_argument("--seed", type=int, default=20260612)
    p_all.add_argument("--selection-mode", choices=["first", "random"], default="first")
    p_all.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    p_all.add_argument("--overlap-words", type=int, default=DEFAULT_OVERLAP_WORDS)

    p0 = sub.add_parser("stage00_prepare_inputs")
    p0.add_argument("--run-name", type=str, default="pilot_val_n10_judge_gemini31_flash_lite")
    p0.add_argument("--split", type=str, default="validation")
    p0.add_argument("--n-per-source", type=int, default=5)
    p0.add_argument("--seed", type=int, default=20260612)
    p0.add_argument("--selection-mode", choices=["first", "random"], default="first")

    p1 = sub.add_parser("stage01_extract_candidate_keep_af")
    add_common(p1)
    p1.add_argument("--model-key", type=str, required=True)
    p1.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    p1.add_argument("--overlap-words", type=int, default=DEFAULT_OVERLAP_WORDS)

    p2 = sub.add_parser("stage02_module2_judge")
    add_common(p2)
    p2.add_argument("--model-key", type=str, required=True)

    p3 = sub.add_parser("stage03_build_final_keep_af")
    p3.add_argument("--run-name", type=str, default="pilot_val_n10_judge_gemini31_flash_lite")
    p3.add_argument("--model-key", type=str, required=True)

    p4 = sub.add_parser("stage04_generate_questions")
    add_common(p4)
    p4.add_argument("--model-key", type=str, required=True)

    p5 = sub.add_parser("stage05_reports")
    add_common(p5)

    args = parser.parse_args()
    if hasattr(args, "judge_model") and args.judge_model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown judge model key: {args.judge_model}. Valid keys include: {sorted(MODEL_CONFIGS)}")

    if args.cmd == "run_all":
        run_all(args)
    elif args.cmd == "stage00_prepare_inputs":
        stage00_prepare_inputs(args.run_name, args.split, args.n_per_source, args.seed, args.selection_mode)
    elif args.cmd == "stage01_extract_candidate_keep_af":
        usage = _usage(args.run_name, resume=not args.no_resume)
        stage01_extract_candidate_keep_af(
            args.run_name,
            args.model_key,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
            max_workers=args.max_workers,
            resume=not args.no_resume,
            usage=usage,
        )
        _save_usage(args.run_name, usage)
    elif args.cmd == "stage02_module2_judge":
        usage = _usage(args.run_name, resume=not args.no_resume)
        stage02_module2_judge(
            args.run_name,
            args.model_key,
            args.judge_model,
            max_workers=args.max_workers,
            resume=not args.no_resume,
            usage=usage,
        )
        _save_usage(args.run_name, usage)
    elif args.cmd == "stage03_build_final_keep_af":
        stage03_build_final_keep_af(args.run_name, args.model_key)
    elif args.cmd == "stage04_generate_questions":
        usage = _usage(args.run_name, resume=not args.no_resume)
        stage04_generate_questions(
            args.run_name,
            args.model_key,
            max_workers=args.max_workers,
            resume=not args.no_resume,
            usage=usage,
        )
        _save_usage(args.run_name, usage)
    elif args.cmd == "stage05_reports":
        usage = _usage(args.run_name, resume=not args.no_resume)
        stage05_reports(args.run_name, args.models, args.judge_model, usage)


if __name__ == "__main__":
    main()
