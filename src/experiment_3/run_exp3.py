from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.experiment_2.run_three_model_exp2 import MODEL_CONFIGS, ProviderClient, UsageTracker
from src.utils import OpenAIClient, load_jsonl, save_json, save_jsonl


EXP3_DATA_DIR = ROOT_DIR / "data" / "experiment_3"
EXP3_RESULTS_DIR = ROOT_DIR / "results" / "experiment_3"
MODULE2_REPRO_DATA_DIR = ROOT_DIR / "data" / "experiment_3_module2_repro"
PROMPTS_EXP2_DIR = ROOT_DIR / "src" / "experiment_2" / "prompts"
PROMPTS_EXP3_DIR = ROOT_DIR / "src" / "experiment_3" / "prompts"
SETUP_C_PROMPT = ROOT_DIR / "src" / "experiment_1" / "prompts_1a_v2_fair_ac" / "setup_c_system.txt"

DATASETS = {
    "PLOS": ("BioLaySumm/BioLaySumm2025-PLOS", "plos"),
    "eLife": ("BioLaySumm/BioLaySumm2025-eLife", "elife"),
}

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_MODEL_KEY = "gpt41_mini"
DEFAULT_TOP_K = 5
DEFAULT_SUMMARY_CHUNK_WORDS = 120
DEFAULT_SUMMARY_OVERLAP_WORDS = 30
DEFAULT_AF_CHUNK_WORDS = 1200
DEFAULT_AF_OVERLAP_WORDS = 120
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_CONTEXT_MODE = "full_summary"


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


def _run_parallel(
    items: Iterable[Any],
    worker: Callable[[Any], Any],
    *,
    max_workers: int,
    desc: str,
) -> list[Any]:
    item_list = list(items)
    if max_workers <= 1:
        return [worker(item) for item in tqdm(item_list, desc=desc, total=len(item_list))]
    out: list[Any] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, item) for item in item_list]
        for fut in tqdm(as_completed(futures), desc=desc, total=len(futures)):
            out.append(fut.result())
    return out


def _usage(run_name: str, resume: bool) -> UsageTracker:
    path = _run_dir(run_name) / "api_usage_calls.csv"
    return UsageTracker.from_csv(path) if resume else UsageTracker(rows=[])


def _save_usage(run_name: str, usage: UsageTracker) -> None:
    usage.save(_run_dir(run_name))


class ChatRunner:
    def __init__(self, *, model: str, model_key: str | None, run_name: str, usage: UsageTracker | None) -> None:
        self.model = model
        self.model_key = model_key
        self.usage = usage
        if model_key and usage is None:
            raise ValueError("usage tracker is required when model_key is used")
        self.provider_client = ProviderClient(model_key, usage) if model_key else None
        self.openai_client = None if model_key else OpenAIClient()

    def chat(
        self,
        *,
        stage: str,
        item_id: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        if self.provider_client is not None:
            return self.provider_client.chat(
                stage=stage,
                item_id=item_id,
                messages=messages,
                temperature=temperature,
                response_format=response_format,
            )
        assert self.openai_client is not None
        resp = self.openai_client.chat(
            messages=messages,
            model=self.model,
            temperature=temperature,
            response_format=response_format,
        )
        return str(resp.get("content", ""))


def _model_label(model: str, model_key: str | None) -> str:
    if model_key:
        cfg = MODEL_CONFIGS[model_key]
        return f"{model_key}:{cfg['model']}"
    return model


def _normalize_options(q: dict[str, Any]) -> dict[str, str]:
    raw = q.get("options", {})
    if isinstance(raw, dict):
        out = {letter: str(raw.get(letter, "")).strip() for letter in ["A", "B", "C", "D", "E"]}
    elif isinstance(raw, list):
        out = {letter: str(raw[i]).strip() if i < len(raw) else "" for i, letter in enumerate(["A", "B", "C", "D"])}
        out["E"] = str(raw[4]).strip() if len(raw) > 4 else "None of the above"
    else:
        out = {letter: "" for letter in ["A", "B", "C", "D"]}
        out["E"] = "None of the above"
    if not out.get("E"):
        out["E"] = "None of the above"
    return out


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


def _embed_texts(
    client: OpenAI,
    texts: list[str],
    model: str,
    batch_size: int = 64,
    max_retries: int = 5,
) -> list[list[float]]:
    embs: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = None
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(min(30.0, 2.0**attempt + random.random()))
        if resp is None:
            raise RuntimeError("embedding request failed without response")
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


def stage00_import_module2_artifacts(
    run_name: str,
    module2_run_name: str,
    module2_model_key: str,
    n_per_source: int | None,
) -> None:
    module2_run_dir = MODULE2_REPRO_DATA_DIR / "runs" / module2_run_name
    articles_path = module2_run_dir / "00_inputs" / "articles.jsonl"
    final_af_path = module2_run_dir / "03_final_keep_af" / module2_model_key / "final_keep_af.jsonl"
    questions_path = module2_run_dir / "04_questions" / module2_model_key / "questions_1t3f_nota.jsonl"
    missing = [str(p) for p in [articles_path, final_af_path, questions_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Module 2 artifact(s): " + ", ".join(missing))

    all_articles = load_jsonl(articles_path)
    selected: list[dict[str, Any]] = []
    by_source_count: Counter[str] = Counter()
    for art in all_articles:
        source = str(art["source_dataset"])
        if n_per_source is not None and by_source_count[source] >= n_per_source:
            continue
        row = dict(art)
        row["reference"] = str(row.get("expert_summary", row.get("reference", "")))
        row["module2_source_run_name"] = module2_run_name
        row["module2_model_key"] = module2_model_key
        selected.append(row)
        by_source_count[source] += 1

    selected_ids = {str(r["id"]) for r in selected}
    final_af = [dict(r, keep=True) for r in load_jsonl(final_af_path) if str(r.get("article_id")) in selected_ids]
    questions = [r for r in load_jsonl(questions_path) if str(r.get("article_id")) in selected_ids]
    question_af_ids = {str(r.get("af_id")) for r in questions}
    final_af = [r for r in final_af if str(r.get("af_id")) in question_af_ids]

    in_dir = _stage_dir(run_name, "00_inputs")
    save_jsonl(selected, in_dir / "articles.jsonl")
    save_jsonl(
        [{"document": r["document"], "reference": r["reference"]} for r in selected if str(r["source_dataset"]) == "eLife"],
        in_dir / "eLife_test.jsonl",
    )
    save_jsonl(
        [{"document": r["document"], "reference": r["reference"]} for r in selected if str(r["source_dataset"]) == "PLOS"],
        in_dir / "PLOS_test.jsonl",
    )
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "source": "experiment_3_module2_repro",
            "module2_run_name": module2_run_name,
            "module2_model_key": module2_model_key,
            "n_per_source": n_per_source,
            "total_articles": len(selected),
            "by_source": dict(Counter(str(r["source_dataset"]) for r in selected)),
            "selected_article_ids": sorted(selected_ids),
            "note": "Imported expert_summary as reference for local BioLaySumm evaluation.",
        },
        in_dir / "input_metadata.json",
    )

    af_dir = _stage_dir(run_name, "01_selected_keep_af")
    save_jsonl(final_af, af_dir / "selected_keep_af.jsonl")
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "source": str(final_af_path),
            "method": "Imported simulated Module 2 final_keep_af; AF extraction and human review skipped in exp3.",
            "selected_keep_af_total": len(final_af),
            "selected_keep_af_by_source": dict(Counter(str(r["source_dataset"]) for r in final_af)),
            "selected_keep_af_by_article": dict(Counter(str(r["article_id"]) for r in final_af)),
        },
        af_dir / "selected_keep_af_metadata.json",
    )

    q_dir = _stage_dir(run_name, "02_questions")
    save_jsonl(questions, q_dir / "questions_1t3f_keep_af.jsonl")
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "source": str(questions_path),
            "question_format": "1T3F + E(None of the above)",
            "questions": len(questions),
            "questions_by_source": dict(Counter(str(r["source_dataset"]) for r in questions)),
            "questions_by_article": dict(Counter(str(r["article_id"]) for r in questions)),
        },
        q_dir / "question_metadata.json",
    )
    print(
        f"[stage00] imported module2 articles={len(selected)} final_keep_AF={len(final_af)} "
        f"questions={len(questions)} -> {_run_dir(run_name)}"
    )


def stage01_extract_keep_af(
    run_name: str,
    model: str,
    chunk_words: int,
    overlap_words: int,
    resume: bool,
    max_workers: int = 1,
) -> None:
    articles = _load_articles(run_name)
    out_dir = _stage_dir(run_name, "01_selected_keep_af")
    out_path = out_dir / "selected_keep_af.jsonl"
    prompt = (PROMPTS_EXP2_DIR / "selective_keep_af_ultra_recall.txt").read_text(encoding="utf-8")

    rows = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("article_id")) for r in rows}
    def worker(art: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        client = OpenAIClient()
        article_id = str(art["id"])
        local_rows: list[dict[str, Any]] = []
        local_failed: list[dict[str, Any]] = []
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
                    local_rows.append(
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
                local_failed.append({"article_id": article_id, "chunk_idx": ch["chunk_idx"], "error": str(exc), "raw_preview": raw[:300]})
        return local_rows, local_failed

    todo = [art for art in articles if not (resume and str(art["id"]) in done)]
    results = _run_parallel(todo, worker, max_workers=max_workers, desc="stage01 | Part C selected keep AF")
    failed: list[dict[str, Any]] = []
    for new_rows, new_failed in results:
        rows.extend(new_rows)
        failed.extend(new_failed)
    rows.sort(key=lambda r: (str(r["article_id"]), str(r["af_id"])))
    save_jsonl(rows, out_path)

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "method": "Experiment 2 Part C selected keep AF; Module 2 human review skipped.",
            "chunk_words": chunk_words,
            "overlap_words": overlap_words,
            "max_workers": max_workers,
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


def stage02_generate_questions(run_name: str, model: str, resume: bool, max_workers: int = 1) -> None:
    out_dir = _stage_dir(run_name, "02_questions")
    out_path = out_dir / "questions_1t3f_keep_af.jsonl"
    af_rows = load_jsonl(_stage_dir(run_name, "01_selected_keep_af") / "selected_keep_af.jsonl")
    existing = load_jsonl(out_path) if resume and out_path.exists() else []
    questions = {str(q["af_id"]): q for q in existing if str(q.get("af_id", ""))}
    prompt = (PROMPTS_EXP2_DIR / "coverage_question_generation.txt").read_text(encoding="utf-8")

    def worker(af: dict[str, Any]) -> dict[str, Any]:
        client = OpenAIClient()
        return _generate_question(client, af, model, prompt)

    todo = [af for af in af_rows if not (resume and str(af["af_id"]) in questions)]
    for q in _run_parallel(todo, worker, max_workers=max_workers, desc="stage02 | build 1T3F questions"):
        questions[str(q["af_id"])] = q
    out_rows = list(questions.values())
    out_rows.sort(key=lambda r: (str(r["article_id"]), str(r["af_id"])))
    save_jsonl(out_rows, out_path)

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "max_workers": max_workers,
            "prompt_source": str(PROMPTS_EXP2_DIR / "coverage_question_generation.txt"),
            "af_total": len(af_rows),
            "questions": len(questions),
        },
        out_dir / "question_metadata.json",
    )
    print(f"[stage02] questions={len(questions)} -> {out_dir}")


def _span_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _build_evidence_packets(article: dict[str, Any], facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    article_text = str(article.get("document", ""))
    article_norm = _norm(article_text)
    candidates: list[dict[str, Any]] = []
    for af in facts:
        span = str(af.get("source_span", "")).strip() or str(af.get("fact", "")).strip()
        tokens = _span_tokens(span)
        norm_span = _norm(span)
        candidates.append(
            {
                "span": span,
                "tokens": tokens,
                "article_position": article_norm.find(norm_span),
                "af_ids": [str(af["af_id"])],
                "facts": [str(af["fact"])],
            }
        )

    packets: list[dict[str, Any]] = []
    for cand in candidates:
        best_idx = -1
        best_overlap = 0.0
        for i, packet in enumerate(packets):
            union = cand["tokens"] | packet["tokens"]
            overlap = len(cand["tokens"] & packet["tokens"]) / len(union) if union else 0.0
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        if best_idx >= 0 and best_overlap >= 0.72:
            packet = packets[best_idx]
            packet["af_ids"].extend(cand["af_ids"])
            packet["facts"].extend(cand["facts"])
            if len(cand["span"].split()) > len(packet["span"].split()):
                packet["span"] = cand["span"]
                packet["tokens"] = cand["tokens"]
            positions = [p for p in [packet["article_position"], cand["article_position"]] if p >= 0]
            packet["article_position"] = min(positions) if positions else -1
        else:
            packets.append(dict(cand))

    packets.sort(key=lambda p: (p["article_position"] < 0, p["article_position"], p["af_ids"][0]))
    out: list[dict[str, Any]] = []
    for i, packet in enumerate(packets, start=1):
        out.append(
            {
                "evidence_id": f"E{i:03d}",
                "source_span": packet["span"],
                "af_ids": sorted(set(packet["af_ids"])),
                "atomic_facts": list(dict.fromkeys(packet["facts"])),
                "article_position": packet["article_position"],
            }
        )
    return out


def _format_evidence_packets(packets: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"{p['evidence_id']}\n"
        f"ARTICLE EVIDENCE: {p['source_span']}\n"
        f"RELATED FINAL KEEP AFS: " + " | ".join(p["atomic_facts"])
        for p in packets
    )


def stage03_generate_initial_summaries(
    run_name: str,
    model: str,
    resume: bool,
    *,
    model_key: str | None = None,
    max_workers: int = 1,
    usage: UsageTracker | None = None,
) -> None:
    articles = _load_articles(run_name)
    af_rows = load_jsonl(_stage_dir(run_name, "01_selected_keep_af") / "selected_keep_af.jsonl")
    facts_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for af in af_rows:
        facts_by_article[str(af["article_id"])].append(af)
    out_dir = _stage_dir(run_name, "03_summaries")
    out_path = out_dir / "summaries_iter0.jsonl"
    evidence_path = out_dir / "evidence_packets_iter0.jsonl"
    prompt = (PROMPTS_EXP3_DIR / "summary_generation.txt").read_text(encoding="utf-8")
    rows = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("article_id")) for r in rows}
    evidence_by_article = {
        str(art["id"]): _build_evidence_packets(art, facts_by_article.get(str(art["id"]), []))
        for art in articles
    }
    save_jsonl(
        [
            {
                "article_id": art["id"],
                "source_dataset": art["source_dataset"],
                "original_index": art["original_index"],
                "evidence_packets": evidence_by_article[str(art["id"])],
            }
            for art in articles
        ],
        evidence_path,
    )

    def worker(art: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        article_id = str(art["id"])
        client = ChatRunner(model=model, model_key=model_key, run_name=run_name, usage=usage)
        raw = ""
        summary = ""
        parse_error = False
        failed: dict[str, Any] | None = None
        try:
            target_word_count = _target_summary_word_count(art)
            packets = evidence_by_article[article_id]
            valid_evidence_ids = {str(p["evidence_id"]) for p in packets}
            raw = client.chat(
                stage=f"stage03_initial_summary:{_model_label(model, model_key)}",
                item_id=article_id,
                messages=[
                    {
                        "role": "user",
                        "content": prompt.replace("{article}", str(art["document"]))
                        .replace("{evidence_packets}", _format_evidence_packets(packets))
                        .replace("{target_word_count}", str(target_word_count)),
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = json.loads(raw)
            raw_sentences = obj.get("sentences", [])
            if not isinstance(raw_sentences, list) or not raw_sentences:
                raise ValueError("sentences must be a non-empty list")
            sentence_rows: list[dict[str, Any]] = []
            for entry in raw_sentences:
                if not isinstance(entry, dict):
                    raise ValueError("each sentence must be an object")
                text = str(entry.get("text", "")).strip()
                raw_evidence_ids = entry.get("evidence_ids", entry.get("evidence_id", []))
                if isinstance(raw_evidence_ids, str):
                    evidence_ids = [raw_evidence_ids.strip()] if raw_evidence_ids.strip() else []
                elif isinstance(raw_evidence_ids, list):
                    evidence_ids = list(dict.fromkeys(str(x).strip() for x in raw_evidence_ids if str(x).strip()))
                else:
                    evidence_ids = []
                section = str(entry.get("section", "")).strip().lower()
                if not text or not evidence_ids or any(eid not in valid_evidence_ids for eid in evidence_ids):
                    raise ValueError(f"invalid sentence evidence mapping: {evidence_ids}")
                if section not in {"background", "methods", "results", "implications"}:
                    raise ValueError(f"invalid section: {section}")
                sentence_rows.append({"text": text, "evidence_ids": evidence_ids, "section": section})
            summary = " ".join(r["text"] for r in sentence_rows)
        except Exception as exc:
            parse_error = True
            failed = {"article_id": article_id, "error": str(exc), "raw_preview": raw[:300]}
            summary = raw.strip()
            sentence_rows = []
        return (
            {
                "article_id": article_id,
                "source_dataset": art["source_dataset"],
                "original_index": art["original_index"],
                "iteration": 0,
                "summary": summary,
                "summary_word_count": _word_count(summary),
                "target_word_count": _target_summary_word_count(art),
                "generation_mode": "af_guided_full_article_context",
                "evidence_packet_count": len(evidence_by_article[article_id]),
                "sentence_evidence": sentence_rows,
                "parse_error": parse_error,
            },
            failed,
        )

    todo = [art for art in articles if not (resume and str(art["id"]) in done)]
    results = _run_parallel(todo, worker, max_workers=max_workers, desc="stage03 | initial summaries")
    failed = [f for _, f in results if f is not None]
    rows.extend([r for r, _ in results])
    rows.sort(key=lambda r: (str(r["source_dataset"]), int(r["original_index"]), str(r["article_id"])))
    save_jsonl(rows, out_path)

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": _model_label(model, model_key),
            "max_workers": max_workers,
            "generation_mode": "evidence_first",
            "evidence_packets": sum(len(v) for v in evidence_by_article.values()),
            "summaries": len(rows),
            "failed": failed,
        },
        out_dir / "summaries_iter0_metadata.json",
    )
    if failed:
        raise RuntimeError(f"initial summary generation failed for {len(failed)} articles; rerun with resume")
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


def _full_summary_context(
    *,
    article_id: str,
    summary: str,
    facts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunk = {
        "article_id": article_id,
        "chunk_id": f"{article_id}_summary_full",
        "chunk_idx": 0,
        "start_word": 0,
        "end_word": _word_count(summary),
        "chunk_text": str(summary or "").strip(),
    }
    chunks = [chunk] if chunk["chunk_text"] else []
    retrieval_rows = [
        {
            "af_id": af["af_id"],
            "article_id": article_id,
            "top_k": 1,
            "context_mode": "full_summary",
            "retrieved_chunks": [
                {
                    "rank": 1,
                    "chunk_id": chunk["chunk_id"],
                    "score": 1.0,
                    "chunk_text": chunk["chunk_text"],
                }
            ]
            if chunks
            else [],
        }
        for af in facts
    ]
    return chunks, retrieval_rows


def _summaries_for_iteration(run_name: str, iteration: int) -> dict[str, dict[str, Any]]:
    path = _stage_dir(run_name, "03_summaries") / f"summaries_iter{iteration}.jsonl"
    if not path.exists():
        return {}
    return {str(r["article_id"]): r for r in load_jsonl(path)}


def _target_summary_word_count(article: dict[str, Any]) -> int:
    target_text = str(article.get("reference") or article.get("expert_summary") or "")
    return _word_count(target_text) or 200


def _split_summary_sentences(summary: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(summary or "").strip())
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[\"'A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def _numbered_sentences(sentences: list[str]) -> str:
    return "\n".join(f"[{i}] {sentence}" for i, sentence in enumerate(sentences))


def stage04_check_iteration(
    run_name: str,
    model: str,
    iteration: int,
    top_k: int,
    chunk_words: int,
    overlap_words: int,
    embed_model: str,
    resume: bool,
    *,
    model_key: str | None = None,
    max_workers: int = 1,
    only_previous_errors: bool = False,
    context_mode: str = DEFAULT_CONTEXT_MODE,
    usage: UsageTracker | None = None,
) -> None:
    if context_mode not in {"full_summary", "embedding"}:
        raise ValueError("context_mode must be 'full_summary' or 'embedding'")
    if context_mode == "embedding" and not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing. Required for summary chunk retrieval embeddings.")
    out_dir = _stage_dir(run_name, f"04_checks_iter{iteration}")
    out_path = out_dir / "coverage_predictions.jsonl"
    out_retrieval = out_dir / "retrieval_topk.jsonl"
    out_chunks = out_dir / "summary_chunks.jsonl"
    af_rows = load_jsonl(_stage_dir(run_name, "01_selected_keep_af") / "selected_keep_af.jsonl")
    questions = {str(q["af_id"]): q for q in load_jsonl(_stage_dir(run_name, "02_questions") / "questions_1t3f_keep_af.jsonl")}
    if only_previous_errors and iteration > 0:
        prev_path = _stage_dir(run_name, f"04_checks_iter{iteration - 1}") / "coverage_predictions.jsonl"
        prev_rows = load_jsonl(prev_path) if prev_path.exists() else []
        active_af_ids = {str(r["af_id"]) for r in prev_rows if not r.get("covered")}
        af_rows = [r for r in af_rows if str(r.get("af_id")) in active_af_ids]
    summaries = _summaries_for_iteration(run_name, iteration)
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for af in af_rows:
        by_article[str(af["article_id"])].append(af)

    rows = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("af_id")) for r in rows}
    all_chunks: list[dict[str, Any]] = []
    all_retrieval: list[dict[str, Any]] = []
    setup_c_system = SETUP_C_PROMPT.read_text(encoding="utf-8")

    def worker(item: tuple[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
        article_id, summ = item
        facts = by_article.get(article_id, [])
        if context_mode == "full_summary":
            chunks, retrieval = _full_summary_context(
                article_id=article_id,
                summary=str(summ["summary"]),
                facts=facts,
            )
        else:
            chunks, retrieval = _retrieve_summary_chunks(
                article_id=article_id,
                summary=str(summ["summary"]),
                facts=facts,
                top_k=top_k,
                chunk_words=chunk_words,
                overlap_words=overlap_words,
                embed_model=embed_model,
            )
        retr_map = {str(r["af_id"]): r for r in retrieval}
        client = ChatRunner(model=model, model_key=model_key, run_name=run_name, usage=usage)
        out_rows: list[dict[str, Any]] = []
        parse_errors = 0
        for af in facts:
            af_id = str(af["af_id"])
            if resume and af_id in done:
                continue
            q = questions[af_id]
            opts = _normalize_options(q)
            chunks_txt = "\n\n".join([f"[{c['rank']}] {c['chunk_text']}" for c in retr_map.get(af_id, {}).get("retrieved_chunks", [])])
            user_msg = (
                "Context Chunks:\n"
                f"{chunks_txt}\n\n"
                "Question: Which option is supported by the context under the SAME strict policy as direct claim checking?\n"
                f"A. {opts['A']}\n"
                f"B. {opts['B']}\n"
                f"C. {opts['C']}\n"
                f"D. {opts['D']}\n"
                f"E. {opts['E']}\n"
            )
            option_eval: dict[str, dict[str, Any]] = {}
            pred_letter = "E"
            reason = ""
            parse_error = False
            try:
                raw = client.chat(
                    stage=f"stage04_check_iter{iteration}:{_model_label(model, model_key)}",
                    item_id=af_id,
                    messages=[
                        {"role": "system", "content": setup_c_system},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                obj = json.loads(raw)
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
            out_rows.append(
                {
                    "af_id": af_id,
                    "article_id": article_id,
                    "source_dataset": af["source_dataset"],
                    "iteration": iteration,
                    "fact": af["fact"],
                    "source_span": af.get("source_span", ""),
                    "question_id": q["question_id"],
                    "options": opts,
                    "correct_letter": q["correct_letter"],
                    "correct_answer": opts.get(str(q["correct_letter"]), ""),
                    "predicted_letter": pred_letter,
                    "predicted_answer": opts.get(pred_letter, ""),
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
        return chunks, retrieval, out_rows, parse_errors

    summary_items = [(article_id, summ) for article_id, summ in summaries.items() if by_article.get(article_id)]
    results = _run_parallel(summary_items, worker, max_workers=max_workers, desc=f"stage04 | check iter {iteration}")
    parse_errors_this_run = 0
    for chunks, retrieval, new_rows, parse_errors in results:
        all_chunks.extend(chunks)
        all_retrieval.extend(retrieval)
        rows.extend(new_rows)
        parse_errors_this_run += parse_errors
    rows.sort(key=lambda r: (str(r["article_id"]), str(r["af_id"]), int(r["iteration"])))
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
            "model": _model_label(model, model_key),
            "iteration": iteration,
            "setup": "Experiment 1 Setup C 1T3F + E(None), fair strict policy",
            "context_mode": context_mode,
            "only_previous_errors": only_previous_errors,
            "max_workers": max_workers,
            "top_k": top_k,
            "summary_chunk_words": chunk_words,
            "summary_overlap_words": overlap_words,
            "embedding_model": embed_model,
            "af_total": total,
            "covered": covered_count,
            "errors": total - covered_count,
            "coverage_rate": round(covered_count / total, 4) if total else 0.0,
            "parse_error_count": sum(1 for r in rows if r.get("parse_error")),
            "parse_errors_this_run": parse_errors_this_run,
            "by_article": by_article_metrics,
        },
        out_dir / "coverage_summary.json",
    )
    print(f"[stage04] iter={iteration} covered={covered_count}/{total} -> {out_dir}")


def stage05_rewrite_iteration(
    run_name: str,
    model: str,
    from_iteration: int,
    resume: bool,
    *,
    model_key: str | None = None,
    max_workers: int = 1,
    usage: UsageTracker | None = None,
) -> None:
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

    def worker(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        article_id, summ = item
        if resume and article_id in done:
            return {}
        errors = by_article_errors.get(article_id, [])
        if not errors:
            return {
                **summ,
                "iteration": from_iteration + 1,
                "summary": summ["summary"],
                "summary_word_count": _word_count(str(summ["summary"])),
                "target_word_count": _target_summary_word_count(articles[article_id]),
                "rewrite_skipped_no_errors": True,
            }
        feedback = "\n".join(
            (
                f"- Missing/incorrect fact {i + 1}: {e['fact']}\n"
                f"  Article evidence: {e.get('source_span', '')}\n"
                f"  Correct answer should be {e.get('correct_letter')}: {e.get('correct_answer', '')}\n"
                f"  The checker selected {e.get('predicted_letter')}: {e.get('predicted_answer', '')}\n"
                f"  Checker reason: {e.get('reason', '')}"
            )
            for i, e in enumerate(errors)
        )
        client = ChatRunner(model=model, model_key=model_key, run_name=run_name, usage=usage)
        raw = ""
        new_summary = ""
        revision_notes: list[str] = []
        parse_error = False
        try:
            target_word_count = _target_summary_word_count(articles[article_id])
            raw = client.chat(
                stage=f"stage05_rewrite_iter{from_iteration + 1}:{_model_label(model, model_key)}",
                item_id=article_id,
                messages=[
                    {
                        "role": "user",
                        "content": prompt.replace("{article}", str(articles[article_id]["document"]))
                        .replace("{summary}", str(summ["summary"]))
                        .replace("{feedback}", feedback)
                        .replace("{target_word_count}", str(target_word_count)),
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
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
        return {
            "article_id": article_id,
            "source_dataset": summ["source_dataset"],
            "original_index": summ["original_index"],
            "iteration": from_iteration + 1,
            "summary": new_summary,
            "summary_word_count": _word_count(new_summary),
            "target_word_count": _target_summary_word_count(articles[article_id]),
            "feedback_error_count": len(errors),
            "feedback_mode": "only_current_error_af_with_correct_answer",
            "revision_notes": revision_notes,
            "parse_error": parse_error,
        }

    new_rows = [
        r
        for r in _run_parallel(list(current.items()), worker, max_workers=max_workers, desc=f"stage05 | rewrite iter {from_iteration + 1}")
        if r
    ]
    rows.extend(new_rows)
    rows.sort(key=lambda r: (str(r["source_dataset"]), int(r["original_index"]), str(r["article_id"])))
    save_jsonl(rows, out_path)
    print(f"[stage05] wrote summaries_iter{from_iteration + 1}.jsonl")


def stage05_sentence_factuality_check(
    run_name: str,
    model: str,
    iteration: int,
    resume: bool,
    *,
    model_key: str | None = None,
    max_workers: int = 1,
    usage: UsageTracker | None = None,
) -> None:
    articles = {str(r["id"]): r for r in _load_articles(run_name)}
    summaries = _summaries_for_iteration(run_name, iteration)
    out_dir = _stage_dir(run_name, f"05_sentence_checks_iter{iteration}")
    out_path = out_dir / "sentence_factuality.jsonl"
    rows = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r["article_id"]) for r in rows if not r.get("parse_error")}
    prompt = (PROMPTS_EXP3_DIR / "sentence_factuality_check.txt").read_text(encoding="utf-8")

    def worker(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        article_id, summ = item
        sentences = _split_summary_sentences(str(summ["summary"]))
        client = ChatRunner(model=model, model_key=model_key, run_name=run_name, usage=usage)
        raw = ""
        parse_error = False
        error = ""
        evaluations: list[dict[str, Any]] = []
        try:
            raw = client.chat(
                stage=f"stage05_sentence_check_iter{iteration}:{_model_label(model, model_key)}",
                item_id=article_id,
                messages=[
                    {
                        "role": "user",
                        "content": prompt.replace("{article}", str(articles[article_id]["document"])).replace(
                            "{numbered_sentences}", _numbered_sentences(sentences)
                        ),
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = json.loads(raw)
            raw_evals = obj.get("sentence_evaluations", [])
            if not isinstance(raw_evals, list):
                raise ValueError("sentence_evaluations must be a list")
            by_index: dict[int, dict[str, Any]] = {}
            for entry in raw_evals:
                if not isinstance(entry, dict):
                    continue
                idx = int(entry.get("sentence_index", -1))
                verdict = str(entry.get("verdict", "")).strip().lower()
                if 0 <= idx < len(sentences) and verdict in {"supported", "partial", "unsupported", "contradicted"}:
                    by_index[idx] = {
                        "sentence_index": idx,
                        "sentence": sentences[idx],
                        "verdict": verdict,
                        "evidence": str(entry.get("evidence", "")).strip(),
                        "reason": str(entry.get("reason", "")).strip(),
                    }
            if len(by_index) != len(sentences):
                missing = sorted(set(range(len(sentences))) - set(by_index))
                raise ValueError(f"missing sentence evaluations: {missing}")
            evaluations = [by_index[i] for i in range(len(sentences))]
        except Exception as exc:
            parse_error = True
            error = str(exc)
        return {
            "article_id": article_id,
            "source_dataset": summ["source_dataset"],
            "original_index": summ["original_index"],
            "iteration": iteration,
            "sentence_count": len(sentences),
            "sentence_evaluations": evaluations,
            "problematic_count": sum(1 for e in evaluations if e["verdict"] != "supported"),
            "parse_error": parse_error,
            "error": error,
            "raw_preview": raw[:500] if parse_error else "",
        }

    todo = [(article_id, summ) for article_id, summ in summaries.items() if article_id not in done]
    new_rows = _run_parallel(todo, worker, max_workers=max_workers, desc=f"stage05 | sentence check iter {iteration}")
    new_ids = {str(r["article_id"]) for r in new_rows}
    rows = [r for r in rows if str(r["article_id"]) not in new_ids] + new_rows
    rows.sort(key=lambda r: (str(r["source_dataset"]), int(r["original_index"]), str(r["article_id"])))
    save_jsonl(rows, out_path)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": _model_label(model, model_key),
            "iteration": iteration,
            "max_workers": max_workers,
            "articles": len(rows),
            "sentences": sum(int(r.get("sentence_count", 0)) for r in rows),
            "problematic_sentences": sum(int(r.get("problematic_count", 0)) for r in rows),
            "parse_error_count": sum(1 for r in rows if r.get("parse_error")),
        },
        out_dir / "sentence_factuality_summary.json",
    )
    parse_errors = [r for r in rows if r.get("parse_error")]
    if parse_errors:
        raise RuntimeError(f"sentence factuality check has {len(parse_errors)} parse/runtime errors; rerun with resume")
    print(
        f"[stage05] sentence check iter={iteration} problematic="
        f"{sum(int(r.get('problematic_count', 0)) for r in rows)}"
    )


def stage05_minimal_sentence_repair(
    run_name: str,
    model: str,
    iteration: int,
    resume: bool,
    *,
    model_key: str | None = None,
    max_workers: int = 1,
    usage: UsageTracker | None = None,
) -> None:
    articles = {str(r["id"]): r for r in _load_articles(run_name)}
    summaries_path = _stage_dir(run_name, "03_summaries") / f"summaries_iter{iteration}.jsonl"
    backup_path = _stage_dir(run_name, "03_summaries") / f"summaries_iter{iteration}_pre_factuality.jsonl"
    if backup_path.exists():
        source_rows = load_jsonl(backup_path)
    else:
        source_rows = load_jsonl(summaries_path)
        save_jsonl(source_rows, backup_path)
    summaries = {str(r["article_id"]): r for r in source_rows}
    check_path = _stage_dir(run_name, f"05_sentence_checks_iter{iteration}") / "sentence_factuality.jsonl"
    checks = {str(r["article_id"]): r for r in load_jsonl(check_path)}
    out_dir = _stage_dir(run_name, f"05_sentence_repairs_iter{iteration}")
    out_path = out_dir / "sentence_repairs.jsonl"
    rows = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r["article_id"]) for r in rows if not r.get("parse_error")}
    prompt = (PROMPTS_EXP3_DIR / "minimal_sentence_repair.txt").read_text(encoding="utf-8")

    def worker(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        article_id, summ = item
        sentences = _split_summary_sentences(str(summ["summary"]))
        evaluations = checks[article_id].get("sentence_evaluations", [])
        problematic = [e for e in evaluations if str(e.get("verdict")) != "supported"]
        if not problematic:
            return {
                **summ,
                "sentence_repair_iteration": iteration,
                "problematic_sentence_count": 0,
                "replacements": [],
                "parse_error": False,
            }
        feedback = "\n".join(
            f"[{e['sentence_index']}] verdict={e['verdict']}\n"
            f"Sentence: {e['sentence']}\nEvidence: {e.get('evidence', '')}\nReason: {e.get('reason', '')}"
            for e in problematic
        )
        client = ChatRunner(model=model, model_key=model_key, run_name=run_name, usage=usage)
        raw = ""
        parse_error = False
        error = ""
        replacements: list[dict[str, Any]] = []
        repaired_sentences = list(sentences)
        try:
            raw = client.chat(
                stage=f"stage05_sentence_repair_iter{iteration}:{_model_label(model, model_key)}",
                item_id=article_id,
                messages=[
                    {
                        "role": "user",
                        "content": prompt.replace("{article}", str(articles[article_id]["document"]))
                        .replace("{numbered_sentences}", _numbered_sentences(sentences))
                        .replace("{factuality_feedback}", feedback),
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = json.loads(raw)
            raw_replacements = obj.get("replacements", [])
            if not isinstance(raw_replacements, list):
                raise ValueError("replacements must be a list")
            allowed_indices = {int(e["sentence_index"]) for e in problematic}
            seen_indices: set[int] = set()
            for entry in raw_replacements:
                if not isinstance(entry, dict):
                    continue
                idx = int(entry.get("sentence_index", -1))
                action = str(entry.get("action", "")).strip().lower()
                replacement = str(entry.get("replacement", "")).strip()
                if idx not in allowed_indices or idx in seen_indices or action not in {"replace", "delete"}:
                    continue
                if action == "replace" and not replacement:
                    raise ValueError(f"empty replacement for sentence {idx}")
                repaired_sentences[idx] = replacement if action == "replace" else ""
                seen_indices.add(idx)
                replacements.append(
                    {
                        "sentence_index": idx,
                        "action": action,
                        "original": sentences[idx],
                        "replacement": replacement,
                        "reason": str(entry.get("reason", "")).strip(),
                    }
                )
            if seen_indices != allowed_indices:
                missing = sorted(allowed_indices - seen_indices)
                raise ValueError(f"missing repairs for problematic sentences: {missing}")
        except Exception as exc:
            parse_error = True
            error = str(exc)
            repaired_sentences = sentences
        repaired_summary = " ".join(s for s in repaired_sentences if s).strip()
        return {
            "article_id": article_id,
            "source_dataset": summ["source_dataset"],
            "original_index": summ["original_index"],
            "iteration": iteration,
            "summary": repaired_summary,
            "summary_word_count": _word_count(repaired_summary),
            "target_word_count": _target_summary_word_count(articles[article_id]),
            "sentence_repair_iteration": iteration,
            "problematic_sentence_count": len(problematic),
            "replacements": replacements,
            "parse_error": parse_error,
            "error": error,
            "raw_preview": raw[:500] if parse_error else "",
        }

    todo = [(article_id, summ) for article_id, summ in summaries.items() if article_id not in done]
    new_rows = _run_parallel(todo, worker, max_workers=max_workers, desc=f"stage05 | sentence repair iter {iteration}")
    new_ids = {str(r["article_id"]) for r in new_rows}
    rows = [r for r in rows if str(r["article_id"]) not in new_ids] + new_rows
    rows.sort(key=lambda r: (str(r["source_dataset"]), int(r["original_index"]), str(r["article_id"])))
    save_jsonl(rows, out_path)
    save_jsonl(rows, summaries_path)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": _model_label(model, model_key),
            "iteration": iteration,
            "max_workers": max_workers,
            "articles": len(rows),
            "problematic_sentences": sum(int(r.get("problematic_sentence_count", 0)) for r in rows),
            "repaired_or_deleted_sentences": sum(len(r.get("replacements", [])) for r in rows),
            "parse_error_count": sum(1 for r in rows if r.get("parse_error")),
        },
        out_dir / "sentence_repair_summary.json",
    )
    parse_errors = [r for r in rows if r.get("parse_error")]
    if parse_errors:
        raise RuntimeError(f"sentence repair has {len(parse_errors)} parse/runtime errors; rerun with resume")
    print(
        f"[stage05] sentence repair iter={iteration} repaired="
        f"{sum(len(r.get('replacements', [])) for r in rows)}"
    )


def stage06_build_submission_files(run_name: str, final_iteration: int) -> None:
    articles = _load_articles(run_name)
    initial_raw_path = _stage_dir(run_name, "03_summaries") / "summaries_iter0_pre_factuality.jsonl"
    initial = (
        {str(r["article_id"]): r for r in load_jsonl(initial_raw_path)}
        if initial_raw_path.exists()
        else _summaries_for_iteration(run_name, 0)
    )
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
            "initial_variant": "evidence_first_pre_factuality_repair" if initial_raw_path.exists() else "iteration_0",
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
    model_key = args.model_key
    usage = _usage(args.run_name, resume=not args.no_resume) if model_key else None
    if model_key and model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}. Valid keys include: {sorted(MODEL_CONFIGS)}")
    if args.module2_run_name:
        module2_model_key = args.module2_model_key or model_key
        if not module2_model_key:
            raise ValueError("--module2-model-key is required when --module2-run-name is used without --model-key")
        stage00_import_module2_artifacts(args.run_name, args.module2_run_name, module2_model_key, args.n_per_source)
    else:
        stage00_prepare_inputs(args.run_name, args.split, args.n_per_source, args.seed)
        stage01_extract_keep_af(
            args.run_name,
            args.model,
            args.af_chunk_words,
            args.af_overlap_words,
            resume=not args.no_resume,
            max_workers=args.max_workers,
        )
        stage02_generate_questions(args.run_name, args.model, resume=not args.no_resume, max_workers=args.max_workers)
    stage03_generate_initial_summaries(
        args.run_name,
        args.model,
        resume=not args.no_resume,
        model_key=model_key,
        max_workers=args.max_workers,
        usage=usage,
    )
    if usage is not None:
        _save_usage(args.run_name, usage)
    stage05_sentence_factuality_check(
        args.run_name,
        args.model,
        0,
        resume=not args.no_resume,
        model_key=model_key,
        max_workers=args.max_workers,
        usage=usage,
    )
    if usage is not None:
        _save_usage(args.run_name, usage)
    stage05_minimal_sentence_repair(
        args.run_name,
        args.model,
        0,
        resume=not args.no_resume,
        model_key=model_key,
        max_workers=args.max_workers,
        usage=usage,
    )
    if usage is not None:
        _save_usage(args.run_name, usage)

    if args.initial_only:
        stage06_build_submission_files(args.run_name, 0)
        stage07_summarize_run(args.run_name, 0)
        return

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
            model_key=model_key,
            max_workers=args.max_workers,
            only_previous_errors=False,
            context_mode=args.context_mode,
            usage=usage,
        )
        if usage is not None:
            _save_usage(args.run_name, usage)
        summary_path = _stage_dir(args.run_name, f"04_checks_iter{iteration}") / "coverage_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        final_iteration = iteration
        if int(summary.get("errors", 0)) == 0:
            break
        if iteration < args.max_rewrites:
            stage05_rewrite_iteration(
                args.run_name,
                args.model,
                iteration,
                resume=not args.no_resume,
                model_key=model_key,
                max_workers=args.max_workers,
                usage=usage,
            )
            if usage is not None:
                _save_usage(args.run_name, usage)
            next_iteration = iteration + 1
            stage05_sentence_factuality_check(
                args.run_name,
                args.model,
                next_iteration,
                resume=not args.no_resume,
                model_key=model_key,
                max_workers=args.max_workers,
                usage=usage,
            )
            if usage is not None:
                _save_usage(args.run_name, usage)
            stage05_minimal_sentence_repair(
                args.run_name,
                args.model,
                next_iteration,
                resume=not args.no_resume,
                model_key=model_key,
                max_workers=args.max_workers,
                usage=usage,
            )
            if usage is not None:
                _save_usage(args.run_name, usage)

    stage06_build_submission_files(args.run_name, final_iteration)
    stage07_summarize_run(args.run_name, final_iteration)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3 AF-guided iterative lay summarization pipeline.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run-name", type=str, default="pilot_n20_gpt41_mini")
        p.add_argument("--model", type=str, default=DEFAULT_MODEL)
        p.add_argument("--model-key", type=str, default=None, help="Optional MODEL_CONFIGS key for non-OpenAI providers, e.g. gemini3_flash_preview_minimal.")
        p.add_argument("--max-workers", type=int, default=1)
        p.add_argument("--no-resume", action="store_true")

    p_all = sub.add_parser("run_all")
    add_common(p_all)
    p_all.add_argument("--split", type=str, default="test")
    p_all.add_argument("--n-per-source", type=int, default=10, help="Use 10 for a 20-article pilot; omit via stage00 for full 142+142.")
    p_all.add_argument("--seed", type=int, default=20260611)
    p_all.add_argument("--max-rewrites", type=int, default=2)
    p_all.add_argument(
        "--initial-only",
        action="store_true",
        help="Stop after initial sentence factuality checking and minimal repair; skip AF coverage checks and rewrites.",
    )
    p_all.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p_all.add_argument("--summary-chunk-words", type=int, default=DEFAULT_SUMMARY_CHUNK_WORDS)
    p_all.add_argument("--summary-overlap-words", type=int, default=DEFAULT_SUMMARY_OVERLAP_WORDS)
    p_all.add_argument("--af-chunk-words", type=int, default=DEFAULT_AF_CHUNK_WORDS)
    p_all.add_argument("--af-overlap-words", type=int, default=DEFAULT_AF_OVERLAP_WORDS)
    p_all.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL)
    p_all.add_argument("--context-mode", choices=["full_summary", "embedding"], default=DEFAULT_CONTEXT_MODE)
    p_all.add_argument("--module2-run-name", type=str, default=None, help="Import simulated Module 2 final keep AFs/questions from this run.")
    p_all.add_argument("--module2-model-key", type=str, default=None, help="Module 2 artifact branch to import. Defaults to --model-key.")

    p0 = sub.add_parser("stage00_prepare_inputs")
    p0.add_argument("--run-name", type=str, default="pilot_n20_gpt41_mini")
    p0.add_argument("--split", type=str, default="test")
    p0.add_argument("--n-per-source", type=int, default=10)
    p0.add_argument("--full-test", action="store_true")
    p0.add_argument("--seed", type=int, default=20260611)

    p0m = sub.add_parser("stage00_import_module2_artifacts")
    p0m.add_argument("--run-name", type=str, default="pilot_module2_gemini3_flash_preview")
    p0m.add_argument("--module2-run-name", type=str, required=True)
    p0m.add_argument("--module2-model-key", type=str, required=True)
    p0m.add_argument("--n-per-source", type=int, default=5)

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
    p4.add_argument("--context-mode", choices=["full_summary", "embedding"], default=DEFAULT_CONTEXT_MODE)
    p4.add_argument("--only-previous-errors", action="store_true")

    p5 = sub.add_parser("stage05_rewrite_iteration")
    add_common(p5)
    p5.add_argument("--from-iteration", type=int, required=True)

    p5s = sub.add_parser("stage05_sentence_factuality_check")
    add_common(p5s)
    p5s.add_argument("--iteration", type=int, required=True)

    p5r = sub.add_parser("stage05_minimal_sentence_repair")
    add_common(p5r)
    p5r.add_argument("--iteration", type=int, required=True)

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
    elif args.cmd == "stage00_import_module2_artifacts":
        stage00_import_module2_artifacts(args.run_name, args.module2_run_name, args.module2_model_key, args.n_per_source)
    elif args.cmd == "stage01_extract_keep_af":
        stage01_extract_keep_af(
            args.run_name,
            args.model,
            args.af_chunk_words,
            args.af_overlap_words,
            resume=not args.no_resume,
            max_workers=args.max_workers,
        )
    elif args.cmd == "stage02_generate_questions":
        stage02_generate_questions(args.run_name, args.model, resume=not args.no_resume, max_workers=args.max_workers)
    elif args.cmd == "stage03_generate_initial_summaries":
        usage = _usage(args.run_name, resume=not args.no_resume) if args.model_key else None
        stage03_generate_initial_summaries(
            args.run_name,
            args.model,
            resume=not args.no_resume,
            model_key=args.model_key,
            max_workers=args.max_workers,
            usage=usage,
        )
        if usage is not None:
            _save_usage(args.run_name, usage)
    elif args.cmd == "stage04_check_iteration":
        usage = _usage(args.run_name, resume=not args.no_resume) if args.model_key else None
        stage04_check_iteration(
            args.run_name,
            args.model,
            args.iteration,
            args.top_k,
            args.summary_chunk_words,
            args.summary_overlap_words,
            args.embed_model,
            resume=not args.no_resume,
            model_key=args.model_key,
            max_workers=args.max_workers,
            only_previous_errors=args.only_previous_errors,
            context_mode=args.context_mode,
            usage=usage,
        )
        if usage is not None:
            _save_usage(args.run_name, usage)
    elif args.cmd == "stage05_rewrite_iteration":
        usage = _usage(args.run_name, resume=not args.no_resume) if args.model_key else None
        stage05_rewrite_iteration(
            args.run_name,
            args.model,
            args.from_iteration,
            resume=not args.no_resume,
            model_key=args.model_key,
            max_workers=args.max_workers,
            usage=usage,
        )
        if usage is not None:
            _save_usage(args.run_name, usage)
    elif args.cmd == "stage05_sentence_factuality_check":
        usage = _usage(args.run_name, resume=not args.no_resume) if args.model_key else None
        stage05_sentence_factuality_check(
            args.run_name,
            args.model,
            args.iteration,
            resume=not args.no_resume,
            model_key=args.model_key,
            max_workers=args.max_workers,
            usage=usage,
        )
        if usage is not None:
            _save_usage(args.run_name, usage)
    elif args.cmd == "stage05_minimal_sentence_repair":
        usage = _usage(args.run_name, resume=not args.no_resume) if args.model_key else None
        stage05_minimal_sentence_repair(
            args.run_name,
            args.model,
            args.iteration,
            resume=not args.no_resume,
            model_key=args.model_key,
            max_workers=args.max_workers,
            usage=usage,
        )
        if usage is not None:
            _save_usage(args.run_name, usage)
    elif args.cmd == "stage06_build_submission_files":
        stage06_build_submission_files(args.run_name, args.final_iteration)
    elif args.cmd == "stage07_summarize_run":
        stage07_summarize_run(args.run_name, args.max_iteration)


if __name__ == "__main__":
    main()
