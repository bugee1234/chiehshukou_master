from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.utils import OpenAIClient, load_jsonl, save_json, save_jsonl


DATA_1A_DIR = config.DATA_DIR / "experiment_1" / "1a"
RESULTS_1A_DIR = config.RESULTS_DIR / "experiment_1" / "1a"

DIR_00 = DATA_1A_DIR / "00_registry"
DIR_01 = DATA_1A_DIR / "01_inputs"
DIR_02 = DATA_1A_DIR / "02_perturbation"
DIR_03 = DATA_1A_DIR / "03_rag"
DIR_04 = DATA_1A_DIR / "04_judgement"
DIR_05 = DATA_1A_DIR / "05_metrics"


def _ensure_dirs() -> None:
    for p in [DIR_00, DIR_01, DIR_02, DIR_03, DIR_04, DIR_05, RESULTS_1A_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def _word_count(text: str | None) -> int:
    return len(str(text or "").split())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def _save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    save_jsonl(rows, path)


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _harden_false_options(true_statement: str, false_texts: list[str], atomic_fact: str) -> list[str]:
    """Keep 3 hard distractors: unique, not equal to true, same-topic mutations."""
    true_norm = _norm_text(true_statement)
    picked: list[str] = []
    seen: set[str] = set()

    def add_candidate(text: str) -> None:
        t = re.sub(r"\s+", " ", str(text or "").strip())
        if not t:
            return
        n = _norm_text(t)
        if not n or n == true_norm or n in seen:
            return
        seen.add(n)
        picked.append(t)

    for t in false_texts:
        add_candidate(t)

    # Deterministic supplements to avoid trivial/empty generations.
    supplements = [
        _mutate_sentence(true_statement),
        _mutate_sentence(atomic_fact),
        _mutate_sentence("It is not true that " + true_statement),
        _mutate_sentence(true_statement.replace(" is ", " is not ", 1)),
        _mutate_sentence(true_statement.replace(" are ", " are not ", 1)),
    ]
    for t in supplements:
        add_candidate(t)
        if len(picked) >= 3:
            break

    while len(picked) < 3:
        add_candidate(f"{atomic_fact} (not supported)")
        if len(picked) < 3:
            add_candidate(f"{true_statement} (not supported)")

    return picked[:3]


def _load_pilot_article_ids(first_n: int) -> list[str]:
    if first_n <= 0:
        raise ValueError(f"pilot_first_n must be > 0, got {first_n}")
    stats_path = DIR_01 / "pilot_articles_stats.json"
    if not stats_path.exists():
        raise RuntimeError(f"{stats_path} not found. Please run stage01_prepare_inputs first.")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    all_ids = [str(x) for x in stats.get("article_ids", [])]
    if len(all_ids) < first_n:
        raise RuntimeError(f"pilot_articles_stats has only {len(all_ids)} ids, cannot take first {first_n}.")
    return all_ids[:first_n]


def stage00_registry(seed_pool: int, seed_pilot: int, n_full: int, n_pilot: int) -> None:
    _ensure_dirs()

    exp0_bank = config.RESULTS_DIR / "experiment_0" / "final_test_bank.jsonl"
    sampled_old = config.DATA_RAW_DIR / "sampled_articles.jsonl"

    out_pool = DIR_00 / "article_pool_excluding_exp0.jsonl"
    out_pilot = DIR_00 / "pilot_10_manifest.jsonl"
    out_meta = DIR_00 / "registry_metadata.json"

    exp0_article_ids: set[str] = set()
    for row in _load_jsonl(exp0_bank):
        exp0_article_ids.add(str(row["article_id"]))

    used_indices: dict[str, set[int]] = {"PLOS": set(), "eLife": set()}
    for row in _load_jsonl(sampled_old):
        rid = str(row["id"])
        if rid in exp0_article_ids:
            used_indices[str(row["source_dataset"])].add(int(row["original_index"]))

    specs = [
        ("PLOS", "BioLaySumm/BioLaySumm2025-PLOS", "plos"),
        ("eLife", "BioLaySumm/BioLaySumm2025-eLife", "elife"),
    ]

    rng = random.Random(seed_pool)
    records: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "seed_pool": seed_pool,
        "seed_pilot": seed_pilot,
        "n_per_source_full": n_full,
        "n_per_source_pilot": n_pilot,
        "used_indices_from_exp0": {k: len(v) for k, v in used_indices.items()},
        "datasets": {},
    }

    for source, ds_name, prefix in specs:
        ds = load_dataset(ds_name, split="validation", token=config.HF_TOKEN or None)
        cols = set(ds.column_names)
        article_col = "article" if "article" in cols else None
        lay_col = "lay_summary" if "lay_summary" in cols else ("summary" if "summary" in cols else None)
        abstract_col = "abstract" if "abstract" in cols else None
        if article_col is None or lay_col is None:
            raise RuntimeError(f"{source}: cannot find article/summary columns. columns={ds.column_names}")

        all_indices = list(range(len(ds)))
        candidates = [i for i in all_indices if i not in used_indices[source]]
        if len(candidates) < n_full:
            raise RuntimeError(f"{source}: candidate size {len(candidates)} < required {n_full}")
        picked = rng.sample(candidates, n_full)

        for i, idx in enumerate(picked, start=1):
            row = ds[int(idx)]
            article = str(row.get(article_col, ""))
            lay = str(row.get(lay_col, ""))
            abstract = str(row.get(abstract_col, "")) if abstract_col else ""
            records.append(
                {
                    "id": f"exp1_{prefix}_{i:03d}",
                    "source_dataset": source,
                    "original_index": int(idx),
                    "article": article,
                    "lay_summary": lay,
                    "abstract": abstract,
                    "article_word_count": _word_count(article),
                    "lay_summary_word_count": _word_count(lay),
                    "excluded_exp0_overlap": True,
                }
            )

        metadata["datasets"][source] = {
            "dataset_name": ds_name,
            "validation_size": len(ds),
            "excluded_count_from_exp0": len(used_indices[source]),
            "candidate_size_after_exclusion": len(candidates),
            "picked_indices": picked,
        }

    _save_jsonl(records, out_pool)

    rng_p = random.Random(seed_pilot)
    pilot_rows: list[dict[str, Any]] = []
    for source in ["PLOS", "eLife"]:
        subset = [r for r in records if r["source_dataset"] == source]
        selected = rng_p.sample(subset, n_pilot)
        for r in selected:
            pilot_rows.append(
                {"id": r["id"], "source_dataset": r["source_dataset"], "original_index": r["original_index"]}
            )
    _save_jsonl(pilot_rows, out_pilot)

    metadata["pool_size_total"] = len(records)
    metadata["pool_size_by_source"] = dict(Counter(r["source_dataset"] for r in records))
    metadata["pilot_size_total"] = len(pilot_rows)
    metadata["pilot_size_by_source"] = dict(Counter(r["source_dataset"] for r in pilot_rows))
    save_json(metadata, out_meta)
    print(f"[stage00] pool={len(records)} pilot={len(pilot_rows)} -> {DIR_00}")


def stage01_prepare_inputs() -> None:
    _ensure_dirs()
    pool = {r["id"]: r for r in _load_jsonl(DIR_00 / "article_pool_excluding_exp0.jsonl")}
    pilot = _load_jsonl(DIR_00 / "pilot_10_manifest.jsonl")
    rows = [pool[r["id"]] for r in pilot if r["id"] in pool]

    out_articles = DIR_01 / "pilot_articles.jsonl"
    out_stats = DIR_01 / "pilot_articles_stats.json"
    _save_jsonl(rows, out_articles)
    stats = {
        "total": len(rows),
        "by_source": dict(Counter(r["source_dataset"] for r in rows)),
        "avg_lay_summary_words": round(sum(r["lay_summary_word_count"] for r in rows) / max(len(rows), 1), 2),
        "avg_article_words": round(sum(r["article_word_count"] for r in rows) / max(len(rows), 1), 2),
        "article_ids": [r["id"] for r in rows],
    }
    save_json(stats, out_stats)
    print(f"[stage01] rows={len(rows)} -> {DIR_01}")


def stage02_extract_af(model: str) -> None:
    _ensure_dirs()
    in_path = DIR_01 / "pilot_articles.jsonl"
    prompt_path = config.PROMPTS_DIR / "af_extraction.txt"
    out_af = DIR_01 / "af_gold_for_1a.jsonl"
    out_meta = DIR_01 / "af_extraction_metadata.json"

    rows = _load_jsonl(in_path)
    prompt_template = prompt_path.read_text(encoding="utf-8")
    client = OpenAIClient()

    af_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    total_in = total_out = total_tok = 0
    afs_per_article: dict[str, int] = {}
    afs_per_source: Counter[str] = Counter()

    for item in rows:
        article_id = str(item["id"])
        source = str(item["source_dataset"])
        lay_summary = str(item["lay_summary"])
        prompt = prompt_template.replace("{source_text}", lay_summary)
        raw_content = ""
        try:
            resp = client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            usage = resp.get("usage", {})
            total_in += int(usage.get("prompt_tokens", 0))
            total_out += int(usage.get("completion_tokens", 0))
            total_tok += int(usage.get("total_tokens", 0))
            raw_content = str(resp.get("content", ""))
            parsed = json.loads(raw_content)
            af_list = parsed.get("atomic_facts", [])
            if not isinstance(af_list, list):
                af_list = []
            cnt = 0
            for af in af_list:
                fact = str(af.get("fact", "")).strip()
                src_sent = str(af.get("source_sentence", "")).strip()
                if not fact or not src_sent:
                    continue
                cnt += 1
                af_rows.append(
                    {
                        "af_id": f"{article_id}_af_{cnt:03d}",
                        "article_id": article_id,
                        "source_dataset": source,
                        "fact": fact,
                        "source_sentence": src_sent,
                        "lay_summary": lay_summary,
                    }
                )
            afs_per_article[article_id] = cnt
            afs_per_source[source] += cnt
        except Exception as exc:
            failed.append({"article_id": article_id, "error": str(exc), "raw_preview": raw_content[:300]})
            afs_per_article[article_id] = 0

    _save_jsonl(af_rows, out_af)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "articles_total": len(rows),
            "articles_failed": len(failed),
            "failed_items": failed,
            "total_afs": len(af_rows),
            "afs_per_article": afs_per_article,
            "afs_per_source": dict(afs_per_source),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_tok,
        },
        out_meta,
    )
    print(f"[stage02] AF={len(af_rows)} failed={len(failed)} -> {DIR_01}")


def _mutate_sentence(sentence: str) -> str:
    t = sentence.strip()
    if not t:
        return t
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", t)
    if m:
        v = float(m.group(1))
        new_v = round(v + 15.0, 1) if v < 80 else round(v - 15.0, 1)
        return t[: m.start(1)] + str(new_v) + t[m.end(1) :]
    m = re.search(r"\b(\d+(?:\.\d+)?)\b", t)
    if m:
        v = float(m.group(1))
        new_v = int(v * 2) if v >= 1 else round(v * 2, 2)
        return t[: m.start(1)] + str(new_v) + t[m.end(1) :]
    replacements = [
        (r"\bincrease(d|s)?\b", "decrease"),
        (r"\bdecrease(d|s)?\b", "increase"),
        (r"\bhigher\b", "lower"),
        (r"\blower\b", "higher"),
        (r"\bmore\b", "less"),
        (r"\bless\b", "more"),
    ]
    for pat, rep in replacements:
        if re.search(pat, t, flags=re.IGNORECASE):
            return re.sub(pat, rep, t, count=1, flags=re.IGNORECASE)
    return "It is not true that " + (t[0].lower() + t[1:] if t else t)


def _llm_subtle_error_mutation(
    client: OpenAIClient, fact: str, source_sentence: str, model: str
) -> str:
    prompt = (
        "Rewrite the following biomedical claim into a subtle but incorrect statement. "
        "Keep topic/entities similar, change only one key detail (number/direction/comparison/negation/qualifier). "
        "Do not add new entities. Return strict JSON.\n\n"
        f"Original fact: {fact}\n"
        f"Source sentence: {source_sentence}\n\n"
        'Return JSON: {"mutated_fact":"..."}'
    )
    try:
        resp = client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        obj = json.loads(str(resp.get("content", "")))
        return str(obj.get("mutated_fact", "")).strip()
    except Exception:
        return ""


def _llm_relation(
    client: OpenAIClient, premise: str, hypothesis: str, model: str
) -> str:
    prompt = (
        "Decide semantic relation from premise to hypothesis.\n"
        "Return one label: contradiction, entailment, or neutral.\n\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n\n"
        'Return JSON: {"relation":"contradiction|entailment|neutral"}'
    )
    try:
        resp = client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        obj = json.loads(str(resp.get("content", "")))
        rel = str(obj.get("relation", "")).strip().lower()
        if rel in {"contradiction", "entailment", "neutral"}:
            return rel
    except Exception:
        pass
    return "neutral"


def _llm_is_supported(
    client: OpenAIClient, claim: str, text: str, model: str
) -> bool:
    prompt = (
        "Decide whether the claim is supported by the text. "
        "Paraphrase counts as support. Return strict JSON.\n\n"
        f"Claim: {claim}\n\n"
        f"Text: {text}\n\n"
        'Return JSON: {"supported":0|1}'
    )
    try:
        resp = client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        obj = json.loads(str(resp.get("content", "")))
        return int(obj.get("supported", 0)) == 1
    except Exception:
        return False


def stage03_perturb(seed: int, ratio: float, model: str = "gpt-4.1-mini") -> None:
    _ensure_dirs()
    afs = _load_jsonl(DIR_01 / "af_gold_for_1a.jsonl")
    out_gt = DIR_02 / "gt_labels_1a.jsonl"
    out_sprime = DIR_02 / "perturbed_summary.jsonl"
    out_meta = DIR_02 / "perturbation_metadata.json"
    client = OpenAIClient()

    rng = random.Random(seed)
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in afs:
        by_article[str(r["article_id"])].append(r)

    gt_rows: list[dict[str, Any]] = []
    sprime_rows: list[dict[str, Any]] = []
    c = Counter()
    sprime_method_counter = Counter()
    sprime_method_by_article: dict[str, str] = {}
    error_mutation_method_counter = Counter()
    survival_check_counter = Counter()

    for article_id, items in by_article.items():
        source = str(items[0]["source_dataset"])
        original_summary = str(items[0]["lay_summary"])
        n = len(items)
        n_perturb = max(1, round(n * ratio))
        n_del = n_perturb // 2
        n_error = n_perturb - n_del
        shuffled = items[:]
        rng.shuffle(shuffled)
        del_targets = set(x["af_id"] for x in shuffled[:n_del])
        err_targets = set(x["af_id"] for x in shuffled[n_del : n_del + n_error])
        applied_del: list[str] = []
        applied_error: list[str] = []
        perturbed_facts_for_summary: list[str] = []
        applied_error_mutations: dict[str, str] = {}

        # AF-level perturbation: operate directly on each AF text.
        # This removes sentence-level coupling noise between multiple AFs sharing one source sentence.
        for af in items:
            af_id = str(af["af_id"])
            fact = str(af["fact"]).strip()
            if af_id in del_targets:
                applied_del.append(af_id)
                continue
            if af_id in err_targets:
                candidates: list[tuple[str, str]] = []
                subtle = _llm_subtle_error_mutation(
                    client, fact, str(af.get("source_sentence", "")), model
                )
                if subtle:
                    candidates.append(("llm_subtle", subtle))
                candidates.append(("heuristic", _mutate_sentence(fact)))
                candidates.append(("heuristic_neg", _mutate_sentence("It is not true that " + fact)))

                chosen_mut = ""
                chosen_method = ""
                for method_name, cand in candidates:
                    cand = str(cand or "").strip()
                    if not cand or cand == fact:
                        continue
                    rel = _llm_relation(client, fact, cand, model)
                    if rel == "contradiction":
                        chosen_mut = cand
                        chosen_method = method_name
                        break
                if chosen_mut:
                    applied_error.append(af_id)
                    applied_error_mutations[af_id] = chosen_mut
                    perturbed_facts_for_summary.append(chosen_mut)
                    error_mutation_method_counter[chosen_method] += 1
                else:
                    perturbed_facts_for_summary.append(fact)
                    error_mutation_method_counter["failed_to_contradict"] += 1
                continue
            perturbed_facts_for_summary.append(fact)

        facts_text = "\n".join(perturbed_facts_for_summary).strip()
        sprime = ""
        sprime_generation = "fallback_concat"
        if facts_text:
            prompt = (
                "You are a science writer. Given the following atomic facts from a biomedical \n"
                "study, write a coherent lay summary paragraph (3-5 sentences) suitable for a \n"
                "general audience. Do NOT add any information not present in the facts. \n"
                "You MUST aggressively paraphrase: avoid copying original wording, prefer \n"
                "synonyms, abstraction, and varied sentence structures while preserving meaning. \n"
                "For kept facts, keep semantics faithful but do not use near-verbatim phrasing. \n"
                "Do NOT use bullet points. Output only the summary paragraph, no preamble.\n\n"
                "Facts:\n"
                f"{facts_text}"
            )
            try:
                resp = client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=0.0,
                )
                sprime = str(resp.get("content", "")).strip()
                if sprime:
                    sprime_generation = "llm"
            except Exception:
                sprime = ""

        if not sprime:
            sprime = " ".join(perturbed_facts_for_summary)
            sprime = re.sub(r"\s+", " ", sprime).strip()
            sprime_generation = "fallback_concat"

        # Ensure mutated errors survive in generated summary; otherwise fallback.
        if sprime_generation == "llm" and applied_error:
            all_survive = True
            for af_id in applied_error:
                mut_claim = applied_error_mutations.get(af_id, "").strip()
                if not mut_claim:
                    continue
                if not _llm_is_supported(client, mut_claim, sprime, model):
                    all_survive = False
                    break
            if all_survive:
                survival_check_counter["survived"] += 1
            else:
                survival_check_counter["failed"] += 1
                sprime = " ".join(perturbed_facts_for_summary)
                sprime = re.sub(r"\s+", " ", sprime).strip()
                sprime_generation = "fallback_concat_survival"

        sprime_method_counter[sprime_generation] += 1
        sprime_method_by_article[article_id] = sprime_generation

        for af in items:
            af_id = str(af["af_id"])
            if af_id in applied_del:
                ptype, label = "F_del", 1
            elif af_id in applied_error:
                ptype, label = "F_error", 1
            else:
                ptype, label = "F_kept", 0
            c[ptype] += 1
            gt_rows.append(
                {
                    "af_id": af_id,
                    "article_id": article_id,
                    "source_dataset": source,
                    "fact": str(af["fact"]),
                    "source_sentence": str(af.get("source_sentence", "")),
                    "perturbation_type": ptype,
                    "hallucination_label": label,
                }
            )

        sprime_rows.append(
            {
                "article_id": article_id,
                "source_dataset": source,
                "original_summary": original_summary,
                "perturbed_summary": sprime,
                "n_af_total": n,
                "n_del": len(applied_del),
                "n_error": len(applied_error),
                "n_kept": n - len(applied_del) - len(applied_error),
                "perturbation_ratio_actual": round((len(applied_del) + len(applied_error)) / max(n, 1), 4),
                "del_af_ids": applied_del,
                "error_af_ids": applied_error,
                "sprime_generation": sprime_generation,
            }
        )

    _save_jsonl(gt_rows, out_gt)
    _save_jsonl(sprime_rows, out_sprime)
    save_json(
        {
            "seed": seed,
            "target_ratio": ratio,
            "model": model,
            "total_af": len(gt_rows),
            "F_del": c["F_del"],
            "F_error": c["F_error"],
            "F_kept": c["F_kept"],
            "hallucination_positive": c["F_del"] + c["F_error"],
            "hallucination_negative": c["F_kept"],
            "sprime_generation": {
                "llm": sprime_method_counter["llm"],
                "fallback_concat": sprime_method_counter["fallback_concat"],
                "fallback_concat_survival": sprime_method_counter["fallback_concat_survival"],
            },
            "error_mutation_method": dict(error_mutation_method_counter),
            "error_survival_check": dict(survival_check_counter),
            "sprime_generation_by_article": sprime_method_by_article,
        },
        out_meta,
    )
    print(f"[stage03] total={len(gt_rows)} pos={c['F_del']+c['F_error']} -> {DIR_02}")


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
    return dot / (na * nb)


def stage04_rag(top_k: int, chunk_words: int, overlap_words: int, embed_model: str) -> None:
    _ensure_dirs()
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing. Please set it in .env before stage04.")
    afs = _load_jsonl(DIR_01 / "af_gold_for_1a.jsonl")
    sprime_rows = _load_jsonl(DIR_02 / "perturbed_summary.jsonl")
    out_chunks = DIR_03 / "chunks.jsonl"
    out_retr = DIR_03 / f"retrieval_topk{top_k}.jsonl"
    out_meta = DIR_03 / "chunk_index_meta.json"

    def chunk_text(text: str) -> list[dict[str, Any]]:
        words = re.findall(r"\S+", text or "")
        step = max(1, chunk_words - overlap_words)
        out = []
        i = 0
        cid = 0
        while i < len(words):
            seg = words[i : i + chunk_words]
            if not seg:
                break
            out.append(
                {
                    "chunk_idx": cid,
                    "start_word": i,
                    "end_word": min(i + chunk_words, len(words)),
                    "chunk_text": " ".join(seg),
                }
            )
            cid += 1
            i += step
        return out

    chunks: list[dict[str, Any]] = []
    for row in tqdm(sprime_rows, desc="stage04 | chunking summaries", total=len(sprime_rows)):
        article_id = str(row["article_id"])
        source = str(row["source_dataset"])
        for ch in chunk_text(str(row["perturbed_summary"])):
            chunks.append(
                {
                    "article_id": article_id,
                    "source_dataset": source,
                    "chunk_id": f"{article_id}_c{int(ch['chunk_idx']):03d}",
                    "chunk_idx": ch["chunk_idx"],
                    "start_word": ch["start_word"],
                    "end_word": ch["end_word"],
                    "chunk_text": ch["chunk_text"],
                }
            )

    oai = OpenAI(api_key=config.OPENAI_API_KEY)
    chunk_embs = _embed_texts(oai, [c["chunk_text"] for c in chunks], model=embed_model)
    for i, emb in enumerate(chunk_embs):
        chunks[i]["_emb"] = emb

    query_embs = _embed_texts(oai, [str(a["fact"]) for a in afs], model=embed_model)
    retrieval_rows: list[dict[str, Any]] = []
    for i, af in tqdm(
        enumerate(afs), desc=f"stage04 | retrieve top-{top_k}", total=len(afs)
    ):
        qemb = query_embs[i]
        article_id = str(af["article_id"])
        cands = [c for c in chunks if c["article_id"] == article_id]
        scored = [(_cosine(qemb, c["_emb"]), c) for c in cands]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        retrieval_rows.append(
            {
                "af_id": af["af_id"],
                "article_id": af["article_id"],
                "source_dataset": af["source_dataset"],
                "fact": af["fact"],
                "top_k": top_k,
                "retrieved_chunks": [
                    {
                        "rank": rank + 1,
                        "chunk_id": c["chunk_id"],
                        "score": round(float(s), 6),
                        "chunk_text": c["chunk_text"],
                    }
                    for rank, (s, c) in enumerate(top)
                ],
            }
        )

    _save_jsonl([{k: v for k, v in c.items() if k != "_emb"} for c in chunks], out_chunks)
    _save_jsonl(retrieval_rows, out_retr)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "embedding_model": embed_model,
            "top_k": top_k,
            "chunk_words": chunk_words,
            "chunk_overlap": overlap_words,
            "articles": len(sprime_rows),
            "af_total": len(afs),
            "chunks_total": len(chunks),
            "retrieval_rows": len(retrieval_rows),
        },
        out_meta,
    )
    print(f"[stage04] chunks={len(chunks)} retrieval_rows={len(retrieval_rows)} -> {DIR_03}")


def stage05_judgement(
    model: str,
    top_k: int,
    parallel_setups: bool = False,
    reuse_questions: bool = False,
    pilot_first_n: int | None = None,
    c_recheck_mode: str = "strict",
) -> None:
    _ensure_dirs()
    retrieval = _load_jsonl(DIR_03 / f"retrieval_topk{top_k}.jsonl")
    af_rows = _load_jsonl(DIR_01 / "af_gold_for_1a.jsonl")
    if c_recheck_mode not in {"off", "lenient", "strict"}:
        raise ValueError("c_recheck_mode must be one of: off, lenient, strict")
    if pilot_first_n is not None:
        pilot_ids = set(_load_pilot_article_ids(pilot_first_n))
        retrieval = [r for r in retrieval if str(r.get("article_id", "")) in pilot_ids]
        print(f"[stage05] pilot_first_n={pilot_first_n} filtered_rows={len(retrieval)}")
    out_a = DIR_04 / "setup_a_predictions.jsonl"
    out_b = DIR_04 / "setup_b_predictions.jsonl"
    out_c = DIR_04 / "setup_c_predictions.jsonl"
    out_q = DIR_04 / "questions_1t3f.jsonl"
    out_meta = DIR_04 / "judgement_metadata.json"

    client = OpenAIClient()
    q_prompt_template = (config.PROMPTS_DIR / "question_generation.txt").read_text(encoding="utf-8")
    solver_prompt_template = (config.PROMPTS_DIR / "solver_gold.txt").read_text(encoding="utf-8")

    af_map = {str(r["af_id"]): r for r in af_rows}

    def to_letter(i: int) -> str:
        return "ABCD"[i]

    questions: dict[str, dict[str, Any]] = {}
    question_source = "generated"
    required_af_ids = {str(r["af_id"]) for r in retrieval}
    reused_questions_count = 0

    if reuse_questions and out_q.exists():
        cached_rows = _load_jsonl(out_q)
        cached_map = {
            str(r.get("af_id", "")): r
            for r in cached_rows
            if str(r.get("af_id", "")) and isinstance(r.get("options"), list) and len(r.get("options", [])) == 4
        }
        if required_af_ids.issubset(set(cached_map.keys())):
            questions = {af_id: cached_map[af_id] for af_id in required_af_ids}
            question_source = "reused"
            reused_questions_count = len(questions)
            print(f"[stage05] reuse questions_1t3f: {reused_questions_count} loaded from {out_q}")

    if not questions:
        # Build true 1T3F options with Exp0-validated generation prompt.
        for r in tqdm(retrieval, desc="stage05 | build 1T3F", total=len(retrieval)):
            af_id = str(r["af_id"])
            if af_id in questions:
                continue
            af = af_map[af_id]
            source_paragraph = str(af["lay_summary"])
            atomic_fact = str(af["fact"])
            prompt = q_prompt_template.replace("{source_paragraph}", source_paragraph).replace(
                "{atomic_fact}", atomic_fact
            )
            raw = ""
            try:
                resp = client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                raw = str(resp.get("content", ""))
                obj = json.loads(raw)
                true_statement = str(obj["true_statement"])
                false_items = obj["false_statements"]
                if not isinstance(false_items, list) or len(false_items) != 3:
                    raise ValueError("false_statements must be length 3")
                false_texts = [str(x["text"]) for x in false_items]
            except Exception:
                # fallback: deterministic simple 1T3F if generation fails
                true_statement = atomic_fact
                false_texts = [
                    _mutate_sentence(atomic_fact),
                    _mutate_sentence("It is not true that " + atomic_fact),
                    _mutate_sentence(atomic_fact.replace(" is ", " is not ", 1)),
                ]
                false_texts = [t if t else atomic_fact + " (not supported)" for t in false_texts][:3]

            false_texts = _harden_false_options(true_statement, false_texts, atomic_fact)

            options_with_label = [("TRUE", true_statement)] + [("FALSE", t) for t in false_texts]
            shuffler = random.Random(hash(af_id) % (2**32))
            shuffler.shuffle(options_with_label)
            options = [t for _, t in options_with_label]
            correct_idx = next(i for i, (lab, _) in enumerate(options_with_label) if lab == "TRUE")
            questions[af_id] = {
                "af_id": af_id,
                "options": options,
                "correct_letter": to_letter(correct_idx),
                "question_id": f"{af_id}_q",
            }

    prompt_a = (
        "You are a context-grounded factual verifier. "
        "Use ONLY the provided context chunks and do NOT use outside knowledge. "
        "Decide whether the claim is supported by the context. "
        "The context may paraphrase or reword the claim; semantic equivalence counts as support only when critical details match. "
        "Before deciding, extract a verbatim quote from context that supports ALL critical details of the claim "
        "(entity, number/time, direction/polarity, and key qualifiers/modifiers). "
        "If no such quote exists, set supporting_evidence to 'Quote: NONE'. "
        "Beware of over-generalization: if the claim is specific but context is only generic, treat as unsupported. "
        "If there is any contradiction in critical details, treat as unsupported. "
        "A support judgement must be grounded in explicit evidence from the provided chunks. "
        "If unsupported or contradicted, set hallucination=1; otherwise hallucination=0. "
        'Return ONLY JSON: {"hallucination": 0 or 1, "supporting_evidence": "...", "reason": "..."}'
    )

    def run_one(row: dict[str, Any], setup_name: str, system_prompt: str) -> dict[str, Any]:
        chunks_txt = "\n\n".join(
            [f"[{c['rank']}] {c['chunk_text']}" for c in row.get("retrieved_chunks", [])]
        )
        user_msg = f"Claim:\n{row['fact']}\n\nContext Chunks:\n{chunks_txt}"
        raw = ""
        hallucination = 1
        supporting_evidence = ""
        reason = ""
        parse_error = False
        try:
            resp = client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = str(resp.get("content", ""))
            obj = json.loads(raw)
            hallucination = int(obj.get("hallucination", 1))
            if hallucination not in (0, 1):
                hallucination = 1
            supporting_evidence = str(obj.get("supporting_evidence", "")).strip()
            reason = str(obj.get("reason", ""))
        except Exception:
            parse_error = True
            reason = "parse_or_runtime_error"
        return {
            "af_id": row["af_id"],
            "article_id": row["article_id"],
            "source_dataset": row["source_dataset"],
            "fact": row["fact"],
            "top_k": row["top_k"],
            "setup": setup_name,
            "pred_hallucination": hallucination,
            "supporting_evidence": supporting_evidence,
            "reason": reason,
            "parse_error": parse_error,
        }

    def run_setup_a() -> list[dict[str, Any]]:
        return [
            run_one(r, "A", prompt_a)
            for r in tqdm(retrieval, desc="stage05 | setup A judgement", total=len(retrieval))
        ]

    # True Setup B: choose single correct option from 1T3F using Exp0 solver prompt.
    def run_setup_b() -> list[dict[str, Any]]:
        rows_b: list[dict[str, Any]] = []
        local_client = OpenAIClient()
        for r in tqdm(retrieval, desc="stage05 | setup B 1T3F solve", total=len(retrieval)):
            af_id = str(r["af_id"])
            q = questions[af_id]
            chunks_txt = "\n\n".join([f"[{c['rank']}] {c['chunk_text']}" for c in r.get("retrieved_chunks", [])])
            opts = q["options"]
            prompt = (
                solver_prompt_template.replace("{source_text}", chunks_txt)
                .replace("{option_a}", str(opts[0]))
                .replace("{option_b}", str(opts[1]))
                .replace("{option_c}", str(opts[2]))
                .replace("{option_d}", str(opts[3]))
            )
            raw = ""
            pred_letter = ""
            reason = ""
            parse_error = False
            try:
                resp = local_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                raw = str(resp.get("content", ""))
                obj = json.loads(raw)
                pred_letter = str(obj.get("answer", "")).strip().upper()
                if pred_letter not in {"A", "B", "C", "D"}:
                    pred_letter = ""
                reason = str(obj.get("reasoning", ""))
            except Exception:
                parse_error = True
                reason = "parse_or_runtime_error"

            is_correct = pred_letter == q["correct_letter"]
            pred_hallucination = 0 if is_correct else 1
            rows_b.append(
                {
                    "af_id": r["af_id"],
                    "article_id": r["article_id"],
                    "source_dataset": r["source_dataset"],
                    "fact": r["fact"],
                    "top_k": r["top_k"],
                    "setup": "B",
                    "question_id": q["question_id"],
                    "options": q["options"],
                    "correct_letter": q["correct_letter"],
                    "predicted_letter": pred_letter,
                    "pred_hallucination": pred_hallucination,
                    "reason": reason,
                    "parse_error": parse_error,
                }
            )
        return rows_b

    # Setup C: 1T3F + None-of-the-above (E), context-grounded with evidence.
    prompt_c = (
        "You are a strict medical evaluator using atomic falsification. "
        "Use ONLY the provided context chunks and ignore outside knowledge. "
        "Step 1 (Contrastive Variables): before reading for similarity, compare A/B/C/D and list the exact differing variables "
        "(numbers, entities, direction words, negations, key modifiers, locations, time). "
        "Step 2 (Targeted Search): scan the context only to locate the exact state of those variables. "
        "Step 3 (Strict Falsification): if a candidate has any contradiction or any critical variable is missing, reject that candidate. "
        "Do not accept partial lexical overlap as support. "
        "Evaluate options A/B/C/D one by one before deciding. "
        "For EACH option, run this 3-step falsification check: "
        "(1) Detail Check: if the option adds critical details not supported by context (numbers, named entities, locations, time, explicit qualifiers), reject it. "
        "(2) Direction Check: if subject/object/direction/polarity mismatches (male vs female, increase vs decrease, has vs lacks), treat as fatal contradiction and reject. "
        "(3) Modifier Check: if key modifiers conflict (fixed vs deformable, partial vs complete, early vs late), treat as fatal contradiction and reject. "
        "Devil's Advocate Check: before finalizing any A-D choice, actively search for ONE conflicting word/modifier/logic flow against that choice. "
        "If any conflict is found, you MUST reject that choice and output E unless another option is fully supported without conflict. "
        "Do not over-generalize: a generic context statement cannot support a specific option detail. "
        "Negative exemplar (must output E): if context says 'Andeans, Tibetans, Ethiopian highlanders lived at altitude for thousands of years' "
        "but an option adds unsupported specific locations like 'Andean Altiplano / Himalayan plateau / Semian Plateau', "
        "treat it as insufficient support and reject. "
        "Positive exemplar (can select option): if context explicitly states all critical details (same entities, direction/polarity, numbers/modifiers), "
        "and no contradiction exists, selecting that single supported option is valid. "
        "Before selecting A-D, you must provide a verbatim quote covering all critical details. "
        "If no quote covers all critical details, that option is not supported. "
        "After checks, assign one verdict per option: supported, contradicted, or insufficient. "
        "Select A-D only if exactly ONE option is fully supported with NO contradiction. "
        "If all options fail any check, or none is fully supported, output E (None of the above). "
        "Do not guess or choose the closest option by lexical overlap. "
        "When selecting A-D, supporting_evidence must include the exact supporting context quote. "
        "When selecting E, supporting_evidence should be 'Quote: NONE'. "
        'Return ONLY JSON: {"variables":["..."],"option_checks":{"A":"supported|contradicted|insufficient","B":"supported|contradicted|insufficient","C":"supported|contradicted|insufficient","D":"supported|contradicted|insufficient"},"devils_advocate_conflict_found":true|false,"devils_advocate_note":"...","answer":"A|B|C|D|E","supporting_evidence":"<exact quote from context; Quote: NONE if E>","reasoning":"..."}'
    )
    prompt_c_recheck = (
        "You are a strict second-pass checker for E decisions using atomic falsification. "
        "A previous pass selected E (None of the above). "
        "Re-run contrastive variable isolation and targeted search before deciding. "
        "Re-evaluate A/B/C/D option-by-option using the same 3-step falsification checks: detail, direction, modifier. "
        "Apply Devil's Advocate Check again to any tentative A-D choice; any conflict means reject. "
        "Do not over-generalize generic statements to specific details. "
        "Keep the same exemplar policy: unsupported added specifics => reject; full critical-detail match without contradiction => allow. "
        "Only consider an option supported if a verbatim quote covers all critical details. "
        "Any failed check means the option cannot be selected. "
        "Override E only if exactly one option is fully supported without contradiction and has textual evidence. "
        "Otherwise keep E. "
        "Use ONLY the provided context and ignore outside knowledge. "
        'Return ONLY JSON: {"variables":["..."],"option_checks":{"A":"supported|contradicted|insufficient","B":"supported|contradicted|insufficient","C":"supported|contradicted|insufficient","D":"supported|contradicted|insufficient"},"devils_advocate_conflict_found":true|false,"devils_advocate_note":"...","answer":"A|B|C|D|E", "supporting_evidence":"...", "reasoning":"..."}'
    )

    def run_setup_c() -> list[dict[str, Any]]:
        rows_c: list[dict[str, Any]] = []
        local_client = OpenAIClient()
        for r in tqdm(retrieval, desc="stage05 | setup C 1T3F+NOTA", total=len(retrieval)):
            af_id = str(r["af_id"])
            q = questions[af_id]
            opts = q["options"]
            chunks_txt = "\n\n".join([f"[{c['rank']}] {c['chunk_text']}" for c in r.get("retrieved_chunks", [])])
            user_msg = (
                "Context Chunks:\n"
                f"{chunks_txt}\n\n"
                "Question: Which option is supported by the context?\n"
                f"A. {opts[0]}\n"
                f"B. {opts[1]}\n"
                f"C. {opts[2]}\n"
                f"D. {opts[3]}\n"
                "E. None of the above\n"
            )
            pred_letter = ""
            reason = ""
            supporting_evidence = ""
            quote_found = False
            parse_error = False
            try:
                resp = local_client.chat(
                    messages=[
                        {"role": "system", "content": prompt_c},
                        {"role": "user", "content": user_msg},
                    ],
                    model=model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                obj = json.loads(str(resp.get("content", "")))
                pred_letter = str(obj.get("answer", "")).strip().upper()
                if pred_letter not in {"A", "B", "C", "D", "E"}:
                    pred_letter = ""
                supporting_evidence = str(obj.get("supporting_evidence", "")).strip()
                quote_found = supporting_evidence.lower() not in {"", "n/a", "none", "null"}
                reason = str(obj.get("reasoning", ""))
                if pred_letter == "E" and c_recheck_mode != "off":
                    recheck_msg = (
                        "Context Chunks:\n"
                        f"{chunks_txt}\n\n"
                        "Question: Which option is supported by the context?\n"
                        f"A. {opts[0]}\n"
                        f"B. {opts[1]}\n"
                        f"C. {opts[2]}\n"
                        f"D. {opts[3]}\n"
                        "E. None of the above\n\n"
                        "Previous pass selected E. Re-check carefully and override E if any option A-D has support."
                    )
                    recheck_resp = local_client.chat(
                        messages=[
                            {"role": "system", "content": prompt_c_recheck},
                            {"role": "user", "content": recheck_msg},
                        ],
                        model=model,
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                    re_obj = json.loads(str(recheck_resp.get("content", "")))
                    re_letter = str(re_obj.get("answer", "")).strip().upper()
                    re_evidence = str(re_obj.get("supporting_evidence", "")).strip()
                    if c_recheck_mode == "strict":
                        # Conservative override: only switch E -> A-D when evidence is present.
                        if re_letter in {"A", "B", "C", "D"} and re_evidence:
                            pred_letter = re_letter
                            supporting_evidence = re_evidence
                        else:
                            pred_letter = "E"
                    else:  # lenient
                        if re_letter in {"A", "B", "C", "D"}:
                            pred_letter = re_letter
                            if re_evidence:
                                supporting_evidence = re_evidence
                        else:
                            pred_letter = "E"
                    quote_found = supporting_evidence.lower() not in {"", "n/a", "none", "null"}
                    reason = str(re_obj.get("reasoning", "")) or reason
            except Exception:
                parse_error = True
                reason = "parse_or_runtime_error"

            # For hallucination detection:
            # choose true option => supported => hallucination=0
            # choose distractor or E => unsupported => hallucination=1
            is_correct_true_pick = pred_letter == q["correct_letter"]
            pred_hallucination = 0 if is_correct_true_pick else 1
            rows_c.append(
                {
                    "af_id": r["af_id"],
                    "article_id": r["article_id"],
                    "source_dataset": r["source_dataset"],
                    "fact": r["fact"],
                    "top_k": r["top_k"],
                    "setup": "C",
                    "question_id": q["question_id"],
                    "options": q["options"] + ["None of the above"],
                    "correct_letter": q["correct_letter"],
                    "predicted_letter": pred_letter,
                    "pred_hallucination": pred_hallucination,
                    "supporting_evidence": supporting_evidence,
                    "quote_found": quote_found,
                    "reason": reason,
                    "parse_error": parse_error,
                }
            )
        return rows_c

    if parallel_setups:
        with ThreadPoolExecutor(max_workers=3) as ex:
            f_a = ex.submit(run_setup_a)
            f_b = ex.submit(run_setup_b)
            f_c = ex.submit(run_setup_c)
            rows_a = f_a.result()
            rows_b = f_b.result()
            rows_c = f_c.result()
    else:
        rows_a = run_setup_a()
        rows_b = run_setup_b()
        rows_c = run_setup_c()

    _save_jsonl(list(questions.values()), out_q)
    _save_jsonl(rows_a, out_a)
    _save_jsonl(rows_b, out_b)
    _save_jsonl(rows_c, out_c)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "top_k": top_k,
            "total_rows": len(retrieval),
            "questions_1t3f": len(questions),
            "question_source": question_source,
            "reuse_questions_requested": reuse_questions,
            "reused_questions_count": reused_questions_count,
            "pilot_first_n": pilot_first_n,
            "c_recheck_mode": c_recheck_mode,
            "setup_a_parse_error": sum(1 for r in rows_a if r["parse_error"]),
            "setup_b_parse_error": sum(1 for r in rows_b if r["parse_error"]),
            "setup_c_parse_error": sum(1 for r in rows_c if r["parse_error"]),
        },
        out_meta,
    )
    print(f"[stage05] rows={len(retrieval)} -> {DIR_04}")


def _compute_binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = (tp + tn) / max(len(y_true), 1)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
    }


def stage06_eval(top_k: int | None = None, pilot_first_n: int | None = None) -> None:
    _ensure_dirs()
    gt = _load_jsonl(DIR_02 / "gt_labels_1a.jsonl")
    a = _load_jsonl(DIR_04 / "setup_a_predictions.jsonl")
    b = _load_jsonl(DIR_04 / "setup_b_predictions.jsonl")
    c = _load_jsonl(DIR_04 / "setup_c_predictions.jsonl")
    if pilot_first_n is not None:
        pilot_ids = set(_load_pilot_article_ids(pilot_first_n))
        gt = [r for r in gt if str(r.get("article_id", "")) in pilot_ids]
        a = [r for r in a if str(r.get("article_id", "")) in pilot_ids]
        b = [r for r in b if str(r.get("article_id", "")) in pilot_ids]
        c = [r for r in c if str(r.get("article_id", "")) in pilot_ids]
        print(f"[stage06] pilot_first_n={pilot_first_n} filtered_af={len(gt)}")
    out_a = DIR_05 / "metrics_setup_a.json"
    out_b = DIR_05 / "metrics_setup_b.json"
    out_c = DIR_05 / "metrics_setup_c.json"
    out_cm = DIR_05 / "confusion_matrix.csv"
    out_task = DIR_05 / "metrics_taskwise_primary.json"

    gt_map = {str(r["af_id"]): int(r["hallucination_label"]) for r in gt}
    a_map = {str(r["af_id"]): int(r["pred_hallucination"]) for r in a}
    b_map = {str(r["af_id"]): int(r["pred_hallucination"]) for r in b}
    c_map = {str(r["af_id"]): int(r["pred_hallucination"]) for r in c}
    af_ids = sorted(gt_map.keys())
    y_true = [gt_map[k] for k in af_ids]
    y_pred_a = [a_map.get(k, 1) for k in af_ids]
    y_pred_b = [b_map.get(k, 1) for k in af_ids]
    y_pred_c = [c_map.get(k, 1) for k in af_ids]

    ma = _compute_binary_metrics(y_true, y_pred_a)
    mb = _compute_binary_metrics(y_true, y_pred_b)
    mc = _compute_binary_metrics(y_true, y_pred_c)
    save_json({"setup": "A", **ma}, out_a)
    save_json({"setup": "B", **mb}, out_b)
    save_json({"setup": "C", **mc}, out_c)

    lines = [
        "setup,tp,tn,fp,fn,precision,recall,f1,accuracy",
        f"A,{ma['tp']},{ma['tn']},{ma['fp']},{ma['fn']},{ma['precision']},{ma['recall']},{ma['f1']},{ma['accuracy']}",
        f"B,{mb['tp']},{mb['tn']},{mb['fp']},{mb['fn']},{mb['precision']},{mb['recall']},{mb['f1']},{mb['accuracy']}",
        f"C,{mc['tp']},{mc['tn']},{mc['fp']},{mc['fn']},{mc['precision']},{mc['recall']},{mc['f1']},{mc['accuracy']}",
    ]
    out_cm.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if top_k is None:
        for rows in (a, b, c):
            if rows and rows[0].get("top_k") is not None:
                try:
                    top_k = int(rows[0].get("top_k"))
                    break
                except Exception:
                    pass
    if top_k is None:
        top_k = -1

    gt_type_map = {str(r["af_id"]): str(r["perturbation_type"]) for r in gt}

    def _detailed_for_setup(rows: list[dict[str, Any]], setup_name: str) -> dict[str, Any]:
        af_ids_local = [str(r["af_id"]) for r in rows]
        y_true_local = [gt_map[i] for i in af_ids_local]
        y_pred_local = [int(r.get("pred_hallucination", 1)) for r in rows]
        out: dict[str, Any] = {
            "overall": _compute_binary_metrics(y_true_local, y_pred_local),
            "by_type": {},
        }

        for ptype in ["F_del", "F_error", "F_kept"]:
            idx = [i for i, af_id in enumerate(af_ids_local) if gt_type_map.get(af_id) == ptype]
            sub_true = [y_true_local[i] for i in idx]
            sub_pred = [y_pred_local[i] for i in idx]
            n = len(idx)
            correct = (
                sum(1 for t, p in zip(sub_true, sub_pred) if t == p) / n if n else 0.0
            )
            pred_h_rate = (sum(sub_pred) / n) if n else 0.0
            out["by_type"][ptype] = {
                "n": n,
                "correct_rate": round(correct, 4),
                "pred_hallucination_rate": round(pred_h_rate, 4),
                **_compute_binary_metrics(sub_true, sub_pred),
            }

        if setup_name == "C":
            letters = [str(r.get("predicted_letter", "")).upper() for r in rows]
            e_rate = (sum(1 for x in letters if x == "E") / max(len(letters), 1))
            out["E_rate_overall"] = round(e_rate, 4)
            out["E_rate_by_type"] = {}
            for ptype in ["F_del", "F_error", "F_kept"]:
                idx = [i for i, af_id in enumerate(af_ids_local) if gt_type_map.get(af_id) == ptype]
                n = len(idx)
                e_cnt = sum(1 for i in idx if letters[i] == "E")
                out["E_rate_by_type"][ptype] = round(e_cnt / n, 4) if n else 0.0

            # Backward compatible with old field name.
            evidence_key = (
                "supporting_evidence"
                if any("supporting_evidence" in r for r in rows)
                else "supporting_quote"
            )
            non_e = [r for r in rows if str(r.get("predicted_letter", "")).upper() in {"A", "B", "C", "D"}]
            empty_evidence = sum(1 for r in non_e if not str(r.get(evidence_key, "")).strip())
            out["empty_supporting_evidence_rate_when_choose_ABCD"] = round(
                empty_evidence / len(non_e), 4
            ) if non_e else 0.0

        return out

    details = {
        "k": top_k,
        "pilot_first_n": pilot_first_n,
        "reporting_notes": {
            "primary_analysis": "Task-wise reporting by perturbation type (F_del omission, F_error alteration, F_kept retention).",
            "aggregate_warning": "Aggregate binary F1 mixes omission and alteration signals; interpret with caution.",
            "setup_B_positioning": "Setup B is forced-choice 1T3F without abstain (E), treated as an ablation baseline rather than the main method.",
        },
        "setups": {
            "A": _detailed_for_setup(a, "A"),
            "B": _detailed_for_setup(b, "B"),
            "C": _detailed_for_setup(c, "C"),
        },
    }
    out_detail = DIR_05 / f"metrics_detailed_k{top_k}.json"
    save_json(details, out_detail)

    # Primary fairness report: task-wise comparison, separating omission vs alteration.
    def _task_slice(pred_map: dict[str, int], ptypes: set[str]) -> dict[str, Any]:
        af_ids_local = [af_id for af_id in af_ids if gt_type_map.get(af_id) in ptypes]
        y_t = [gt_map[i] for i in af_ids_local]
        y_p = [int(pred_map.get(i, 1)) for i in af_ids_local]
        return {"n": len(af_ids_local), **_compute_binary_metrics(y_t, y_p)}

    taskwise = {
        "k": top_k,
        "pilot_first_n": pilot_first_n,
        "tasks": {
            "omission_detection_F_del": {
                "A": _task_slice(a_map, {"F_del"}),
                "B": _task_slice(b_map, {"F_del"}),
                "C": _task_slice(c_map, {"F_del"}),
            },
            "error_detection_F_error": {
                "A": _task_slice(a_map, {"F_error"}),
                "B": _task_slice(b_map, {"F_error"}),
                "C": _task_slice(c_map, {"F_error"}),
            },
            "retention_specificity_F_kept": {
                "A": _task_slice(a_map, {"F_kept"}),
                "B": _task_slice(b_map, {"F_kept"}),
                "C": _task_slice(c_map, {"F_kept"}),
            },
            "positive_combined_F_del_plus_F_error": {
                "A": _task_slice(a_map, {"F_del", "F_error"}),
                "B": _task_slice(b_map, {"F_del", "F_error"}),
                "C": _task_slice(c_map, {"F_del", "F_error"}),
            },
        },
        "notes": {
            "setup_B": "Forced-choice baseline without abstain; omission rows may induce random guessing noise.",
            "interpretation": "Prefer omission_detection_F_del and retention_specificity_F_kept as primary axes for NOTA-enabled setup evaluation.",
        },
    }
    save_json(taskwise, out_task)
    print(f"[stage06] A_f1={ma['f1']} B_f1={mb['f1']} C_f1={mc['f1']} -> {DIR_05}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1A pipeline runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("stage00_registry")
    p0.add_argument("--seed-pool", type=int, default=20260520)
    p0.add_argument("--seed-pilot", type=int, default=20260521)
    p0.add_argument("--n-full", type=int, default=20)
    p0.add_argument("--n-pilot", type=int, default=5)

    sub.add_parser("stage01_prepare_inputs")

    p2 = sub.add_parser("stage02_extract_af")
    p2.add_argument("--model", type=str, default="gpt-4.1-mini")

    p3 = sub.add_parser("stage03_perturb")
    p3.add_argument("--seed", type=int, default=20260522)
    p3.add_argument("--ratio", type=float, default=0.30)
    p3.add_argument("--model", type=str, default="gpt-4.1-mini")

    p4 = sub.add_parser("stage04_rag")
    p4.add_argument("--top-k", type=int, default=3)
    p4.add_argument("--chunk-words", type=int, default=120)
    p4.add_argument("--overlap-words", type=int, default=30)
    p4.add_argument("--embed-model", type=str, default="text-embedding-3-small")

    p5 = sub.add_parser("stage05_judgement")
    p5.add_argument("--model", type=str, default="gpt-4.1-mini")
    p5.add_argument("--top-k", type=int, default=3)
    p5.add_argument("--parallel-setups", action="store_true")
    p5.add_argument("--reuse-questions", action="store_true")
    p5.add_argument("--pilot-first-n", type=int, default=None)
    p5.add_argument(
        "--c-recheck-mode",
        type=str,
        default="strict",
        choices=["off", "lenient", "strict"],
    )

    p6 = sub.add_parser("stage06_eval")
    p6.add_argument("--top-k", type=int, default=None)
    p6.add_argument("--pilot-first-n", type=int, default=None)

    args = parser.parse_args()
    if args.cmd == "stage00_registry":
        stage00_registry(args.seed_pool, args.seed_pilot, args.n_full, args.n_pilot)
    elif args.cmd == "stage01_prepare_inputs":
        stage01_prepare_inputs()
    elif args.cmd == "stage02_extract_af":
        stage02_extract_af(args.model)
    elif args.cmd == "stage03_perturb":
        stage03_perturb(args.seed, args.ratio, args.model)
    elif args.cmd == "stage04_rag":
        stage04_rag(args.top_k, args.chunk_words, args.overlap_words, args.embed_model)
    elif args.cmd == "stage05_judgement":
        stage05_judgement(
            args.model,
            args.top_k,
            parallel_setups=args.parallel_setups,
            reuse_questions=args.reuse_questions,
            pilot_first_n=args.pilot_first_n,
            c_recheck_mode=args.c_recheck_mode,
        )
    elif args.cmd == "stage06_eval":
        stage06_eval(args.top_k, args.pilot_first_n)


if __name__ == "__main__":
    main()
