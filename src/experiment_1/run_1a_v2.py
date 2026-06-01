from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
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


DATA_1A_V2_DIR = config.DATA_DIR / "experiment_1" / "1a_v2"
RESULTS_1A_V2_DIR = config.RESULTS_DIR / "experiment_1" / "1a_v2"

DIR_00 = DATA_1A_V2_DIR / "00_registry"
DIR_01 = DATA_1A_V2_DIR / "01_inputs"
DIR_02 = DATA_1A_V2_DIR / "02_perturbation"
DIR_03 = DATA_1A_V2_DIR / "03_rag"
DIR_04 = DATA_1A_V2_DIR / "04_judgement"
DIR_05 = DATA_1A_V2_DIR / "05_metrics"

PROMPTS_1A_V2_DIR = ROOT_DIR / "src" / "experiment_1" / "prompts_1a_v2"


def _ensure_dirs() -> None:
    for p in [DIR_00, DIR_01, DIR_02, DIR_03, DIR_04, DIR_05, RESULTS_1A_V2_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def _judgement_dir(top_k: int) -> Path:
    d = DIR_04 / f"k{top_k}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _metrics_dir(top_k: int) -> Path:
    d = DIR_05 / f"k{top_k}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _word_count(text: str | None) -> int:
    return len(str(text or "").split())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def _save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    save_jsonl(rows, path)


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _extract_critical_cues(text: str) -> dict[str, set[str]]:
    t = str(text or "").lower()
    number_tokens = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", t))
    direction_vocab = {
        "increase",
        "increased",
        "decrease",
        "decreased",
        "higher",
        "lower",
        "upregulate",
        "upregulated",
        "downregulate",
        "downregulated",
        "more",
        "less",
        "improve",
        "improved",
        "worse",
        "reduced",
        "reduction",
        "elevated",
    }
    negation_vocab = {"no", "not", "none", "without", "never", "lack", "lacks", "lacking"}
    words = set(re.findall(r"[a-z]+", t))
    return {
        "numbers": number_tokens,
        "directions": words & direction_vocab,
        "negations": words & negation_vocab,
    }


def _enforce_true_option_specificity(atomic_fact: str, true_statement: str) -> str:
    """
    Keep TRUE option aligned with critical details in AF.
    Fall back to atomic_fact when generated TRUE is over-generalized.
    """
    af = str(atomic_fact or "").strip()
    ts = str(true_statement or "").strip()
    if not af:
        return ts
    if not ts:
        return af
    af_cues = _extract_critical_cues(af)
    ts_cues = _extract_critical_cues(ts)
    for k in ["numbers", "directions", "negations"]:
        if af_cues[k] and not ts_cues[k]:
            return af
    return ts


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

    # If fallback candidates keep getting deduped by `seen`, avoid an infinite loop.
    _fallback_suffixes = [" (not supported)", " (unverified)", " (incorrect)"]
    _idx = 0
    while len(picked) < 3:
        suffix = _fallback_suffixes[_idx % len(_fallback_suffixes)]
        _idx += 1
        add_candidate(f"{atomic_fact}{suffix}")
        if len(picked) < 3:
            add_candidate(f"{true_statement}{suffix}")

        # Safety valve: keep runtime bounded even in pathological dedupe cases.
        if _idx > 20:
            base = atomic_fact if _norm_text(atomic_fact) != true_norm else "fallback"
            k = 0
            while len(picked) < 3 and k < 10:
                add_candidate(f"{base} [fallback {k}]")
                k += 1
            break

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
                    "id": f"exp1v2_{prefix}_{i:03d}",
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
    out_af = DIR_01 / "af_gold_for_1a_v2.jsonl"
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


def _small_numeric_perturb(text: str) -> str:
    t = str(text or "").strip()
    if not t:
        return t

    # Subtle percentage shift (e.g., 13% -> 15%).
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", t)
    if m:
        raw = float(m.group(1))
        delta = 2.0 if raw >= 5 else 0.5
        new_v = raw + delta
        if new_v == raw:
            new_v = raw - delta
        rep = str(int(new_v)) if float(new_v).is_integer() else str(round(new_v, 2))
        return t[: m.start(1)] + rep + t[m.end(1) :]

    # Subtle absolute-number shift (e.g., 57,000 -> 59,000; 12 -> 14).
    m = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\b", t)
    if not m:
        return ""
    raw_s = m.group(1)
    num = float(raw_s.replace(",", ""))
    if num >= 1000:
        delta = max(1.0, round(num * 0.03, 0))
    else:
        delta = 2.0 if num >= 10 else 1.0
    new_v = num + delta
    if float(new_v).is_integer():
        rep = f"{int(new_v):,}"
    else:
        rep = str(round(new_v, 2))
    return t[: m.start(1)] + rep + t[m.end(1) :]


def _qualifier_scope_swap(text: str) -> str:
    t = str(text or "").strip()
    if not t:
        return ""
    rules = [
        (r"\ball\b", "most"),
        (r"\bmost\b", "all"),
        (r"\bonly\b", "primarily"),
        (r"\bprimarily\b", "only"),
        (r"\bsignificantly\b", "slightly"),
        (r"\bslightly\b", "significantly"),
        (r"\balways\b", "sometimes"),
        (r"\bsometimes\b", "always"),
        (r"\bcan\b", "cannot"),
        (r"\bcannot\b", "can"),
        (r"\bunder treatment\b", "without treatment"),
        (r"\bwithout treatment\b", "under treatment"),
    ]
    for pat, rep in rules:
        if re.search(pat, t, flags=re.IGNORECASE):
            return re.sub(pat, rep, t, count=1, flags=re.IGNORECASE)
    return ""


def _generate_subtle_mutation_candidates(fact: str) -> list[str]:
    cands: list[str] = []
    for fn in (_small_numeric_perturb, _qualifier_scope_swap):
        try:
            cand = fn(fact).strip()
        except Exception:
            cand = ""
        if cand and cand != fact and cand not in cands:
            cands.append(cand)
    fallback = _mutate_sentence(fact).strip()
    if fallback and fallback != fact and fallback not in cands:
        cands.append(fallback)
    return cands


def _llm_subtle_error_mutation(
    client: OpenAIClient, fact: str, source_sentence: str, model: str
) -> str:
    prompt = (
        "Rewrite the following biomedical claim into a subtle but incorrect statement. "
        "Keep topic/entities similar, change only one key detail "
        "(number/direction/comparison/negation/qualifier/scope). "
        "Prefer subtle hard cases such as:\n"
        "- near-number perturbation (small numeric shift)\n"
        "- qualifier/scope change (all/most, significantly/slightly, under treatment/without treatment)\n"
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


def _llm_error_mutation_sentence(
    client: OpenAIClient,
    sentence: str,
    atomic_fact: str,
    model: str,
) -> str:
    prompt = (
        "Rewrite the following sentence from a biomedical lay summary "
        "to introduce a subtle factual error. "
        "Keep the sentence style, length, and framing intact. "
        "Change only ONE key detail in the core claim "
        "(a number, direction, entity, or qualifier). "
        "Do not change the sentence structure or add new information.\n\n"
        f"Original sentence: {sentence}\n"
        f"Core claim to corrupt: {atomic_fact}\n\n"
        'Return JSON: {"mutated_sentence": "..."}'
    )
    try:
        resp = client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        obj = json.loads(str(resp.get("content", "")))
        return str(obj.get("mutated_sentence", "")).strip()
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


def _llm_vague_replacement(
    client: OpenAIClient,
    sentence: str,
    atomic_fact: str,
    model: str,
) -> str:
    prompt = (PROMPTS_1A_V2_DIR / "vague_replacement.txt").read_text(encoding="utf-8")
    try:
        resp = client.chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt.replace("{sentence}", sentence).replace("{atomic_fact}", atomic_fact),
                }
            ],
            model=model,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        obj = json.loads(str(resp.get("content", "")))
        return str(obj.get("vague_sentence", "")).strip()
    except Exception:
        return ""


def _safe_sentence_split(text: str) -> list[str]:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if s.strip()]
    if sents:
        return sents
    fallback = str(text or "").strip()
    return [fallback] if fallback else []


def _assign_sentence_types(
    sentences: list[str],
    rng: random.Random,
    ratio_del: float,
    ratio_error: float,
) -> dict[str, str]:
    total = len(sentences)
    if total == 0:
        return {}

    n_del = max(1, round(total * ratio_del))
    n_error = max(1, round(total * ratio_error))

    while n_del + n_error > total:
        if n_error > 0:
            n_error -= 1
        elif n_del > 0:
            n_del -= 1
        else:
            break

    shuffled = sentences[:]
    rng.shuffle(shuffled)
    sentence_type: dict[str, str] = {}

    idx = 0
    for s in shuffled[idx : idx + n_del]:
        sentence_type[s] = "F_del"
    idx += n_del
    for s in shuffled[idx : idx + n_error]:
        sentence_type[s] = "F_error"
    idx += n_error
    for s in shuffled[idx:]:
        sentence_type[s] = "F_kept"
    return sentence_type


def _normalize_summary_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _llm_remove_fact_from_summary(
    client: OpenAIClient,
    summary: str,
    atomic_fact: str,
    model: str,
) -> str:
    prompt = (
        "Rewrite the biomedical lay summary to remove one specific claim.\n"
        "Requirements:\n"
        "- Remove any explicit mention and implicit evidence of the target claim.\n"
        "- Keep the rest of the summary content and writing style as intact as possible.\n"
        "- Do not add unrelated new facts.\n"
        "- Return strict JSON only.\n\n"
        f"Target claim to remove: {atomic_fact}\n\n"
        f"Original summary:\n{summary}\n\n"
        'Return JSON: {"rewritten_summary":"..."}'
    )
    try:
        resp = client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        obj = json.loads(str(resp.get("content", "")))
        return _normalize_summary_text(str(obj.get("rewritten_summary", "")).strip())
    except Exception:
        return ""


def _llm_integrate_false_fact_into_summary(
    client: OpenAIClient,
    summary: str,
    true_fact: str,
    false_fact: str,
    model: str,
) -> str:
    prompt = (
        "Rewrite the biomedical lay summary by replacing a true claim with a false claim.\n"
        "Requirements:\n"
        "- Integrate the false claim naturally into the summary.\n"
        "- Remove or adjust conflicting original statements about the true claim.\n"
        "- Keep other unrelated information unchanged as much as possible.\n"
        "- Do not add unrelated new entities.\n"
        "- Return strict JSON only.\n\n"
        f"True claim to replace: {true_fact}\n"
        f"False claim to integrate: {false_fact}\n\n"
        f"Original summary:\n{summary}\n\n"
        'Return JSON: {"rewritten_summary":"..."}'
    )
    try:
        resp = client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        obj = json.loads(str(resp.get("content", "")))
        return _normalize_summary_text(str(obj.get("rewritten_summary", "")).strip())
    except Exception:
        return ""


def stage03_perturb(
    seed: int,
    ratio_del: float = 0.20,
    ratio_error: float = 0.20,
    model: str = "gpt-4.1-mini",
) -> None:
    _ensure_dirs()
    afs = _load_jsonl(DIR_01 / "af_gold_for_1a_v2.jsonl")
    out_gt = DIR_02 / "gt_labels_1a_v2.jsonl"
    out_sprime = DIR_02 / "perturbed_summary_1a_v2.jsonl"
    out_meta = DIR_02 / "perturbation_metadata.json"
    client = OpenAIClient()

    rng = random.Random(seed)
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in afs:
        by_article[str(r["article_id"])].append(r)

    gt_rows: list[dict[str, Any]] = []
    sprime_rows: list[dict[str, Any]] = []
    c = Counter()
    total_rewrite_failures = 0
    total_error_mutation_fallback = 0
    total_del_verification_failures = 0
    total_error_verification_failures = 0
    error_downgrade_non_contradiction_count = 0
    relation_cache: dict[tuple[str, str], str] = {}

    for article_id, items in tqdm(by_article.items(), desc="stage03 | paragraph-level perturb", total=len(by_article)):
        source = str(items[0]["source_dataset"])
        original_summary = _normalize_summary_text(str(items[0]["lay_summary"]))
        working_summary = original_summary

        af_ids = [str(x["af_id"]) for x in items]
        af_type = _assign_sentence_types(
            af_ids,
            rng,
            ratio_del=ratio_del,
            ratio_error=ratio_error,
        )
        per_af_type_final: dict[str, str] = {}
        per_af_mutated_fact: dict[str, str | None] = {}
        per_af_rewrite_ok: dict[str, bool] = {}

        rewrite_failures_this_article = 0
        error_fallback_count = 0
        del_verification_failures_this_article = 0
        error_verification_failures_this_article = 0

        for af in items:
            af_id = str(af["af_id"])
            fact = str(af.get("fact", "")).strip()
            source_sentence = str(af.get("source_sentence", "")).strip()
            ptype = af_type.get(af_id, "F_kept")

            final_type = ptype
            mutated_fact: str | None = None
            rewrite_ok = True

            if final_type == "F_kept":
                per_af_type_final[af_id] = final_type
                per_af_mutated_fact[af_id] = None
                per_af_rewrite_ok[af_id] = True
                continue

            if final_type == "F_del":
                rewritten = ""
                for _ in range(2):
                    cand = _llm_remove_fact_from_summary(client, working_summary, fact, model)
                    if not cand:
                        continue
                    if not _llm_is_supported(client, fact, cand, model):
                        rewritten = cand
                        break
                if rewritten:
                    working_summary = rewritten
                else:
                    rewrite_ok = False
                    rewrite_failures_this_article += 1
                    del_verification_failures_this_article += 1
                    final_type = "F_kept"

            elif final_type == "F_error":
                mutation = ""
                for _ in range(3):
                    cand = _llm_subtle_error_mutation(client, fact, source_sentence, model).strip()
                    if not cand:
                        continue
                    rel = _llm_relation(client, fact, cand, model)
                    if rel == "contradiction":
                        mutation = cand
                        break
                if not mutation:
                    error_fallback_count += 1
                    for cand in _generate_subtle_mutation_candidates(fact):
                        rel = _llm_relation(client, fact, cand, model)
                        if rel == "contradiction":
                            mutation = cand
                            break
                    if not mutation:
                        final_type = "F_kept"
                        error_downgrade_non_contradiction_count += 1

                if final_type == "F_error" and mutation:
                    rewritten = ""
                    for _ in range(2):
                        cand = _llm_integrate_false_fact_into_summary(
                            client=client,
                            summary=working_summary,
                            true_fact=fact,
                            false_fact=mutation,
                            model=model,
                        )
                        if not cand:
                            continue
                        supports_false = _llm_is_supported(client, mutation, cand, model)
                        supports_true = _llm_is_supported(client, fact, cand, model)
                        if supports_false and not supports_true:
                            rewritten = cand
                            break
                    if rewritten:
                        working_summary = rewritten
                        mutated_fact = mutation
                    else:
                        rewrite_ok = False
                        rewrite_failures_this_article += 1
                        error_verification_failures_this_article += 1
                        final_type = "F_kept"

            per_af_type_final[af_id] = final_type
            per_af_mutated_fact[af_id] = mutated_fact
            per_af_rewrite_ok[af_id] = rewrite_ok

        for af in items:
            af_id = str(af["af_id"])
            ptype = per_af_type_final.get(af_id, "F_kept")
            label = 1 if ptype in {"F_del", "F_error"} else 0
            c[ptype] += 1
            gt_rows.append(
                {
                    "af_id": af["af_id"],
                    "article_id": af["article_id"],
                    "source_dataset": af["source_dataset"],
                    "fact": af["fact"],
                    "source_sentence": af.get("source_sentence", ""),
                    "perturbation_type": ptype,
                    "hallucination_label": label,
                    "sentence_matched": True,
                    "mutated_sentence": per_af_mutated_fact.get(af_id),
                    "rewrite_success": bool(per_af_rewrite_ok.get(af_id, False)),
                }
            )

        perturbed_summary = _normalize_summary_text(working_summary)
        n_del = sum(1 for t in per_af_type_final.values() if t == "F_del")
        n_error = sum(1 for t in per_af_type_final.values() if t == "F_error")
        n_kept = sum(1 for t in per_af_type_final.values() if t == "F_kept")

        sprime_rows.append(
            {
                "article_id": article_id,
                "source_dataset": source,
                "original_summary": original_summary,
                "perturbed_summary": perturbed_summary,
                "sprime_construction": "af_driven_paragraph_rewrite",
                "n_del": n_del,
                "n_error": n_error,
                "n_kept": n_kept,
                "sentence_count_original": len(_safe_sentence_split(original_summary)),
                "sentence_count_sprime": len(_safe_sentence_split(perturbed_summary)),
                "sentence_match_failures": 0,
                "error_fallback_count": error_fallback_count,
                "rewrite_failures": rewrite_failures_this_article,
                "del_verification_failures": del_verification_failures_this_article,
                "error_verification_failures": error_verification_failures_this_article,
            }
        )

        total_rewrite_failures += rewrite_failures_this_article
        total_error_mutation_fallback += error_fallback_count
        total_del_verification_failures += del_verification_failures_this_article
        total_error_verification_failures += error_verification_failures_this_article

    _save_jsonl(gt_rows, out_gt)
    _save_jsonl(sprime_rows, out_sprime)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "seed": seed,
            "ratio_del": ratio_del,
            "ratio_error": ratio_error,
            "model": model,
            "total_af": len(gt_rows),
            "F_del": c["F_del"],
            "F_error": c["F_error"],
            "F_kept": c["F_kept"],
            "hallucination_positive": c["F_del"] + c["F_error"],
            "hallucination_negative": c["F_kept"],
            "sentence_match_failure_count": 0,
            "error_fallback_count": total_error_mutation_fallback,
            "rewrite_failure_count": total_rewrite_failures,
            "del_verification_failure_count": total_del_verification_failures,
            "error_verification_failure_count": total_error_verification_failures,
            "error_downgrade_non_contradiction_count": error_downgrade_non_contradiction_count,
        },
        out_meta,
    )
    print(
        f"[stage03] total={len(gt_rows)} "
        f"pos={c['F_del'] + c['F_error']} -> {DIR_02}"
    )


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
    afs = _load_jsonl(DIR_01 / "af_gold_for_1a_v2.jsonl")
    sprime_rows = _load_jsonl(DIR_02 / "perturbed_summary_1a_v2.jsonl")
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


SYSTEM_PROMPT_A = """Determine whether the provided context contains the information described in the claim.

Your task is to determine if the claim is generally consistent with the provided context chunks.

CRITICAL INSTRUCTION:
Focus on the overall topic and main entities. If the claim's general semantic meaning aligns with the context, consider it supported (hallucination = 0).
Do NOT be overly strict about minor phrasing differences, exact numbers, or subtle directional changes. If it looks mostly correct based on a quick skim, accept it.

Return ONLY JSON format without any explanations or reasoning:
{"hallucination": 0 or 1}"""


SYSTEM_PROMPT_C = """You are a strict medical auditor evaluating multiple-choice options.
Use ONLY the provided context chunks. Do NOT use outside knowledge.

Your task: evaluate each option (A/B/C/D) independently.
Do NOT output a final answer letter.

Step 1 — Identify contrastive variables:
List the exact details that differ between options A/B/C/D (numbers, entities, directions, negations).

Step 2 — Search & Match:
Find verbatim quotes from the context that address those exact variables.

Step 3 — Flaw-First Evaluation:
For each option, you MUST explicitly identify any flaws.
- Invented Details: Does the option introduce a specific number, entity, or qualifier NOT found in the text? (If yes -> flaw).
- Contradictions: Does the text say "increase" while the option says "decrease"? Does the text lack a "not" that the option has? (If yes -> flaw).

RELAXED PARAPHRASE RULE (CRITICAL FOR TRUE OPTIONS):
- An option is "supported" if all its claims are proven by the context.
- It is entirely acceptable for an option to be shorter, less detailed, or a summarized version of the context. Do NOT mark an option as "insufficient" just because it omits extra background details found in the text.
- Synonyms and different sentence structures are perfectly acceptable.

HARD CONTRADICTION RULE:
- If context contains explicit negation or opposite direction against an option, set negation_flag=true and status="contradicted".
- If an option invents unverified specific details, set status="insufficient".

ANTI-GUESSING RULE:
If ALL options genuinely contradict the text or invent unverified details, mark them all as "insufficient" or "contradicted". However, if one option accurately summarizes the context without inventing new facts, mark it as "supported".

Return ONLY JSON:
{
  "contrastive_variables": ["..."],
  "option_evaluation": {
    "A": {
      "evidence": "<exact quote or Quote: NONE>",
      "identified_flaws": "<List invented/contradicted details. If perfectly matching or a valid summary, write 'NONE'>",
      "negation_flag": true/false,
      "status": "supported|contradicted|insufficient"
    },
    "B": {
      "evidence": "<exact quote or Quote: NONE>",
      "identified_flaws": "<List invented/contradicted details. If perfectly matching or a valid summary, write 'NONE'>",
      "negation_flag": true/false,
      "status": "supported|contradicted|insufficient"
    },
    "C": {
      "evidence": "<exact quote or Quote: NONE>",
      "identified_flaws": "<List invented/contradicted details. If perfectly matching or a valid summary, write 'NONE'>",
      "negation_flag": true/false,
      "status": "supported|contradicted|insufficient"
    },
    "D": {
      "evidence": "<exact quote or Quote: NONE>",
      "identified_flaws": "<List invented/contradicted details. If perfectly matching or a valid summary, write 'NONE'>",
      "negation_flag": true/false,
      "status": "supported|contradicted|insufficient"
    }
  },
  "reasoning": "..."
}"""


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v or "").strip().lower()
    return s in {"1", "true", "yes", "y"}


def _parse_setup_c_option_eval(obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
    letters = ["A", "B", "C", "D"]
    out: dict[str, dict[str, Any]] = {k: {"status": "", "negation_flag": False, "evidence": ""} for k in letters}

    raw_eval = obj.get("option_evaluation", {})
    if isinstance(raw_eval, dict):
        for k in letters:
            entry = raw_eval.get(k)
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status", "")).strip().lower()
            if status not in {"supported", "contradicted", "insufficient"}:
                status = ""
            out[k] = {
                "status": status,
                "negation_flag": _coerce_bool(entry.get("negation_flag", False)),
                "evidence": str(entry.get("evidence", "")).strip(),
            }

    # Backward-compatible fallback if model returns the old schema.
    raw_checks = obj.get("option_checks", {})
    if isinstance(raw_checks, dict):
        for k in letters:
            if out[k]["status"]:
                continue
            status = str(raw_checks.get(k, "")).strip().lower()
            if status in {"supported", "contradicted", "insufficient"}:
                out[k]["status"] = status

    return out


def _decide_setup_c_answer(option_eval: dict[str, dict[str, Any]]) -> str:
    supported: list[str] = []
    for letter in ["A", "B", "C", "D"]:
        entry = option_eval.get(letter, {})
        status = str(entry.get("status", "")).strip().lower()
        has_negation = _coerce_bool(entry.get("negation_flag", False))
        if status == "supported" and not has_negation:
            supported.append(letter)

    if len(supported) == 1:
        return supported[0]
    if len(supported) == 0:
        return "E"
    # Multi-support: choose the option with the strongest evidence signal (longer,
    # non-empty evidence). This avoids a blanket abstain that inflates FPs.
    def evidence_score(letter: str) -> int:
        ev = str(option_eval.get(letter, {}).get("evidence", "")).strip()
        if not ev:
            return 0
        if ev.lower() in {"quote: none", "none"}:
            return 0
        return len(ev)

    return max(supported, key=evidence_score)


def stage05_judgement(
    model: str,
    top_k: int,
    parallel_setups: bool = False,
    reuse_questions: bool = False,
    pilot_first_n: int | None = None,
) -> None:
    _ensure_dirs()
    retrieval = _load_jsonl(DIR_03 / f"retrieval_topk{top_k}.jsonl")
    af_rows = _load_jsonl(DIR_01 / "af_gold_for_1a_v2.jsonl")
    if pilot_first_n is not None:
        pilot_ids = set(_load_pilot_article_ids(pilot_first_n))
        retrieval = [r for r in retrieval if str(r.get("article_id", "")) in pilot_ids]
        print(f"[stage05] pilot_first_n={pilot_first_n} filtered_rows={len(retrieval)}")

    judgement_dir = _judgement_dir(top_k)
    out_a = judgement_dir / "setup_a_predictions.jsonl"
    out_b = judgement_dir / "setup_b_predictions.jsonl"
    out_c = judgement_dir / "setup_c_predictions.jsonl"
    # Questions are k-independent; keep one shared bank for reuse across k runs.
    out_q = DIR_04 / "questions_1t3f.jsonl"
    out_meta = judgement_dir / "judgement_metadata.json"

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
        reused_ids = {af_id for af_id in required_af_ids if af_id in cached_map}
        if reused_ids:
            questions = {af_id: cached_map[af_id] for af_id in reused_ids}
            reused_questions_count = len(questions)
            question_source = (
                "reused" if reused_questions_count == len(required_af_ids) else "reused_partial"
            )
            print(
                f"[stage05] reuse questions_1t3f: {reused_questions_count}/{len(required_af_ids)} loaded from {out_q}"
            )

    # If we only reuse part of cached questions, still build the missing ones.
    if len(questions) < len(required_af_ids):
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
            try:
                resp = client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                obj = json.loads(str(resp.get("content", "")))
                true_statement = str(obj["true_statement"])
                false_items = obj["false_statements"]
                if not isinstance(false_items, list) or len(false_items) != 3:
                    raise ValueError("false_statements must be length 3")
                false_texts = [str(x["text"]) for x in false_items]
            except Exception:
                true_statement = atomic_fact
                false_texts = [
                    _mutate_sentence(atomic_fact),
                    _mutate_sentence("It is not true that " + atomic_fact),
                    _mutate_sentence(atomic_fact.replace(" is ", " is not ", 1)),
                ]
                false_texts = [t if t else atomic_fact + " (not supported)" for t in false_texts][:3]

            true_statement = _enforce_true_option_specificity(atomic_fact, true_statement)
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

    def run_one(row: dict[str, Any], setup_name: str, system_prompt: str) -> dict[str, Any]:
        chunks_txt = "\n\n".join([f"[{c['rank']}] {c['chunk_text']}" for c in row.get("retrieved_chunks", [])])
        user_msg = f"Claim:\n{row['fact']}\n\nContext Chunks:\n{chunks_txt}"
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
            obj = json.loads(str(resp.get("content", "")))
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
            run_one(r, "A", SYSTEM_PROMPT_A)
            for r in tqdm(retrieval, desc="stage05 | setup A judgement", total=len(retrieval))
        ]

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
                obj = json.loads(str(resp.get("content", "")))
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
            parse_error = False
            option_checks: dict[str, str] = {}
            option_evaluation: dict[str, dict[str, Any]] = {}
            contrastive_variables: list[str] = []
            try:
                resp = local_client.chat(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_C},
                        {"role": "user", "content": user_msg},
                    ],
                    model=model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                obj = json.loads(str(resp.get("content", "")))
                option_evaluation = _parse_setup_c_option_eval(obj)
                pred_letter = _decide_setup_c_answer(option_evaluation)
                reason = str(obj.get("reasoning", ""))
                contrastive_variables = [str(x) for x in obj.get("contrastive_variables", []) if str(x).strip()]
                option_checks = {
                    k: str(option_evaluation.get(k, {}).get("status", "")).strip()
                    for k in ["A", "B", "C", "D"]
                }
                if pred_letter in {"A", "B", "C", "D"}:
                    supporting_evidence = str(option_evaluation.get(pred_letter, {}).get("evidence", "")).strip()
                    if not supporting_evidence:
                        supporting_evidence = "Quote: NONE"
                else:
                    supporting_evidence = "Quote: NONE"
                if not any(option_checks.values()):
                    parse_error = True
                    reason = "invalid_setup_c_schema_or_empty_option_eval"
            except Exception:
                parse_error = True
                reason = "parse_or_runtime_error"

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
                    "reason": reason,
                    "contrastive_variables": contrastive_variables,
                    "option_checks": option_checks,
                    "option_evaluation": option_evaluation,
                    "parse_error": parse_error,
                }
            )
        return rows_c

    # Optional ablation helper; default pipeline does NOT call this.
    def run_setup_c_with_recheck() -> list[dict[str, Any]]:
        raise NotImplementedError(
            "Recheck ablation not yet implemented. "
            "See run_setup_c() for the default no-recheck implementation."
        )

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
            "setup_a_parse_error": sum(1 for r in rows_a if r["parse_error"]),
            "setup_b_parse_error": sum(1 for r in rows_b if r["parse_error"]),
            "setup_c_parse_error": sum(1 for r in rows_c if r["parse_error"]),
        },
        out_meta,
    )
    print(f"[stage05] top_k={top_k} rows={len(retrieval)} -> {judgement_dir}")


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
    if top_k is None:
        raise ValueError("stage06_eval requires --top-k (e.g. 3 or 5) to read k-scoped prediction outputs.")
    judgement_dir = _judgement_dir(top_k)
    metrics_dir = _metrics_dir(top_k)

    gt = _load_jsonl(DIR_02 / "gt_labels_1a_v2.jsonl")
    a = _load_jsonl(judgement_dir / "setup_a_predictions.jsonl")
    b = _load_jsonl(judgement_dir / "setup_b_predictions.jsonl")
    c = _load_jsonl(judgement_dir / "setup_c_predictions.jsonl")
    if pilot_first_n is not None:
        pilot_ids = set(_load_pilot_article_ids(pilot_first_n))
        gt = [r for r in gt if str(r.get("article_id", "")) in pilot_ids]
        a = [r for r in a if str(r.get("article_id", "")) in pilot_ids]
        b = [r for r in b if str(r.get("article_id", "")) in pilot_ids]
        c = [r for r in c if str(r.get("article_id", "")) in pilot_ids]
        print(f"[stage06] pilot_first_n={pilot_first_n} filtered_af={len(gt)}")

    out_a = metrics_dir / "metrics_setup_a.json"
    out_b = metrics_dir / "metrics_setup_b.json"
    out_c = metrics_dir / "metrics_setup_c.json"
    out_cm = metrics_dir / "confusion_matrix.csv"
    out_task = metrics_dir / "metrics_taskwise_primary.json"

    # Exclude parse/runtime failures from metric computation to avoid systematically
    # biasing predictions toward hallucination=1.
    a_valid = [r for r in a if not r.get("parse_error", False)]
    b_valid = [r for r in b if not r.get("parse_error", False)]
    c_valid = [r for r in c if not r.get("parse_error", False)]

    gt_map = {str(r["af_id"]): int(r["hallucination_label"]) for r in gt}
    a_map = {str(r["af_id"]): int(r["pred_hallucination"]) for r in a_valid}
    b_map = {str(r["af_id"]): int(r["pred_hallucination"]) for r in b_valid}
    c_map = {str(r["af_id"]): int(r["pred_hallucination"]) for r in c_valid}

    def _metrics_for_setup(pred_map: dict[str, int]) -> dict[str, float]:
        af_ids_local = sorted(set(pred_map.keys()) & set(gt_map.keys()))
        y_true_local = [gt_map[k] for k in af_ids_local]
        y_pred_local = [pred_map[k] for k in af_ids_local]
        return _compute_binary_metrics(y_true_local, y_pred_local)

    ma = _metrics_for_setup(a_map)
    mb = _metrics_for_setup(b_map)
    mc = _metrics_for_setup(c_map)
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

    gt_type_map = {str(r["af_id"]): str(r["perturbation_type"]) for r in gt}
    af_ids = sorted(gt_map.keys())

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
            correct = (sum(1 for t, p in zip(sub_true, sub_pred) if t == p) / n) if n else 0.0
            pred_h_rate = (sum(sub_pred) / n) if n else 0.0
            out["by_type"][ptype] = {
                "n": n,
                "correct_rate": round(correct, 4),
                "pred_hallucination_rate": round(pred_h_rate, 4),
                **_compute_binary_metrics(sub_true, sub_pred),
            }

        if setup_name == "C":
            letters = [str(r.get("predicted_letter", "")).upper() for r in rows]
            e_rate = sum(1 for x in letters if x == "E") / max(len(letters), 1)
            out["E_rate_overall"] = round(e_rate, 4)
            out["E_rate_by_type"] = {}
            for ptype in ["F_del", "F_error", "F_kept"]:
                idx = [i for i, af_id in enumerate(af_ids_local) if gt_type_map.get(af_id) == ptype]
                n = len(idx)
                e_cnt = sum(1 for i in idx if letters[i] == "E")
                out["E_rate_by_type"][ptype] = round(e_cnt / n, 4) if n else 0.0

            non_e = [r for r in rows if str(r.get("predicted_letter", "")).upper() in {"A", "B", "C", "D"}]
            empty_evidence = sum(1 for r in non_e if not str(r.get("supporting_evidence", "")).strip())
            out["empty_supporting_evidence_rate_when_choose_ABCD"] = round(
                empty_evidence / len(non_e), 4
            ) if non_e else 0.0

        return out

    details = {
        "k": top_k,
        "pilot_first_n": pilot_first_n,
        "reporting_notes": {
            "primary_analysis": (
                "Task-wise reporting by perturbation type "
                "(F_del hard omission, F_error alteration, F_kept retention)."
            ),
            "aggregate_warning": "Aggregate binary F1 mixes omission and alteration signals; interpret with caution.",
            "setup_B_positioning": "Setup B is forced-choice 1T3F without abstain (E), treated as an ablation baseline.",
        },
        "setups": {
            "A": _detailed_for_setup(a_valid, "A"),
            "B": _detailed_for_setup(b_valid, "B"),
            "C": _detailed_for_setup(c_valid, "C"),
        },
    }
    out_detail = metrics_dir / "metrics_detailed.json"
    save_json(details, out_detail)

    def _task_slice(pred_map: dict[str, int], ptypes: set[str]) -> dict[str, Any]:
        af_ids_local = [
            af_id
            for af_id in af_ids
            if gt_type_map.get(af_id) in ptypes and af_id in pred_map
        ]
        y_t = [gt_map[i] for i in af_ids_local]
        y_p = [pred_map[i] for i in af_ids_local]
        return {"n": len(af_ids_local), **_compute_binary_metrics(y_t, y_p)}

    taskwise = {
        "k": top_k,
        "pilot_first_n": pilot_first_n,
        "tasks": {
            "hard_omission_F_del": {
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
            "positive_combined_all": {
                "A": _task_slice(a_map, {"F_del", "F_error"}),
                "B": _task_slice(b_map, {"F_del", "F_error"}),
                "C": _task_slice(c_map, {"F_del", "F_error"}),
            },
        },
    }
    save_json(taskwise, out_task)
    print(f"[stage06] top_k={top_k} A_f1={ma['f1']} B_f1={mb['f1']} C_f1={mc['f1']} -> {metrics_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1A_v2 pipeline runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("stage00_registry")
    p0.add_argument("--seed-pool", type=int, default=20260601)
    p0.add_argument("--seed-pilot", type=int, default=20260602)
    p0.add_argument("--n-full", type=int, default=20)
    p0.add_argument("--n-pilot", type=int, default=5)

    sub.add_parser("stage01_prepare_inputs")

    p2 = sub.add_parser("stage02_extract_af")
    p2.add_argument("--model", type=str, default="gpt-4.1-mini")

    p3 = sub.add_parser("stage03_perturb")
    p3.add_argument("--seed", type=int, default=20260603)
    p3.add_argument("--ratio-del", type=float, default=0.20)
    p3.add_argument("--ratio-error", type=float, default=0.20)
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

    p6 = sub.add_parser("stage06_eval")
    p6.add_argument("--top-k", type=int, required=True)
    p6.add_argument("--pilot-first-n", type=int, default=None)

    args = parser.parse_args()
    if args.cmd == "stage00_registry":
        stage00_registry(args.seed_pool, args.seed_pilot, args.n_full, args.n_pilot)
    elif args.cmd == "stage01_prepare_inputs":
        stage01_prepare_inputs()
    elif args.cmd == "stage02_extract_af":
        stage02_extract_af(args.model)
    elif args.cmd == "stage03_perturb":
        stage03_perturb(
            args.seed,
            ratio_del=args.ratio_del,
            ratio_error=args.ratio_error,
            model=args.model,
        )
    elif args.cmd == "stage04_rag":
        stage04_rag(args.top_k, args.chunk_words, args.overlap_words, args.embed_model)
    elif args.cmd == "stage05_judgement":
        stage05_judgement(
            args.model,
            args.top_k,
            parallel_setups=args.parallel_setups,
            reuse_questions=args.reuse_questions,
            pilot_first_n=args.pilot_first_n,
        )
    elif args.cmd == "stage06_eval":
        stage06_eval(args.top_k, args.pilot_first_n)


if __name__ == "__main__":
    main()
