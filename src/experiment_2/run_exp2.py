from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.utils import OpenAIClient, load_jsonl, save_json, save_jsonl


DATA_EXP2_DIR = config.DATA_DIR / "experiment_2"
RESULTS_EXP2_DIR = config.RESULTS_DIR / "experiment_2"

DIR_00 = DATA_EXP2_DIR / "00_inputs"
DIR_01 = DATA_EXP2_DIR / "01_article_af"
DIR_01B = DATA_EXP2_DIR / "01b_keep_af_ultra_recall"
DIR_02 = DATA_EXP2_DIR / "02_coverage_setup_c"
DIR_02B = DATA_EXP2_DIR / "02b_direct_coverage"
DIR_03 = DATA_EXP2_DIR / "03_keep_skip"
DIR_04 = DATA_EXP2_DIR / "04_eval"
DIR_05 = DATA_EXP2_DIR / "05_human_check"
DIR_PART_C = DATA_EXP2_DIR / "part_c_selective_keep_af"

PROMPTS_EXP2_DIR = ROOT_DIR / "src" / "experiment_2" / "prompts"


SYSTEM_PROMPT_COVERAGE = """You are a strict medical auditor evaluating multiple-choice options.
Use ONLY the provided lay summary. Do NOT use outside knowledge.

Your task: evaluate each option (A/B/C/D) independently.
Do NOT output a final answer letter.

Step 1 — Identify contrastive variables:
List the exact details that differ between options A/B/C/D (numbers, entities, directions, negations).

Step 2 — Search & Match:
Find verbatim quotes from the lay summary that address those exact variables.

Step 3 — Flaw-First Evaluation:
For each option, you MUST explicitly identify any flaws.
- Invented Details: Does the option introduce a specific number, entity, or qualifier NOT found in the summary? (If yes -> flaw).
- Contradictions: Does the summary say "increase" while the option says "decrease"? Does the summary lack a "not" that the option has? (If yes -> flaw).

RELAXED PARAPHRASE RULE (CRITICAL FOR TRUE OPTIONS):
- An option is "supported" if its core information is conveyed by the lay summary.
- The summary may paraphrase, generalize, or abstract the option rather than state it verbatim.
- It is entirely acceptable for the lay summary to be shorter, less detailed, or more patient-facing than the article-level atomic fact.
- Synonyms and different sentence structures are perfectly acceptable.

HARD CONTRADICTION RULE:
If the lay summary contains explicit negation or opposite direction against an option, set negation_flag=true and status="contradicted".
If an option introduces specific details not conveyed by the lay summary, set status="insufficient".

ANTI-GUESSING RULE:
If ALL options genuinely contradict the summary or invent unverified details, mark them all as "insufficient" or "contradicted". However, if one option accurately summarizes the lay summary without inventing new facts, mark it as "supported".

Return ONLY JSON:
{
  "contrastive_variables": ["..."],
  "option_evaluation": {
    "A": {
      "evidence": "<exact quote or Quote: NONE>",
      "identified_flaws": "<flaws or NONE>",
      "negation_flag": true/false,
      "status": "supported|contradicted|insufficient"
    },
    "B": {
      "evidence": "<exact quote or Quote: NONE>",
      "identified_flaws": "<flaws or NONE>",
      "negation_flag": true/false,
      "status": "supported|contradicted|insufficient"
    },
    "C": {
      "evidence": "<exact quote or Quote: NONE>",
      "identified_flaws": "<flaws or NONE>",
      "negation_flag": true/false,
      "status": "supported|contradicted|insufficient"
    },
    "D": {
      "evidence": "<exact quote or Quote: NONE>",
      "identified_flaws": "<flaws or NONE>",
      "negation_flag": true/false,
      "status": "supported|contradicted|insufficient"
    }
  },
  "reasoning": "..."
}"""


def _ensure_dirs() -> None:
    for d in [DIR_00, DIR_01, DIR_01B, DIR_02, DIR_02B, DIR_03, DIR_04, DIR_05, DIR_PART_C, RESULTS_EXP2_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _variant_dir(base: Path, variant: str | None) -> Path:
    if not variant:
        return base
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", variant.strip())
    if not safe:
        return base
    out = base / safe
    out.mkdir(parents=True, exist_ok=True)
    return out


def _keep_skip_prompt_path(variant: str | None) -> Path:
    if variant:
        candidate = PROMPTS_EXP2_DIR / f"keep_skip_{variant}.txt"
        if candidate.exists():
            return candidate
    return PROMPTS_EXP2_DIR / "keep_skip.txt"


def _load(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def _save(rows: list[dict[str, Any]], path: Path) -> None:
    save_jsonl(rows, path)


def _pilot_article_ids(n_per_source: int | None) -> set[str] | None:
    if n_per_source is None:
        return None
    if n_per_source <= 0:
        raise ValueError(f"n_per_source must be > 0, got {n_per_source}")
    articles = _load(DIR_00 / "articles_50.jsonl")
    picked: set[str] = set()
    by_source: dict[str, int] = defaultdict(int)
    for row in articles:
        source = str(row["source_dataset"])
        if by_source[source] >= n_per_source:
            continue
        picked.add(str(row["id"]))
        by_source[source] += 1
    return picked


def _filter_rows_by_articles(rows: list[dict[str, Any]], article_ids: set[str] | None) -> list[dict[str, Any]]:
    if article_ids is None:
        return rows
    return [
        r
        for r in rows
        if str(r.get("article_id", r.get("id", ""))) in article_ids
    ]


def _word_count(text: str | None) -> int:
    return len(str(text or "").split())


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _chunk_words(text: str, chunk_words: int, overlap_words: int = 0) -> list[dict[str, Any]]:
    words = re.findall(r"\S+", str(text or ""))
    chunks: list[dict[str, Any]] = []
    step = max(1, chunk_words - max(0, overlap_words))
    for i in range(0, len(words), step):
        seg = words[i : i + chunk_words]
        if not seg:
            continue
        chunks.append(
            {
                "chunk_idx": len(chunks),
                "start_word": i,
                "end_word": i + len(seg),
                "text": " ".join(seg),
            }
        )
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
    return "It is not true that " + t[0].lower() + t[1:]


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v or "").strip().lower() in {"1", "true", "yes", "y"}


def _parse_option_eval(obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {
        k: {
            "status": "",
            "negation_flag": False,
            "evidence": "",
            "identified_flaws": "",
        }
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
            "identified_flaws": str(entry.get("identified_flaws", "")).strip(),
        }
    return out


def _decide_answer(option_eval: dict[str, dict[str, Any]]) -> str:
    supported = []
    for letter in ["A", "B", "C", "D"]:
        entry = option_eval.get(letter, {})
        if str(entry.get("status", "")).lower() == "supported" and not _coerce_bool(
            entry.get("negation_flag", False)
        ):
            supported.append(letter)
    if not supported:
        return "E"
    if len(supported) == 1:
        return supported[0]

    def evidence_score(letter: str) -> int:
        ev = str(option_eval.get(letter, {}).get("evidence", "")).strip()
        if not ev or ev.lower() in {"none", "quote: none"}:
            return 0
        return len(ev)

    return max(supported, key=evidence_score)


def stage00_prepare_inputs(
    n_per_source: int,
    seed: int,
    exclude_exp0: bool = True,
) -> None:
    _ensure_dirs()
    out_articles = DIR_00 / "articles_50.jsonl"
    out_meta = DIR_00 / "input_metadata.json"

    used: dict[str, set[int]] = {"PLOS": set(), "eLife": set()}
    sampled_old = config.DATA_RAW_DIR / "sampled_articles.jsonl"
    if exclude_exp0 and sampled_old.exists():
        for row in _load(sampled_old):
            used[str(row["source_dataset"])].add(int(row["original_index"]))

    specs = [
        ("PLOS", "BioLaySumm/BioLaySumm2025-PLOS", "plos"),
        ("eLife", "BioLaySumm/BioLaySumm2025-eLife", "elife"),
    ]
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "n_per_source": n_per_source,
        "exclude_exp0": exclude_exp0,
        "excluded_indices_by_source": {k: len(v) for k, v in used.items()},
        "datasets": {},
    }

    for source, ds_name, prefix in specs:
        ds = load_dataset(ds_name, split="validation", token=config.HF_TOKEN or None)
        cols = set(ds.column_names)
        article_col = "article" if "article" in cols else None
        lay_col = "lay_summary" if "lay_summary" in cols else ("summary" if "summary" in cols else None)
        abstract_col = "abstract" if "abstract" in cols else None
        if article_col is None or lay_col is None:
            raise RuntimeError(f"{source}: cannot identify article/lay_summary columns: {ds.column_names}")

        candidates = [i for i in range(len(ds)) if i not in used[source]]
        if len(candidates) < n_per_source:
            raise RuntimeError(f"{source}: only {len(candidates)} candidates, need {n_per_source}")
        picked = rng.sample(candidates, n_per_source)
        meta["datasets"][source] = {
            "dataset_name": ds_name,
            "validation_size": len(ds),
            "candidate_size_after_exclusion": len(candidates),
            "picked_indices": picked,
        }
        for j, idx in enumerate(picked, start=1):
            item = ds[int(idx)]
            article = str(item.get(article_col, ""))
            lay_summary = str(item.get(lay_col, ""))
            abstract = str(item.get(abstract_col, "")) if abstract_col else ""
            rows.append(
                {
                    "id": f"exp2_{prefix}_{j:03d}",
                    "source_dataset": source,
                    "original_index": int(idx),
                    "article": article,
                    "lay_summary": lay_summary,
                    "abstract": abstract,
                    "article_word_count": _word_count(article),
                    "lay_summary_word_count": _word_count(lay_summary),
                }
            )

    meta["total_articles"] = len(rows)
    meta["by_source"] = dict(Counter(r["source_dataset"] for r in rows))
    _save(rows, out_articles)
    save_json(meta, out_meta)
    print(f"[stage00] articles={len(rows)} -> {out_articles}")


def stage01_extract_article_af(
    model: str,
    chunk_words: int,
    overlap_words: int,
    resume: bool = True,
) -> None:
    _ensure_dirs()
    articles = _load(DIR_00 / "articles_50.jsonl")
    prompt_template = (PROMPTS_EXP2_DIR / "article_af_extraction.txt").read_text(encoding="utf-8")
    out_af = DIR_01 / "af_article.jsonl"
    out_meta = DIR_01 / "af_extraction_metadata.json"

    existing: list[dict[str, Any]] = _load(out_af) if resume and out_af.exists() else []
    done_article_ids = {str(r.get("article_id")) for r in existing}
    af_rows = existing[:]
    failed: list[dict[str, Any]] = []
    total_in = total_out = total_tok = 0
    client = OpenAIClient()

    for art in tqdm(articles, desc="stage01 | article AF extraction", total=len(articles)):
        article_id = str(art["id"])
        if resume and article_id in done_article_ids:
            continue

        seen_norms: set[str] = set()
        article_count = 0
        chunks = _chunk_words(str(art["article"]), chunk_words, overlap_words=overlap_words)
        for ch in chunks:
            raw = ""
            try:
                resp = client.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_template.replace("{source_text}", ch["text"]),
                        }
                    ],
                    model=model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                usage = resp.get("usage", {})
                total_in += int(usage.get("prompt_tokens", 0))
                total_out += int(usage.get("completion_tokens", 0))
                total_tok += int(usage.get("total_tokens", 0))
                raw = str(resp.get("content", ""))
                obj = json.loads(raw)
                facts = obj.get("atomic_facts", [])
                if not isinstance(facts, list):
                    facts = []
                for fact_obj in facts:
                    fact = str(fact_obj.get("fact", "")).strip()
                    source_span = str(fact_obj.get("source_span", "")).strip()
                    n = _norm(fact)
                    if not fact or n in seen_norms:
                        continue
                    seen_norms.add(n)
                    article_count += 1
                    af_rows.append(
                        {
                            "af_id": f"{article_id}_af_{article_count:04d}",
                            "article_id": article_id,
                            "source_dataset": art["source_dataset"],
                            "original_index": art["original_index"],
                            "fact": fact,
                            "source_span": source_span,
                            "article_chunk_idx": ch["chunk_idx"],
                            "chunk_start_word": ch["start_word"],
                            "chunk_end_word": ch["end_word"],
                            "lay_summary": art["lay_summary"],
                        }
                    )
            except Exception as exc:
                failed.append(
                    {
                        "article_id": article_id,
                        "chunk_idx": ch["chunk_idx"],
                        "error": str(exc),
                        "raw_preview": raw[:300],
                    }
                )

        done_article_ids.add(article_id)
        _save(af_rows, out_af)

    by_source = Counter(str(r["source_dataset"]) for r in af_rows)
    by_article = Counter(str(r["article_id"]) for r in af_rows)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "chunk_words": chunk_words,
            "overlap_words": overlap_words,
            "articles_total": len(articles),
            "articles_completed": len(done_article_ids),
            "total_afs": len(af_rows),
            "afs_by_source": dict(by_source),
            "afs_by_article": dict(by_article),
            "failed_chunks": failed,
            "failed_chunk_count": len(failed),
            "total_input_tokens_this_run": total_in,
            "total_output_tokens_this_run": total_out,
            "total_tokens_this_run": total_tok,
        },
        out_meta,
    )
    print(f"[stage01] AF_article={len(af_rows)} failed_chunks={len(failed)} -> {DIR_01}")


def stage01b_extract_keep_af_ultra_recall(
    model: str,
    chunk_words: int,
    overlap_words: int,
    pilot_per_source: int | None = None,
    resume: bool = True,
) -> None:
    _ensure_dirs()
    pilot_ids = _pilot_article_ids(pilot_per_source)
    articles = _filter_rows_by_articles(_load(DIR_00 / "articles_50.jsonl"), pilot_ids)
    if pilot_per_source is not None:
        print(f"[stage01b] pilot_per_source={pilot_per_source} articles={len(articles)}")

    prompt_template = (PROMPTS_EXP2_DIR / "selective_keep_af_ultra_recall.txt").read_text(encoding="utf-8")
    out_af = DIR_01B / "keep_af_article.jsonl"
    out_meta = DIR_01B / "keep_af_extraction_metadata.json"

    existing: list[dict[str, Any]] = _load(out_af) if resume and out_af.exists() else []
    done_article_ids = {str(r.get("article_id")) for r in existing}
    keep_rows = existing[:]
    failed: list[dict[str, Any]] = []
    total_in = total_out = total_tok = 0
    client = OpenAIClient()

    for art in tqdm(articles, desc="stage01b | selective keep AF extraction", total=len(articles)):
        article_id = str(art["id"])
        if resume and article_id in done_article_ids:
            continue

        seen_norms: set[str] = set()
        article_count = 0
        chunks = _chunk_words(str(art["article"]), chunk_words, overlap_words=overlap_words)
        for ch in chunks:
            raw = ""
            try:
                resp = client.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_template.replace("{source_text}", ch["text"]),
                        }
                    ],
                    model=model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                usage = resp.get("usage", {})
                total_in += int(usage.get("prompt_tokens", 0))
                total_out += int(usage.get("completion_tokens", 0))
                total_tok += int(usage.get("total_tokens", 0))
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
                    if not fact or n in seen_norms:
                        continue
                    seen_norms.add(n)
                    article_count += 1
                    keep_rows.append(
                        {
                            "keep_af_id": f"{article_id}_keep_af_{article_count:04d}",
                            "article_id": article_id,
                            "source_dataset": art["source_dataset"],
                            "original_index": art["original_index"],
                            "fact": fact,
                            "source_span": source_span,
                            "keep_reasoning": reasoning,
                            "article_chunk_idx": ch["chunk_idx"],
                            "chunk_start_word": ch["start_word"],
                            "chunk_end_word": ch["end_word"],
                            "lay_summary": art["lay_summary"],
                        }
                    )
            except Exception as exc:
                failed.append(
                    {
                        "article_id": article_id,
                        "chunk_idx": ch["chunk_idx"],
                        "error": str(exc),
                        "raw_preview": raw[:300],
                    }
                )

        done_article_ids.add(article_id)
        _save(keep_rows, out_af)

    by_source = Counter(str(r["source_dataset"]) for r in keep_rows)
    by_article = Counter(str(r["article_id"]) for r in keep_rows)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "chunk_words": chunk_words,
            "overlap_words": overlap_words,
            "pilot_per_source": pilot_per_source,
            "articles_total": len(articles),
            "articles_completed": len(done_article_ids),
            "total_keep_afs": len(keep_rows),
            "keep_afs_by_source": dict(by_source),
            "keep_afs_by_article": dict(by_article),
            "failed_chunks": failed,
            "failed_chunk_count": len(failed),
            "total_input_tokens_this_run": total_in,
            "total_output_tokens_this_run": total_out,
            "total_tokens_this_run": total_tok,
        },
        out_meta,
    )
    print(f"[stage01b] keep_AF={len(keep_rows)} failed_chunks={len(failed)} -> {DIR_01B}")


def _build_question(
    client: OpenAIClient,
    af: dict[str, Any],
    model: str,
    prompt_template: str,
) -> dict[str, Any]:
    try:
        resp = client.chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt_template.replace("{atomic_fact}", str(af["fact"])).replace(
                        "{source_span}", str(af.get("source_span", ""))
                    ),
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
        false_texts = [str(x.get("text", "")).strip() for x in false_items if isinstance(x, dict)]
        false_texts = [x for x in false_texts if x]
    except Exception:
        true_statement = str(af["fact"])
        false_texts = []

    while len(false_texts) < 3:
        cand = _mutate_sentence(true_statement if len(false_texts) % 2 == 0 else str(af["fact"]))
        if _norm(cand) != _norm(true_statement) and cand not in false_texts:
            false_texts.append(cand)
        else:
            false_texts.append(f"{af['fact']} (not supported detail {len(false_texts) + 1})")

    options_with_label = [("TRUE", true_statement)] + [("FALSE", x) for x in false_texts[:3]]
    stable_seed = int(hashlib.md5(str(af["af_id"]).encode("utf-8")).hexdigest()[:8], 16)
    shuffler = random.Random(stable_seed)
    shuffler.shuffle(options_with_label)
    options = [x for _, x in options_with_label]
    correct_idx = next(i for i, (lab, _) in enumerate(options_with_label) if lab == "TRUE")
    return {
        "af_id": af["af_id"],
        "question_id": f"{af['af_id']}_q",
        "options": options,
        "correct_letter": "ABCD"[correct_idx],
    }


def stage02_coverage_setup_c(
    model: str,
    reuse_questions: bool = True,
    resume: bool = True,
    pilot_per_source: int | None = None,
) -> None:
    _ensure_dirs()
    pilot_ids = _pilot_article_ids(pilot_per_source)
    af_rows = _filter_rows_by_articles(_load(DIR_01 / "af_article.jsonl"), pilot_ids)
    if pilot_per_source is not None:
        print(f"[stage02] pilot_per_source={pilot_per_source} filtered_af={len(af_rows)}")
    out_questions = DIR_02 / "questions_1t3f_article_af.jsonl"
    out_cov = DIR_02 / "coverage_predictions.jsonl"
    out_meta = DIR_02 / "coverage_metadata.json"

    client = OpenAIClient()
    q_prompt = (PROMPTS_EXP2_DIR / "coverage_question_generation.txt").read_text(encoding="utf-8")

    questions: dict[str, dict[str, Any]] = {}
    if reuse_questions and out_questions.exists():
        for q in _load(out_questions):
            if str(q.get("af_id", "")) and isinstance(q.get("options"), list) and len(q.get("options", [])) == 4:
                questions[str(q["af_id"])] = q

    existing_cov = _load(out_cov) if resume and out_cov.exists() else []
    done = {str(r.get("af_id")) for r in existing_cov}
    cov_rows = existing_cov[:]
    parse_errors = 0

    for af in tqdm(af_rows, desc="stage02 | Setup C coverage", total=len(af_rows)):
        af_id = str(af["af_id"])
        if resume and af_id in done:
            continue
        if af_id not in questions:
            questions[af_id] = _build_question(client, af, model, q_prompt)

        q = questions[af_id]
        opts = q["options"]
        user_msg = (
            "Lay Summary:\n"
            f"{af['lay_summary']}\n\n"
            "Question: Which option is supported by the lay summary?\n"
            f"A. {opts[0]}\n"
            f"B. {opts[1]}\n"
            f"C. {opts[2]}\n"
            f"D. {opts[3]}\n"
            "E. None of the above\n"
        )

        option_eval: dict[str, dict[str, Any]] = {}
        contrastive_variables: list[str] = []
        pred_letter = "E"
        reason = ""
        parse_error = False
        try:
            resp = client.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_COVERAGE},
                    {"role": "user", "content": user_msg},
                ],
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = json.loads(str(resp.get("content", "")))
            option_eval = _parse_option_eval(obj)
            pred_letter = _decide_answer(option_eval)
            contrastive_variables = [str(x) for x in obj.get("contrastive_variables", []) if str(x).strip()]
            reason = str(obj.get("reasoning", "")).strip()
            if not any(str(option_eval[k].get("status", "")) for k in ["A", "B", "C", "D"]):
                parse_error = True
        except Exception as exc:
            parse_error = True
            reason = f"parse_or_runtime_error: {exc}"

        if parse_error:
            parse_errors += 1
        covered = (not parse_error) and pred_letter == q["correct_letter"]
        true_eval = option_eval.get(q["correct_letter"], {})
        cov_rows.append(
            {
                "af_id": af_id,
                "article_id": af["article_id"],
                "source_dataset": af["source_dataset"],
                "fact": af["fact"],
                "question_id": q["question_id"],
                "options": q["options"] + ["None of the above"],
                "correct_letter": q["correct_letter"],
                "predicted_letter": pred_letter,
                "covered": covered,
                "supporting_span": true_eval.get("evidence", "Quote: NONE") if covered else "Quote: NONE",
                "true_option_status": true_eval.get("status", ""),
                "contrastive_variables": contrastive_variables,
                "option_evaluation": option_eval,
                "reason": reason,
                "parse_error": parse_error,
            }
        )
        done.add(af_id)
        _save(list(questions.values()), out_questions)
        _save(cov_rows, out_cov)

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "total_afs": len(af_rows),
            "coverage_rows": len(cov_rows),
            "questions": len(questions),
            "reuse_questions": reuse_questions,
            "parse_error_count": sum(1 for r in cov_rows if r.get("parse_error")),
        },
        out_meta,
    )
    print(f"[stage02] coverage_rows={len(cov_rows)} parse_errors={parse_errors} -> {DIR_02}")


def stage02b_direct_coverage(
    model: str,
    resume: bool = True,
    pilot_per_source: int | None = None,
) -> None:
    _ensure_dirs()
    pilot_ids = _pilot_article_ids(pilot_per_source)
    af_rows = _filter_rows_by_articles(_load(DIR_01 / "af_article.jsonl"), pilot_ids)
    if pilot_per_source is not None:
        print(f"[stage02b] pilot_per_source={pilot_per_source} filtered_af={len(af_rows)}")
    out_cov = DIR_02B / "coverage_direct_predictions.jsonl"
    out_meta = DIR_02B / "coverage_direct_metadata.json"
    prompt = (PROMPTS_EXP2_DIR / "direct_coverage.txt").read_text(encoding="utf-8")
    client = OpenAIClient()

    rows = _load(out_cov) if resume and out_cov.exists() else []
    done = {str(r.get("af_id")) for r in rows}
    parse_errors = 0

    for af in tqdm(af_rows, desc="stage02b | direct coverage", total=len(af_rows)):
        af_id = str(af["af_id"])
        if resume and af_id in done:
            continue

        covered = False
        supporting_span: str | None = None
        reasoning = ""
        parse_error = False
        try:
            resp = client.chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt.replace("{atomic_fact}", str(af["fact"])).replace(
                            "{lay_summary}", str(af["lay_summary"])
                        ),
                    }
                ],
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = json.loads(str(resp.get("content", "")))
            covered = _coerce_bool(obj.get("covered", False))
            raw_span = obj.get("supporting_span", None)
            supporting_span = None if raw_span is None else str(raw_span).strip()
            if supporting_span in {"", "null", "None"}:
                supporting_span = None
            reasoning = str(obj.get("reasoning", "")).strip()
        except Exception as exc:
            parse_error = True
            parse_errors += 1
            reasoning = f"parse_or_runtime_error: {exc}"

        rows.append(
            {
                "af_id": af_id,
                "article_id": af["article_id"],
                "source_dataset": af["source_dataset"],
                "fact": af["fact"],
                "covered": covered,
                "supporting_span": supporting_span,
                "reasoning": reasoning,
                "parse_error": parse_error,
            }
        )
        done.add(af_id)
        _save(rows, out_cov)

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "total_afs": len(af_rows),
            "coverage_rows": len(rows),
            "parse_error_count": sum(1 for r in rows if r.get("parse_error")),
        },
        out_meta,
    )
    print(f"[stage02b] direct_coverage_rows={len(rows)} parse_errors={parse_errors} -> {DIR_02B}")


def stage03_keep_skip(
    model: str,
    resume: bool = True,
    pilot_per_source: int | None = None,
    variant: str | None = None,
) -> None:
    _ensure_dirs()
    pilot_ids = _pilot_article_ids(pilot_per_source)
    af_rows = _filter_rows_by_articles(_load(DIR_01 / "af_article.jsonl"), pilot_ids)
    if pilot_per_source is not None:
        print(f"[stage03] pilot_per_source={pilot_per_source} filtered_af={len(af_rows)}")
    out_dir = _variant_dir(DIR_03, variant)
    out_keep = out_dir / "keep_skip_predictions.jsonl"
    out_meta = out_dir / "keep_skip_metadata.json"
    prompt_path = _keep_skip_prompt_path(variant)
    prompt = prompt_path.read_text(encoding="utf-8")
    client = OpenAIClient()

    rows = _load(out_keep) if resume and out_keep.exists() else []
    done = {str(r.get("af_id")) for r in rows}
    parse_errors = 0
    for af in tqdm(af_rows, desc="stage03 | keep/skip", total=len(af_rows)):
        af_id = str(af["af_id"])
        if resume and af_id in done:
            continue
        parse_error = False
        keep = False
        confidence = "low"
        reasoning = ""
        try:
            resp = client.chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt.replace("{atomic_fact}", str(af["fact"])),
                    }
                ],
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = json.loads(str(resp.get("content", "")))
            keep = _coerce_bool(obj.get("keep", False))
            confidence = str(obj.get("confidence", "low")).strip().lower()
            if confidence not in {"high", "medium", "low"}:
                confidence = "low"
            reasoning = str(obj.get("reasoning", "")).strip()
        except Exception as exc:
            parse_error = True
            parse_errors += 1
            reasoning = f"parse_or_runtime_error: {exc}"

        rows.append(
            {
                "af_id": af_id,
                "article_id": af["article_id"],
                "source_dataset": af["source_dataset"],
                "fact": af["fact"],
                "module_keep": keep,
                "module_label": "keep" if keep else "skip",
                "confidence": confidence,
                "reasoning": reasoning,
                "parse_error": parse_error,
            }
        )
        done.add(af_id)
        _save(rows, out_keep)

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "variant": variant,
            "prompt_path": str(prompt_path),
            "total_afs": len(af_rows),
            "prediction_rows": len(rows),
            "parse_error_count": sum(1 for r in rows if r.get("parse_error")),
        },
        out_meta,
    )
    print(f"[stage03] keep_skip_rows={len(rows)} parse_errors={parse_errors} -> {out_dir}")


def _binary_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, Any]:
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "dangerous_skip_count": fn,
        "dangerous_skip_rate_among_should_keep": round(fn / (tp + fn), 4) if tp + fn else 0.0,
        "overkeep_count": fp,
        "overkeep_rate_among_should_skip": round(fp / (tn + fp), 4) if tn + fp else 0.0,
    }


def stage04_eval(pilot_per_source: int | None = None, variant: str | None = None) -> None:
    _ensure_dirs()
    keep_dir = _variant_dir(DIR_03, variant)
    eval_dir = _variant_dir(DIR_04, variant)
    results_dir = _variant_dir(RESULTS_EXP2_DIR, variant)
    pilot_ids = _pilot_article_ids(pilot_per_source)
    setup_c_coverage = [
        r
        for r in _filter_rows_by_articles(_load(DIR_02 / "coverage_predictions.jsonl"), pilot_ids)
        if not r.get("parse_error", False)
    ]
    direct_coverage = [
        r
        for r in _filter_rows_by_articles(_load(DIR_02B / "coverage_direct_predictions.jsonl"), pilot_ids)
        if not r.get("parse_error", False)
    ]
    keep_rows = [
        r
        for r in _filter_rows_by_articles(_load(keep_dir / "keep_skip_predictions.jsonl"), pilot_ids)
        if not r.get("parse_error", False)
    ]
    if pilot_per_source is not None:
        print(
            f"[stage04] pilot_per_source={pilot_per_source} "
            f"setup_c={len(setup_c_coverage)} direct={len(direct_coverage)} keep={len(keep_rows)}"
        )
    setup_c_map = {str(r["af_id"]): r for r in setup_c_coverage}
    cov_map = {str(r["af_id"]): r for r in direct_coverage}
    keep_map = {str(r["af_id"]): r for r in keep_rows}
    af_ids = sorted(set(cov_map) & set(keep_map))

    rows: list[dict[str, Any]] = []
    for af_id in af_ids:
        cov = cov_map[af_id]
        keep = keep_map[af_id]
        should_keep = bool(cov["covered"])
        pred_keep = bool(keep["module_keep"])
        rows.append(
            {
                "af_id": af_id,
                "article_id": cov["article_id"],
                "source_dataset": cov["source_dataset"],
                "fact": cov["fact"],
                "ground_truth_keep": should_keep,
                "module_keep": pred_keep,
                "module_label": "keep" if pred_keep else "skip",
                "covered_by_summary": should_keep,
                "supporting_span": cov.get("supporting_span", "Quote: NONE"),
                "part_a_setup_c_covered": bool(setup_c_map.get(af_id, {}).get("covered", False)),
                "part_a_setup_c_predicted_letter": setup_c_map.get(af_id, {}).get("predicted_letter", ""),
                "confidence": keep.get("confidence", "low"),
                "module_reasoning": keep.get("reasoning", ""),
                "correct": should_keep == pred_keep,
                "error_type": (
                    "correct_keep"
                    if should_keep and pred_keep
                    else "correct_skip"
                    if (not should_keep and not pred_keep)
                    else "dangerous_skip"
                    if should_keep and not pred_keep
                    else "overkeep"
                ),
            }
        )

    _save(rows, eval_dir / "keep_skip_vs_coverage.jsonl")

    def summarize(subrows: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(subrows)
        covered = sum(1 for r in subrows if r["ground_truth_keep"])
        not_covered = total - covered
        tp = sum(1 for r in subrows if r["ground_truth_keep"] and r["module_keep"])
        tn = sum(1 for r in subrows if (not r["ground_truth_keep"]) and (not r["module_keep"]))
        fp = sum(1 for r in subrows if (not r["ground_truth_keep"]) and r["module_keep"])
        fn = sum(1 for r in subrows if r["ground_truth_keep"] and (not r["module_keep"]))
        return {
            "n": total,
            "part_a": {
                "coverage_recall": round(covered / total, 4) if total else 0.0,
                "covered_count": covered,
                "not_covered_count": not_covered,
                "safe_simplification_pool_rate": round(not_covered / total, 4) if total else 0.0,
            },
            "part_b": {
                "confusion_matrix": {
                    "module_keep__gt_keep": tp,
                    "module_keep__gt_skip": fp,
                    "module_skip__gt_keep_dangerous": fn,
                    "module_skip__gt_skip": tn,
                },
                **_binary_metrics(tp, tn, fp, fn),
            },
        }

    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "variant": variant,
        "notes": {
            "part_a_ground_truth": "Part A uses Setup C-style 1T3F + None-of-the-above coverage judgement to estimate expert-summary coverage of article facts.",
            "part_b_ground_truth": "Part B uses a direct LLM coverage judgement with supporting span as the keep/skip reference standard, followed by human audit.",
            "part_a": "Low coverage recall demonstrates that many article facts are safe simplifications and Module 2 filtering is necessary.",
            "part_b": "Module keep/skip is evaluated against direct coverage-derived keep/skip ground truth.",
            "dangerous_skip": "Ground truth keep but module predicts skip.",
        },
        "part_a_setup_c": {
            "n": len(setup_c_coverage),
            "coverage_recall": round(
                sum(1 for r in setup_c_coverage if r["covered"]) / len(setup_c_coverage), 4
            )
            if setup_c_coverage
            else 0.0,
            "covered_count": sum(1 for r in setup_c_coverage if r["covered"]),
            "not_covered_count": sum(1 for r in setup_c_coverage if not r["covered"]),
        },
        "part_b_direct_coverage_ground_truth": {
            "n": len(direct_coverage),
            "coverage_recall": round(
                sum(1 for r in direct_coverage if r["covered"]) / len(direct_coverage), 4
            )
            if direct_coverage
            else 0.0,
            "covered_count": sum(1 for r in direct_coverage if r["covered"]),
            "not_covered_count": sum(1 for r in direct_coverage if not r["covered"]),
        },
        "overall": summarize(rows),
        "by_source": {
            source: summarize([r for r in rows if r["source_dataset"] == source])
            for source in sorted({r["source_dataset"] for r in rows})
        },
        "by_confidence": {
            conf: summarize([r for r in rows if r["confidence"] == conf])
            for conf in ["high", "medium", "low"]
        },
    }
    save_json(summary, eval_dir / "metrics_summary.json")

    lines = ["source,tp,tn,fp,fn,accuracy,precision,recall,f1,dangerous_skip_rate_among_should_keep"]
    for source, data in {"overall": summary["overall"], **summary["by_source"]}.items():
        m = data["part_b"]
        cm = m["confusion_matrix"]
        lines.append(
            ",".join(
                [
                    source,
                    str(cm["module_keep__gt_keep"]),
                    str(cm["module_skip__gt_skip"]),
                    str(cm["module_keep__gt_skip"]),
                    str(cm["module_skip__gt_keep_dangerous"]),
                    str(m["accuracy"]),
                    str(m["precision"]),
                    str(m["recall"]),
                    str(m["f1"]),
                    str(m["dangerous_skip_rate_among_should_keep"]),
                ]
            )
        )
    (eval_dir / "confusion_matrix.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(summary, results_dir / "metrics_summary.json")
    (results_dir / "confusion_matrix.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[stage04] n={len(rows)} -> {eval_dir}")


def stage05_sample_human_check(
    n: int,
    seed: int,
    balanced: bool = True,
    variant: str | None = None,
) -> None:
    _ensure_dirs()
    eval_dir = _variant_dir(DIR_04, variant)
    human_dir = _variant_dir(DIR_05, variant)
    rows = _load(eval_dir / "keep_skip_vs_coverage.jsonl")
    af_map = {str(r["af_id"]): r for r in _load(DIR_01 / "af_article.jsonl")}
    rng = random.Random(seed)

    if balanced:
        covered = [r for r in rows if r["ground_truth_keep"]]
        not_covered = [r for r in rows if not r["ground_truth_keep"]]
        half = n // 2
        sample = rng.sample(covered, min(half, len(covered))) + rng.sample(
            not_covered, min(n - half, len(not_covered))
        )
        if len(sample) < n:
            remaining = [r for r in rows if r not in sample]
            sample += rng.sample(remaining, min(n - len(sample), len(remaining)))
    else:
        sample = rng.sample(rows, min(n, len(rows)))
    rng.shuffle(sample)

    out_rows: list[dict[str, Any]] = []
    for i, r in enumerate(sample, start=1):
        af = af_map.get(str(r["af_id"]), {})
        out_rows.append(
            {
                "sample_id": f"exp2_human_{i:03d}",
                "af_id": r["af_id"],
                "source_dataset": r["source_dataset"],
                "article_id": r["article_id"],
                "fact": r["fact"],
                "llm_covered": r["covered_by_summary"],
                "llm_supporting_span": r["supporting_span"],
                "module_keep": r["module_keep"],
                "module_reasoning": r["module_reasoning"],
                "lay_summary": af.get("lay_summary", ""),
                "human_covered": "",
                "human_notes": "",
            }
        )
    _save(out_rows, human_dir / "human_check_sample.jsonl")
    csv_path = human_dir / "human_check_sample.csv"
    fieldnames = [
        "sample_id",
        "af_id",
        "source_dataset",
        "article_id",
        "fact",
        "llm_covered",
        "llm_supporting_span",
        "module_keep",
        "module_reasoning",
        "lay_summary",
        "human_covered",
        "human_notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "variant": variant,
            "seed": seed,
            "requested_n": n,
            "actual_n": len(out_rows),
            "balanced": balanced,
            "covered_count": sum(1 for r in out_rows if r["llm_covered"]),
            "not_covered_count": sum(1 for r in out_rows if not r["llm_covered"]),
        },
        human_dir / "human_check_metadata.json",
    )
    print(f"[stage05] sample={len(out_rows)} -> {human_dir / 'human_check_sample.jsonl'} and {csv_path}")


def _partc_dir(name: str) -> Path:
    d = DIR_PART_C / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def partc_extract_summary_af(
    model: str,
    pilot_per_source: int | None = None,
    resume: bool = True,
) -> None:
    _ensure_dirs()
    pilot_ids = _pilot_article_ids(pilot_per_source)
    articles = _filter_rows_by_articles(_load(DIR_00 / "articles_50.jsonl"), pilot_ids)
    if pilot_per_source is not None:
        print(f"[partc_summary_af] pilot_per_source={pilot_per_source} articles={len(articles)}")

    out_dir = _partc_dir("01_summary_af")
    out_af = out_dir / "summary_af.jsonl"
    out_meta = out_dir / "summary_af_metadata.json"
    prompt = (PROMPTS_EXP2_DIR / "partc_summary_af_extraction.txt").read_text(encoding="utf-8")
    client = OpenAIClient()

    rows = _load(out_af) if resume and out_af.exists() else []
    done = {str(r.get("article_id")) for r in rows}
    failed: list[dict[str, Any]] = []
    total_in = total_out = total_tok = 0

    for art in tqdm(articles, desc="partc | summary AF extraction", total=len(articles)):
        article_id = str(art["id"])
        if resume and article_id in done:
            continue
        raw = ""
        try:
            resp = client.chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt.replace("{lay_summary}", str(art["lay_summary"])),
                    }
                ],
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            usage = resp.get("usage", {})
            total_in += int(usage.get("prompt_tokens", 0))
            total_out += int(usage.get("completion_tokens", 0))
            total_tok += int(usage.get("total_tokens", 0))
            raw = str(resp.get("content", ""))
            obj = json.loads(raw)
            facts = obj.get("summary_atomic_facts", [])
            if not isinstance(facts, list):
                facts = []
            seen: set[str] = set()
            cnt = 0
            for fact_obj in facts:
                if not isinstance(fact_obj, dict):
                    continue
                fact = str(fact_obj.get("fact", "")).strip()
                source_sentence = str(fact_obj.get("source_sentence", "")).strip()
                n = _norm(fact)
                if not fact or n in seen:
                    continue
                seen.add(n)
                cnt += 1
                rows.append(
                    {
                        "summary_af_id": f"{article_id}_summary_af_{cnt:04d}",
                        "article_id": article_id,
                        "source_dataset": art["source_dataset"],
                        "fact": fact,
                        "source_sentence": source_sentence,
                        "lay_summary": art["lay_summary"],
                    }
                )
        except Exception as exc:
            failed.append({"article_id": article_id, "error": str(exc), "raw_preview": raw[:300]})

        done.add(article_id)
        _save(rows, out_af)

    _save(rows, out_af)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "pilot_per_source": pilot_per_source,
            "articles_total": len(articles),
            "summary_af_total": len(rows),
            "summary_af_by_source": dict(Counter(str(r["source_dataset"]) for r in rows)),
            "summary_af_by_article": dict(Counter(str(r["article_id"]) for r in rows)),
            "failed_items": failed,
            "total_input_tokens_this_run": total_in,
            "total_output_tokens_this_run": total_out,
            "total_tokens_this_run": total_tok,
        },
        out_meta,
    )
    print(f"[partc_summary_af] summary_AF={len(rows)} failed={len(failed)} -> {out_dir}")


def partc_extract_keep_af(
    model: str,
    chunk_words: int,
    overlap_words: int,
    pilot_per_source: int | None = None,
    resume: bool = True,
) -> None:
    _ensure_dirs()
    pilot_ids = _pilot_article_ids(pilot_per_source)
    articles = _filter_rows_by_articles(_load(DIR_00 / "articles_50.jsonl"), pilot_ids)
    if pilot_per_source is not None:
        print(f"[partc_keep_af] pilot_per_source={pilot_per_source} articles={len(articles)}")

    out_dir = _partc_dir("02_selected_keep_af")
    out_af = out_dir / "selected_keep_af.jsonl"
    out_meta = out_dir / "selected_keep_af_metadata.json"
    prompt = (PROMPTS_EXP2_DIR / "selective_keep_af_ultra_recall.txt").read_text(encoding="utf-8")
    client = OpenAIClient()

    rows = _load(out_af) if resume and out_af.exists() else []
    done = {str(r.get("article_id")) for r in rows}
    failed: list[dict[str, Any]] = []
    total_in = total_out = total_tok = 0

    for art in tqdm(articles, desc="partc | selected keep AF extraction", total=len(articles)):
        article_id = str(art["id"])
        if resume and article_id in done:
            continue
        seen: set[str] = set()
        cnt = 0
        for ch in _chunk_words(str(art["article"]), chunk_words, overlap_words=overlap_words):
            raw = ""
            try:
                resp = client.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt.replace("{source_text}", ch["text"]),
                        }
                    ],
                    model=model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                usage = resp.get("usage", {})
                total_in += int(usage.get("prompt_tokens", 0))
                total_out += int(usage.get("completion_tokens", 0))
                total_tok += int(usage.get("total_tokens", 0))
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
                            "selected_keep_af_id": f"{article_id}_selected_keep_af_{cnt:04d}",
                            "article_id": article_id,
                            "source_dataset": art["source_dataset"],
                            "fact": fact,
                            "source_span": source_span,
                            "keep_reasoning": reasoning,
                            "article_chunk_idx": ch["chunk_idx"],
                            "chunk_start_word": ch["start_word"],
                            "chunk_end_word": ch["end_word"],
                        }
                    )
            except Exception as exc:
                failed.append(
                    {
                        "article_id": article_id,
                        "chunk_idx": ch["chunk_idx"],
                        "error": str(exc),
                        "raw_preview": raw[:300],
                    }
                )

        done.add(article_id)
        _save(rows, out_af)

    _save(rows, out_af)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "chunk_words": chunk_words,
            "overlap_words": overlap_words,
            "pilot_per_source": pilot_per_source,
            "articles_total": len(articles),
            "selected_keep_af_total": len(rows),
            "selected_keep_af_by_source": dict(Counter(str(r["source_dataset"]) for r in rows)),
            "selected_keep_af_by_article": dict(Counter(str(r["article_id"]) for r in rows)),
            "failed_chunks": failed,
            "failed_chunk_count": len(failed),
            "total_input_tokens_this_run": total_in,
            "total_output_tokens_this_run": total_out,
            "total_tokens_this_run": total_tok,
        },
        out_meta,
    )
    print(f"[partc_keep_af] selected_keep_AF={len(rows)} failed_chunks={len(failed)} -> {out_dir}")


def partc_eval_selected_keep_af(
    model: str,
    pilot_per_source: int | None = None,
    resume: bool = True,
) -> None:
    _ensure_dirs()
    pilot_ids = _pilot_article_ids(pilot_per_source)
    summary_af = _filter_rows_by_articles(_load(_partc_dir("01_summary_af") / "summary_af.jsonl"), pilot_ids)
    selected_keep = _filter_rows_by_articles(
        _load(_partc_dir("02_selected_keep_af") / "selected_keep_af.jsonl"), pilot_ids
    )
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in selected_keep:
        by_article[str(r["article_id"])].append(r)

    out_dir = _partc_dir("03_eval")
    out_rows_path = out_dir / "summary_af_coverage_by_selected_keep.jsonl"
    out_meta = out_dir / "partc_metrics_summary.json"
    prompt = (PROMPTS_EXP2_DIR / "partc_selected_keep_covers_summary_af.txt").read_text(encoding="utf-8")
    client = OpenAIClient()

    rows = _load(out_rows_path) if resume and out_rows_path.exists() else []
    done = {str(r.get("summary_af_id")) for r in rows}
    parse_errors = 0

    for af in tqdm(summary_af, desc="partc | selected keep covers summary AF", total=len(summary_af)):
        sid = str(af["summary_af_id"])
        if resume and sid in done:
            continue
        cands = by_article.get(str(af["article_id"]), [])
        selected_text = "\n".join(
            f"- {r['selected_keep_af_id']}: {r['fact']}" for r in cands
        )
        if not selected_text:
            selected_text = "NONE"

        covered = False
        support_id: str | None = None
        support_fact: str | None = None
        support_ids: list[str] = []
        support_facts: list[str] = []
        reasoning = ""
        parse_error = False
        try:
            resp = client.chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt.replace("{summary_fact}", str(af["fact"])).replace(
                            "{selected_keep_facts}", selected_text
                        ),
                    }
                ],
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = json.loads(str(resp.get("content", "")))
            covered = _coerce_bool(obj.get("covered", False))
            raw_ids = obj.get("supporting_keep_af_ids", obj.get("supporting_keep_af_id", []))
            raw_facts = obj.get("supporting_keep_facts", obj.get("supporting_keep_fact", []))
            if not isinstance(raw_ids, list):
                raw_ids = [raw_ids]
            if not isinstance(raw_facts, list):
                raw_facts = [raw_facts]
            support_ids = [
                str(x).strip()
                for x in raw_ids
                if x is not None and str(x).strip() not in {"", "null", "None"}
            ]
            support_facts = [
                str(x).strip()
                for x in raw_facts
                if x is not None and str(x).strip() not in {"", "null", "None"}
            ]
            support_id = support_ids[0] if support_ids else None
            support_fact = support_facts[0] if support_facts else None
            reasoning = str(obj.get("reasoning", "")).strip()
        except Exception as exc:
            parse_error = True
            parse_errors += 1
            reasoning = f"parse_or_runtime_error: {exc}"

        rows.append(
            {
                "summary_af_id": sid,
                "article_id": af["article_id"],
                "source_dataset": af["source_dataset"],
                "summary_fact": af["fact"],
                "covered_by_selected_keep_af": covered,
                "supporting_keep_af_id": support_id,
                "supporting_keep_fact": support_fact,
                "supporting_keep_af_ids": support_ids,
                "supporting_keep_facts": support_facts,
                "selected_keep_af_count_for_article": len(cands),
                "reasoning": reasoning,
                "parse_error": parse_error,
            }
        )
        done.add(sid)
        _save(rows, out_rows_path)

    valid = [r for r in rows if not r.get("parse_error", False)]
    total = len(valid)
    covered_count = sum(1 for r in valid if r["covered_by_selected_keep_af"])
    selected_by_article = Counter(str(r["article_id"]) for r in selected_keep)
    summary_by_article = Counter(str(r["article_id"]) for r in summary_af)
    metrics = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "pilot_per_source": pilot_per_source,
        "summary_af_total": len(summary_af),
        "selected_keep_af_total": len(selected_keep),
        "valid_eval_rows": total,
        "parse_error_count": sum(1 for r in rows if r.get("parse_error")),
        "summary_fact_recall": round(covered_count / total, 4) if total else 0.0,
        "covered_summary_af_count": covered_count,
        "missed_summary_af_count": total - covered_count,
        "selected_keep_af_by_source": dict(Counter(str(r["source_dataset"]) for r in selected_keep)),
        "summary_af_by_source": dict(Counter(str(r["source_dataset"]) for r in summary_af)),
        "by_source": {},
        "by_article": {},
    }
    for source in sorted({str(r["source_dataset"]) for r in valid}):
        sub = [r for r in valid if str(r["source_dataset"]) == source]
        cov = sum(1 for r in sub if r["covered_by_selected_keep_af"])
        metrics["by_source"][source] = {
            "summary_af_total": len(sub),
            "covered": cov,
            "summary_fact_recall": round(cov / len(sub), 4) if sub else 0.0,
        }
    for article_id in sorted(set(summary_by_article) | set(selected_by_article)):
        sub = [r for r in valid if str(r["article_id"]) == article_id]
        cov = sum(1 for r in sub if r["covered_by_selected_keep_af"])
        metrics["by_article"][article_id] = {
            "summary_af_total": len(sub),
            "selected_keep_af_total": selected_by_article.get(article_id, 0),
            "covered": cov,
            "summary_fact_recall": round(cov / len(sub), 4) if sub else 0.0,
        }

    save_json(metrics, out_meta)
    (out_dir / "partc_summary.csv").write_text(
        "metric,value\n"
        f"summary_af_total,{metrics['summary_af_total']}\n"
        f"selected_keep_af_total,{metrics['selected_keep_af_total']}\n"
        f"summary_fact_recall,{metrics['summary_fact_recall']}\n"
        f"covered_summary_af_count,{metrics['covered_summary_af_count']}\n"
        f"missed_summary_af_count,{metrics['missed_summary_af_count']}\n",
        encoding="utf-8",
    )
    print(
        f"[partc_eval] summary_recall={metrics['summary_fact_recall']} "
        f"covered={covered_count}/{total} parse_errors={parse_errors} -> {out_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2 pipeline runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("stage00_prepare_inputs")
    p0.add_argument("--n-per-source", type=int, default=25)
    p0.add_argument("--seed", type=int, default=20260620)
    p0.add_argument("--include-exp0-overlap", action="store_true")

    p1 = sub.add_parser("stage01_extract_article_af")
    p1.add_argument("--model", type=str, default="gpt-4.1")
    p1.add_argument("--chunk-words", type=int, default=1200)
    p1.add_argument("--overlap-words", type=int, default=120)
    p1.add_argument("--no-resume", action="store_true")

    p1b = sub.add_parser("stage01b_extract_keep_af_ultra_recall")
    p1b.add_argument("--model", type=str, default="gpt-4.1")
    p1b.add_argument("--chunk-words", type=int, default=1200)
    p1b.add_argument("--overlap-words", type=int, default=120)
    p1b.add_argument("--pilot-per-source", type=int, default=None)
    p1b.add_argument("--no-resume", action="store_true")

    p2 = sub.add_parser("stage02_coverage_setup_c")
    p2.add_argument("--model", type=str, default="gpt-4.1")
    p2.add_argument("--no-reuse-questions", action="store_true")
    p2.add_argument("--no-resume", action="store_true")
    p2.add_argument("--pilot-per-source", type=int, default=None)

    p2b = sub.add_parser("stage02b_direct_coverage")
    p2b.add_argument("--model", type=str, default="gpt-4.1")
    p2b.add_argument("--no-resume", action="store_true")
    p2b.add_argument("--pilot-per-source", type=int, default=None)

    p3 = sub.add_parser("stage03_keep_skip")
    p3.add_argument("--model", type=str, default="gpt-4.1")
    p3.add_argument("--no-resume", action="store_true")
    p3.add_argument("--pilot-per-source", type=int, default=None)
    p3.add_argument("--variant", type=str, default=None)

    p4 = sub.add_parser("stage04_eval")
    p4.add_argument("--pilot-per-source", type=int, default=None)
    p4.add_argument("--variant", type=str, default=None)

    p5 = sub.add_parser("stage05_sample_human_check")
    p5.add_argument("--n", type=int, default=80)
    p5.add_argument("--seed", type=int, default=20260621)
    p5.add_argument("--unbalanced", action="store_true")
    p5.add_argument("--variant", type=str, default=None)

    pc1 = sub.add_parser("partc_extract_summary_af")
    pc1.add_argument("--model", type=str, default="gpt-4.1")
    pc1.add_argument("--pilot-per-source", type=int, default=None)
    pc1.add_argument("--no-resume", action="store_true")

    pc2 = sub.add_parser("partc_extract_keep_af")
    pc2.add_argument("--model", type=str, default="gpt-4.1")
    pc2.add_argument("--chunk-words", type=int, default=1200)
    pc2.add_argument("--overlap-words", type=int, default=120)
    pc2.add_argument("--pilot-per-source", type=int, default=None)
    pc2.add_argument("--no-resume", action="store_true")

    pc3 = sub.add_parser("partc_eval_selected_keep_af")
    pc3.add_argument("--model", type=str, default="gpt-4.1")
    pc3.add_argument("--pilot-per-source", type=int, default=None)
    pc3.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()
    if args.cmd == "stage00_prepare_inputs":
        stage00_prepare_inputs(
            n_per_source=args.n_per_source,
            seed=args.seed,
            exclude_exp0=not args.include_exp0_overlap,
        )
    elif args.cmd == "stage01_extract_article_af":
        stage01_extract_article_af(
            model=args.model,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
            resume=not args.no_resume,
        )
    elif args.cmd == "stage01b_extract_keep_af_ultra_recall":
        stage01b_extract_keep_af_ultra_recall(
            model=args.model,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
            pilot_per_source=args.pilot_per_source,
            resume=not args.no_resume,
        )
    elif args.cmd == "stage02_coverage_setup_c":
        stage02_coverage_setup_c(
            model=args.model,
            reuse_questions=not args.no_reuse_questions,
            resume=not args.no_resume,
            pilot_per_source=args.pilot_per_source,
        )
    elif args.cmd == "stage02b_direct_coverage":
        stage02b_direct_coverage(
            model=args.model,
            resume=not args.no_resume,
            pilot_per_source=args.pilot_per_source,
        )
    elif args.cmd == "stage03_keep_skip":
        stage03_keep_skip(
            model=args.model,
            resume=not args.no_resume,
            pilot_per_source=args.pilot_per_source,
            variant=args.variant,
        )
    elif args.cmd == "stage04_eval":
        stage04_eval(pilot_per_source=args.pilot_per_source, variant=args.variant)
    elif args.cmd == "stage05_sample_human_check":
        stage05_sample_human_check(
            n=args.n,
            seed=args.seed,
            balanced=not args.unbalanced,
            variant=args.variant,
        )
    elif args.cmd == "partc_extract_summary_af":
        partc_extract_summary_af(
            model=args.model,
            pilot_per_source=args.pilot_per_source,
            resume=not args.no_resume,
        )
    elif args.cmd == "partc_extract_keep_af":
        partc_extract_keep_af(
            model=args.model,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
            pilot_per_source=args.pilot_per_source,
            resume=not args.no_resume,
        )
    elif args.cmd == "partc_eval_selected_keep_af":
        partc_eval_selected_keep_af(
            model=args.model,
            pilot_per_source=args.pilot_per_source,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()
