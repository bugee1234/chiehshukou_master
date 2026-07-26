from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from src.experiment_2.run_three_model_exp2 import MODEL_CONFIGS
from src.thesis_laysumm.paths import run_data_dir
from src.thesis_laysumm_v11.run_phase_b import answer_questions, generate_summaries, rewrite_summaries
from src.thesis_laysumm_v11.validate_outputs import validate_run
from src.utils import load_jsonl, save_json, save_jsonl


ROOT_DIR = Path(__file__).resolve().parents[2]
UPSTREAM_MODEL_KEY = "gemini25_flash_non_thinking"
DEFAULT_SOURCE_RUN = "pilot_n20_v11_fixed_m12_gemini25_same_model_phaseb_factuality_chase"
FINAL_284_RUN = "v11_constellation_validation284_fixed_m12_pilot20"
REFERENCE_RUN = "atlas_v11_gemini25_fixed_upstream_phaseb_n10_compact"
DEFAULT_ARTICLES_PER_SOURCE = 5
OFFICIAL_BACKBONE_MODEL_KEYS = {
    "qwen25_7b_instruct_openrouter",
    "llama3_8b_instruct_openrouter",
    "llama31_8b_instruct_openrouter",
    "qwen25_7b_instruct_local_8bit",
    "llama3_8b_instruct_local_8bit",
}


def _articles_path(run_name: str) -> Path:
    return run_data_dir(run_name) / "00_articles" / "articles.jsonl"


def _selected_articles(
    source_run: str,
    articles_per_source: int = DEFAULT_ARTICLES_PER_SOURCE,
) -> list[dict[str, Any]]:
    if articles_per_source < 1:
        raise ValueError("articles_per_source must be at least 1")
    source_path = _articles_path(source_run)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing Gemini 2.5 pilot source: {source_path}")
    source_rows = load_jsonl(source_path)
    questions_path = (
        run_data_dir(source_run)
        / "03_questions_v11"
        / "factuality_chase"
        / UPSTREAM_MODEL_KEY
        / "questions_1t3f_nota.jsonl"
    )
    if not questions_path.exists():
        raise FileNotFoundError(f"Missing fixed Gemini 2.5 questions: {questions_path}")
    question_counts = Counter(str(row.get("article_id")) for row in load_jsonl(questions_path))
    selected: list[dict[str, Any]] = []
    for source in ("PLOS", "eLife"):
        rows = [row for row in source_rows if str(row.get("source_dataset")) == source]
        if len(rows) < articles_per_source:
            raise ValueError(f"Expected at least {articles_per_source} {source} rows in {source_path}")
        rows.sort(key=lambda row: (question_counts[str(row["id"])], int(row["original_index"])))
        selected.extend(rows[:articles_per_source])
    counts = Counter(str(row.get("source_dataset")) for row in selected)
    expected_counts = Counter({"PLOS": articles_per_source, "eLife": articles_per_source})
    if len(selected) != 2 * articles_per_source or counts != expected_counts:
        raise AssertionError(f"Unexpected fixed subset: {dict(counts)}")

    final_path = _articles_path(FINAL_284_RUN)
    if not final_path.exists():
        raise FileNotFoundError(f"Missing final 284-article thesis run: {final_path}")
    final_ids = {str(row["id"]) for row in load_jsonl(final_path)}
    missing = sorted({str(row["id"]) for row in selected} - final_ids)
    if missing:
        raise ValueError(f"Selected pilot rows are not all in the final 284 validation run: {missing}")
    return selected


def _write_articles(*, run_name: str, rows: list[dict[str, Any]]) -> None:
    target = _articles_path(run_name)
    selected_ids = [str(row["id"]) for row in rows]
    if target.exists():
        existing_ids = [str(row["id"]) for row in load_jsonl(target)]
        if existing_ids != selected_ids:
            raise ValueError(
                f"Refusing to reuse {run_name!r}: its articles differ from the fixed 5+5 subset."
            )
    else:
        save_jsonl(rows, target)


def _filter_jsonl(source: Path, target: Path, article_ids: set[str]) -> int:
    if not source.exists():
        raise FileNotFoundError(source)
    rows = load_jsonl(source)
    filtered = [
        row
        for row in rows
        if str(row.get("article_id") or row.get("id")) in article_ids
    ]
    if not filtered:
        raise ValueError(f"No selected rows found in {source}")
    save_jsonl(filtered, target)
    return len(filtered)


def prepare_fixed_gemini_upstream(
    *,
    run_name: str,
    target_model_key: str,
    source_run: str = DEFAULT_SOURCE_RUN,
    articles_per_source: int = DEFAULT_ARTICLES_PER_SOURCE,
) -> dict[str, Any]:
    rows = _selected_articles(source_run, articles_per_source=articles_per_source)
    _write_articles(run_name=run_name, rows=rows)
    article_ids = {str(row["id"]) for row in rows}
    source_dir = run_data_dir(source_run)
    target_dir = run_data_dir(run_name)

    evidence_source = source_dir / "02_5_evidence_table" / "factuality_chase" / UPSTREAM_MODEL_KEY
    evidence_target = target_dir / "02_5_evidence_table" / "factuality_chase" / target_model_key
    questions_source = source_dir / "03_questions_v11" / "factuality_chase" / UPSTREAM_MODEL_KEY
    questions_target = target_dir / "03_questions_v11" / "factuality_chase" / target_model_key
    copied_counts = {
        name: _filter_jsonl(evidence_source / name, evidence_target / name, article_ids)
        for name in ("core_keep_af.jsonl", "check_af.jsonl", "evidence_table.jsonl")
    }
    copied_counts["questions_1t3f_nota.jsonl"] = _filter_jsonl(
        questions_source / "questions_1t3f_nota.jsonl",
        questions_target / "questions_1t3f_nota.jsonl",
        article_ids,
    )
    manifest = {
        "design": "fixed_gemini25_upstream_phase_b_backbone_ablation",
        "run_name": run_name,
        "target_model_key": target_model_key,
        "target_model_config": MODEL_CONFIGS[target_model_key],
        "source_run": source_run,
        "upstream_model_key": UPSTREAM_MODEL_KEY,
        "mode": "factuality_chase",
        "articles_per_source": articles_per_source,
        "selection_strategy": (
            f"Within each source, select the {articles_per_source} articles with the fewest fixed Gemini 2.5 verification "
            "questions; break ties by original_index. This bounds local Phase-B runtime and is fixed before "
            "running Qwen or Llama."
        ),
        "article_count": len(rows),
        "article_ids": [str(row["id"]) for row in rows],
        "question_counts_by_article": dict(
            Counter(
                str(row.get("article_id"))
                for row in load_jsonl(
                    run_data_dir(source_run)
                    / "03_questions_v11"
                    / "factuality_chase"
                    / UPSTREAM_MODEL_KEY
                    / "questions_1t3f_nota.jsonl"
                )
                if str(row.get("article_id")) in {str(article["id"]) for article in rows}
            )
        ),
        "copied_counts": copied_counts,
        "model_specific_stages": ["generate-summary", "answer-questions", "rewrite-summaries"],
    }
    save_json(manifest, target_dir / "fixed_upstream_manifest.json")
    print(
        f"[ATLAS backbone] Gemini 2.5 upstream ready: {run_name} "
        f"({articles_per_source} PLOS + {articles_per_source} eLife)"
    )
    return manifest


def prepare_gemini_reference(*, source_run: str = DEFAULT_SOURCE_RUN) -> dict[str, Any]:
    manifest = prepare_fixed_gemini_upstream(
        run_name=REFERENCE_RUN,
        target_model_key=UPSTREAM_MODEL_KEY,
        source_run=source_run,
    )
    article_ids = set(manifest["article_ids"])
    source_dir = run_data_dir(source_run)
    target_dir = run_data_dir(REFERENCE_RUN)
    paths = [
        ("04_summaries", "generated_summaries.jsonl"),
        ("06_rewritten", "rewritten_summaries.jsonl"),
    ]
    for stage, filename in paths:
        _filter_jsonl(
            source_dir / stage / UPSTREAM_MODEL_KEY / filename,
            target_dir / stage / UPSTREAM_MODEL_KEY / filename,
            article_ids,
        )
    answer_source = source_dir / "05_module3_answers_v11" / "factuality_chase" / UPSTREAM_MODEL_KEY
    answer_target = target_dir / "05_module3_answers_v11" / "factuality_chase" / UPSTREAM_MODEL_KEY
    for filename in ("module3_answers.jsonl", "wrong_answers.jsonl"):
        source = answer_source / filename
        if source.exists():
            _filter_jsonl(source, answer_target / filename, article_ids)
    print(f"[local ATLAS] Gemini 2.5 n10 reference ready: {REFERENCE_RUN}")
    return manifest


def run_pipeline(
    *,
    run_name: str,
    model_key: str,
    source_run: str,
    articles_per_source: int,
    max_workers: int,
    resume: bool,
) -> None:
    if model_key not in OFFICIAL_BACKBONE_MODEL_KEYS:
        raise ValueError(f"--model-key must be one of {sorted(OFFICIAL_BACKBONE_MODEL_KEYS)}")
    if max_workers < 1:
        raise ValueError("--max-workers must be at least 1")
    manifest = prepare_fixed_gemini_upstream(
        run_name=run_name,
        target_model_key=model_key,
        source_run=source_run,
        articles_per_source=articles_per_source,
    )
    validate_run(
        run_name=run_name,
        model_key=model_key,
        mode="factuality_chase",
        stage="questions",
        require_eval=False,
    )
    common = {
        "run_name": run_name,
        "model_key": model_key,
        "mode": "factuality_chase",
        "max_workers": max_workers,
        "resume": resume,
    }
    question_count = int((manifest.get("copied_counts") or {}).get("questions_1t3f_nota.jsonl") or 0)
    print(
        f"[Phase B 1/3] Generate summaries: {2 * articles_per_source} articles | "
        f"workers={max_workers} | model={model_key}",
        flush=True,
    )
    generate_summaries(**common)
    validate_run(run_name=run_name, model_key=model_key, mode="factuality_chase", stage="summaries", require_eval=False)
    print(
        f"[Phase B 2/3] Answer verification questions: {question_count} questions | "
        f"workers={max_workers} | model={model_key}",
        flush=True,
    )
    answer_questions(**common)
    validate_run(run_name=run_name, model_key=model_key, mode="factuality_chase", stage="answers", require_eval=False)
    print(
        f"[Phase B 3/3] Rewrite summaries: {2 * articles_per_source} articles | "
        f"workers={max_workers} | model={model_key}",
        flush=True,
    )
    rewrite_summaries(**common)
    validate_run(run_name=run_name, model_key=model_key, mode="factuality_chase", stage="rewritten", require_eval=False)
    print(f"[ATLAS backbone] Phase B complete: {run_name} / {model_key}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an off-the-shelf backbone through ATLAS Phase B with fixed Gemini 2.5 upstream artifacts."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-key", choices=sorted(OFFICIAL_BACKBONE_MODEL_KEYS), required=True)
    parser.add_argument("--source-run", default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--articles-per-source", type=int, default=DEFAULT_ARTICLES_PER_SOURCE)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(
        run_name=args.run_name,
        model_key=args.model_key,
        source_run=args.source_run,
        articles_per_source=args.articles_per_source,
        max_workers=args.max_workers,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
