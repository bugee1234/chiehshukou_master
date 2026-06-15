from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset, load_dataset

from src import config

DATASETS: dict[str, tuple[str, str]] = {
    "PLOS": ("BioLaySumm/BioLaySumm2025-PLOS", "plos"),
    "eLife": ("BioLaySumm/BioLaySumm2025-eLife", "elife"),
}

EXPECTED_TEST_SIZE = 142
THESIS_SAMPLE_PER_SOURCE = 142
VALIDATION_POOL_SIZE = {
    "PLOS": 1376,
    "eLife": 241,
}


@dataclass
class ArticleValidation:
    source_dataset: str
    original_index: int
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_blocks: int = 0
    n_headings: int = 0
    first_heading: str = ""
    abstract_word_count: int = 0
    article_word_count: int = 0
    expert_summary_word_count: int = 0


def word_count(text: str | None) -> int:
    return len(re.findall(r"\S+", str(text or "")))


def load_biolaysumm_split(source: str, split: str) -> Dataset:
    if source not in DATASETS:
        raise ValueError(f"Unknown source {source!r}; expected one of {sorted(DATASETS)}")
    dataset_name, _ = DATASETS[source]
    return load_dataset(dataset_name, split=split, token=config.HF_TOKEN or None)


def extract_article_sections(article: str, section_headings: list[str] | None) -> dict[str, Any]:
    """Split article into major blocks; first block is abstract when heading says so."""
    text = str(article or "")
    blocks = text.split("\n") if text else []
    headings = [str(h).strip() for h in (section_headings or [])]

    abstract = ""
    body_blocks: list[str] = []
    if blocks and headings and headings[0].lower() == "abstract":
        abstract = blocks[0].strip()
        body_blocks = [b.strip() for b in blocks[1:] if b.strip()]
    elif blocks:
        abstract = blocks[0].strip()
        body_blocks = [b.strip() for b in blocks[1:] if b.strip()]
        if not headings or headings[0].lower() != "abstract":
            headings = ["Abstract"] + headings

    article_body = "\n".join(body_blocks).strip()
    return {
        "blocks": blocks,
        "section_headings": headings,
        "abstract": abstract,
        "article_body": article_body,
    }


def build_eval_document(abstract: str, article: str) -> str:
    """Format used by BioLaySumm official evaluation (LENS reads first line as abstract)."""
    abstract = str(abstract or "").strip()
    article = str(article or "").strip()
    if abstract and article:
        return f"{abstract}\n{article}"
    return article or abstract


def _reference_text(item: dict[str, Any]) -> str:
    for key in ("lay_summary", "summary", "reference"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value)
    for key in ("lay_summary", "summary", "reference"):
        if key in item:
            return str(item.get(key, "") or "")
    return ""


def validate_article_row(
    *,
    source_dataset: str,
    original_index: int,
    item: dict[str, Any],
    require_expert_summary: bool,
) -> ArticleValidation:
    result = ArticleValidation(
        source_dataset=source_dataset,
        original_index=original_index,
        ok=True,
    )

    article = str(item.get("article", "") or "")
    expert_summary = _reference_text(item)
    headings_raw = item.get("section_headings") or []
    if not isinstance(headings_raw, list):
        result.errors.append("section_headings is not a list")
        result.ok = False
        headings_raw = []

    if not article.strip():
        result.errors.append("article is empty")
        result.ok = False

    sections = extract_article_sections(article, headings_raw)
    blocks = sections["blocks"]
    headings = sections["section_headings"]
    abstract = sections["abstract"]

    result.n_blocks = len(blocks)
    result.n_headings = len(headings)
    result.first_heading = headings[0] if headings else ""
    result.abstract_word_count = word_count(abstract)
    result.article_word_count = word_count(article)
    result.expert_summary_word_count = word_count(expert_summary)

    if not headings:
        result.errors.append("section_headings is empty")
        result.ok = False
    elif headings[0].lower() != "abstract":
        result.errors.append(f"first section heading is {headings[0]!r}, expected Abstract")
        result.ok = False

    if not abstract:
        result.errors.append("abstract block is empty")
        result.ok = False
    elif result.abstract_word_count < 20:
        result.warnings.append(f"abstract is very short ({result.abstract_word_count} words)")

    if require_expert_summary and not expert_summary.strip():
        result.errors.append("expert summary is empty")
        result.ok = False

    if result.n_headings != result.n_blocks:
        result.warnings.append(
            f"section_headings ({result.n_headings}) != article blocks ({result.n_blocks}); "
            "using first block as abstract"
        )

    # UTF-8 roundtrip sanity check for evaluation I/O.
    for label, text in [
        ("article", article),
        ("abstract", abstract),
        ("expert_summary", expert_summary),
    ]:
        try:
            encoded = text.encode("utf-8")
            if encoded.decode("utf-8") != text:
                result.errors.append(f"{label} failed UTF-8 roundtrip")
                result.ok = False
        except UnicodeEncodeError as exc:
            result.errors.append(f"{label} is not valid UTF-8: {exc}")
            result.ok = False

    return result


def article_record_from_item(
    *,
    source_dataset: str,
    dataset_name: str,
    split: str,
    original_index: int,
    item: dict[str, Any],
    run_order: int,
    article_id: str,
) -> dict[str, Any]:
    article = str(item.get("article", "") or "")
    expert_summary = _reference_text(item)
    title = str(item.get("title", "") or "")
    keywords = item.get("keywords")
    year = item.get("year")
    sections = extract_article_sections(article, item.get("section_headings") or [])
    abstract = sections["abstract"]
    document = build_eval_document(abstract, article)

    return {
        "id": article_id,
        "source_dataset": source_dataset,
        "dataset_name": dataset_name,
        "split": split,
        "original_index": int(original_index),
        "title": title,
        "year": year,
        "keywords": keywords,
        "section_headings": sections["section_headings"],
        "article": article,
        "abstract": abstract,
        "article_body": sections["article_body"],
        "document": document,
        "expert_summary": expert_summary,
        "article_word_count": word_count(article),
        "abstract_word_count": word_count(abstract),
        "expert_summary_word_count": word_count(expert_summary),
        "run_order": run_order,
    }


def eval_reference_record(document: str, reference: str) -> dict[str, str]:
    return {"document": document, "reference": reference}


def select_indices(
    dataset_size: int,
    n_per_source: int | None,
    *,
    selection_mode: str,
    seed: int,
) -> list[int]:
    indices = list(range(dataset_size))
    if n_per_source is None:
        return indices
    if dataset_size < n_per_source:
        raise RuntimeError(f"split has {dataset_size} rows, need {n_per_source}")
    if selection_mode == "first":
        return indices[:n_per_source]
    if selection_mode == "random":
        import random

        rng = random.Random(seed)
        return sorted(rng.sample(indices, n_per_source))
    raise ValueError("selection_mode must be 'first' or 'random'")


_ABSTRACT_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_abstract_sentences(abstract: str) -> list[str]:
    """Split abstract into sentence-like units for abstract-aligned Module 2 judging."""
    text = re.sub(r"\s+", " ", str(abstract or "").strip())
    if not text:
        return []
    parts = [p.strip() for p in _ABSTRACT_SENTENCE_SPLIT.split(text) if p.strip()]
    return parts if parts else [text]


def validation_to_dict(result: ArticleValidation) -> dict[str, Any]:
    return {
        "source_dataset": result.source_dataset,
        "original_index": result.original_index,
        "ok": result.ok,
        "errors": result.errors,
        "warnings": result.warnings,
        "n_blocks": result.n_blocks,
        "n_headings": result.n_headings,
        "first_heading": result.first_heading,
        "abstract_word_count": result.abstract_word_count,
        "article_word_count": result.article_word_count,
        "expert_summary_word_count": result.expert_summary_word_count,
    }


def audit_dataset_split(
    *,
    source: str,
    split: str,
    require_expert_summary: bool,
) -> dict[str, Any]:
    ds = load_biolaysumm_split(source, split)
    if split == "test" and len(ds) != EXPECTED_TEST_SIZE:
        raise RuntimeError(
            f"{source}: expected BioLaySumm Task 1.1 test split size {EXPECTED_TEST_SIZE}, got {len(ds)}"
        )

    validations: list[dict[str, Any]] = []
    for idx in range(len(ds)):
        item = ds[int(idx)]
        result = validate_article_row(
            source_dataset=source,
            original_index=idx,
            item=item,
            require_expert_summary=require_expert_summary,
        )
        validations.append(validation_to_dict(result))

    ok_count = sum(1 for row in validations if row["ok"])
    return {
        "source_dataset": source,
        "split": split,
        "dataset_name": DATASETS[source][0],
        "split_size": len(ds),
        "columns": ds.column_names,
        "ok_count": ok_count,
        "error_count": len(validations) - ok_count,
        "empty_expert_summary_count": sum(
            1 for row in validations if row["expert_summary_word_count"] == 0
        ),
        "validations": validations,
    }
