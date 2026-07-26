from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.thesis_laysumm_direct_baseline.paths import (
    PROMPT_PATH,
    inputs_path,
    references_path,
    summaries_path,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_word_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text or "").strip()))


def find_mojibake(text: str) -> list[str]:
    value = str(text or "")
    hits: list[str] = []
    if "\ufffd" in value:
        hits.append("replacement_character")
    if re.search(r"[\uE000-\uF8FF]", value):
        hits.append("private_use_character")
    if re.search(r"[\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]", value):
        hits.append("cjk_character")
    for match in re.finditer(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", value):
        hits.append(f"control_char_U+{ord(match.group(0)):04X}")
    return hits[:20]


FORBIDDEN_INPUT_KEYS = {
    "expert_summary",
    "evidence",
    "evidence_rows",
    "questions",
    "answers",
    "generated_summary",
    "rewritten_summary",
}
EXPECTED_VISIBLE_FIELDS = ["article"]


def _ids(rows: list[dict[str, Any]], key: str) -> list[str]:
    return [str(row.get(key) or "") for row in rows]


def validate(*, run_name: str, model_key: str | None, expected_count: int | None, prompt_path: Path) -> None:
    in_path = inputs_path(run_name)
    ref_path = references_path(run_name)
    if not in_path.exists() or not ref_path.exists():
        raise FileNotFoundError("Direct-baseline inputs/references have not been prepared")
    inputs = load_jsonl(in_path)
    refs = load_jsonl(ref_path)
    input_ids = _ids(inputs, "id")
    ref_ids = _ids(refs, "id")
    if len(input_ids) != len(set(input_ids)) or len(ref_ids) != len(set(ref_ids)):
        raise ValueError("Duplicate IDs in direct-baseline inputs or references")
    if set(input_ids) != set(ref_ids):
        raise ValueError("Direct-baseline input and reference IDs do not match")
    if expected_count is not None and len(inputs) != expected_count:
        raise ValueError(f"Expected {expected_count} articles, found {len(inputs)}")
    if any(str(row.get("split")) != "validation" for row in inputs):
        raise ValueError("Non-validation article found in direct baseline")
    for row in inputs:
        leaked = FORBIDDEN_INPUT_KEYS.intersection(row)
        if leaked:
            raise ValueError(f"Forbidden generation fields stored in 00_inputs: {sorted(leaked)}")
        if not str(row.get("article") or "").strip():
            raise ValueError(f"Empty article input: {row.get('id')}")
    if any(not str(row.get("expert_summary") or "").strip() for row in refs):
        raise ValueError("Empty expert reference")
    if any(not str(row.get("document") or "").strip() for row in refs):
        raise ValueError("Empty evaluation document")
    by_source = {
        source: sum(str(row.get("source_dataset")) == source for row in inputs)
        for source in ("PLOS", "eLife")
    }
    print(
        f"[direct validate] inputs ok run={run_name} articles={len(inputs)} "
        f"PLOS={by_source['PLOS']} eLife={by_source['eLife']} leakage=0"
    )

    if model_key is None:
        return
    out_path = summaries_path(run_name, model_key)
    if not out_path.exists():
        raise FileNotFoundError(f"Missing direct summaries: {out_path}")
    rows = load_jsonl(out_path)
    row_ids = _ids(rows, "article_id")
    if len(row_ids) != len(set(row_ids)):
        raise ValueError(f"Duplicate summary article IDs for {model_key}")
    if set(row_ids) != set(input_ids):
        missing = sorted(set(input_ids) - set(row_ids))
        extra = sorted(set(row_ids) - set(input_ids))
        raise ValueError(
            f"Direct summary IDs do not match inputs for {model_key}: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    prompt_hash = sha256_text(prompt_path.read_text(encoding="utf-8"))
    word_counts: list[int] = []
    mojibake_rows = 0
    for row in rows:
        aid = str(row.get("article_id"))
        summary = str(row.get("generated_summary") or "").strip()
        if row.get("parse_error") or not summary:
            raise ValueError(f"Empty/parse_error direct summary: {aid}")
        if row.get("model_key") != model_key:
            raise ValueError(f"Wrong model_key in direct summary: {aid}")
        if row.get("visible_input_fields") != EXPECTED_VISIBLE_FIELDS:
            raise ValueError(f"Unexpected visible input fields: {aid}")
        if row.get("prompt_sha256") != prompt_hash:
            raise ValueError(f"Prompt hash mismatch: {aid}")
        raw_response = str(row.get("raw_response") or "").strip()
        if not row.get("truncated_by_word_limit") and raw_response != summary:
            raise ValueError(f"Direct summary was altered after model response: {aid}")
        word_counts.append(safe_word_count(summary))
        mojibake_rows += bool(find_mojibake(summary))
    print(
        f"[direct validate] summaries ok model={model_key} rows={len(rows)} "
        f"wc_min={min(word_counts)} wc_median={statistics.median(word_counts):.1f} "
        f"wc_max={max(word_counts)} mojibake_warnings={mojibake_rows}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate isolated direct-baseline data and summaries.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-key")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--prompt-path", type=Path, default=PROMPT_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate(
        run_name=args.run_name,
        model_key=args.model_key,
        expected_count=args.expected_count,
        prompt_path=args.prompt_path,
    )


if __name__ == "__main__":
    main()
