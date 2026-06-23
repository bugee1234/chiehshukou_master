from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.thesis_laysumm_direct_baseline.paths import PROMPT_PATH
from src.thesis_laysumm_direct_baseline.run_generate import ALLOWED_PROMPT_FIELDS, build_prompt


def main() -> None:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    if set(ALLOWED_PROMPT_FIELDS) != {"article"}:
        raise AssertionError(f"Unexpected prompt fields: {ALLOWED_PROMPT_FIELDS}")
    sentinel = {
        "article": "VISIBLE_ARTICLE_SENTINEL",
        "title": "FORBIDDEN_TITLE_SENTINEL",
        "expert_summary": "FORBIDDEN_REFERENCE_SENTINEL",
        "evidence_rows": "FORBIDDEN_EVIDENCE_SENTINEL",
        "questions": "FORBIDDEN_QUESTIONS_SENTINEL",
        "generated_summary": "FORBIDDEN_PIPELINE_SUMMARY_SENTINEL",
    }
    prompt = build_prompt(sentinel, template)
    for expected in ("VISIBLE_ARTICLE_SENTINEL",):
        if expected not in prompt:
            raise AssertionError(f"Missing allowed input: {expected}")
    for forbidden in (
        "FORBIDDEN_TITLE_SENTINEL",
        "FORBIDDEN_REFERENCE_SENTINEL",
        "FORBIDDEN_EVIDENCE_SENTINEL",
        "FORBIDDEN_QUESTIONS_SENTINEL",
        "FORBIDDEN_PIPELINE_SUMMARY_SENTINEL",
    ):
        if forbidden in prompt:
            raise AssertionError(f"Prompt leakage detected: {forbidden}")
    if "{title}" in template or template.count("{article}") != 1:
        raise AssertionError("Prompt must contain article exactly once and must not contain title")
    print("[direct static check] ok: prompt uses raw article only; no title/pipeline/reference leakage")


if __name__ == "__main__":
    main()
