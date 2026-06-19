from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V11 = ROOT / "src" / "thesis_laysumm_v11"
REQUIRED_FILES = [
    V11 / "__init__.py",
    V11 / "common.py",
    V11 / "run_evidence.py",
    V11 / "run_questions.py",
    V11 / "run_phase_b.py",
    V11 / "validate_outputs.py",
    V11 / "sanity_check.py",
    V11 / "run_pilot_n20_v11.ps1",
    V11 / "prompts" / "build_evidence_table.txt",
    V11 / "prompts" / "generate_summary_v11.txt",
    V11 / "prompts" / "expand_summary_v11.txt",
    V11 / "prompts" / "rewrite_summary_v11.txt",
]
PY_FILES = [
    V11 / "__init__.py",
    V11 / "common.py",
    V11 / "run_evidence.py",
    V11 / "run_questions.py",
    V11 / "run_phase_b.py",
    V11 / "validate_outputs.py",
    V11 / "sanity_check.py",
    V11 / "inspect_run.py",
    V11 / "dry_run_selector_on_v7.py",
]
TEXT_EXTENSIONS = {".py", ".ps1", ".md", ".txt"}
EXPECTED_PROMPT_REFERENCES = [
    "generate_summary_v11.txt",
    "expand_summary_v11.txt",
    "rewrite_summary_v11.txt",
    "03_questions_v11",
    "05_module3_answers_v11",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def check_required_files() -> None:
    missing = [path for path in REQUIRED_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing V11 files: {missing}")


def check_python_syntax() -> None:
    for path in PY_FILES:
        ast.parse(read(path), filename=str(path))


def check_v11_references() -> None:
    run_phase_b = read(V11 / "run_phase_b.py")
    for needle in EXPECTED_PROMPT_REFERENCES:
        if needle not in run_phase_b and needle not in read(V11 / "run_questions.py"):
            raise ValueError(f"Missing expected V11 reference: {needle}")
    forbidden = [
        "src.thesis_laysumm_v10",
        "thesis_laysumm_v10.run",
        "generate_summary_v10.txt",
        "expand_summary_v10.txt",
        "rewrite_summary_v10.txt",
        "03_questions_v10",
        "05_module3_answers_v10",
        "run_pilot_n20_v10.ps1",
    ]
    offenders: list[str] = []
    for path in V11.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_EXTENSIONS:
            continue
        text = read(path)
        for needle in forbidden:
            if needle in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {needle}")
    if offenders:
        raise ValueError("Forbidden V10 references found:\n" + "\n".join(offenders))


def check_no_mojibake_like_text() -> None:
    suspicious = (
        r"[\uE000-\uF8FF\uFFFD"
        r"\u0104\u0105\u0106\u0107\u0118\u0119\u0141\u0142"
        r"\u0143\u0144\u015A\u015B\u0179\u017A\u017B\u017C"
        r"\u02C7\u02D8-\u02DD]"
    )
    pattern = re.compile(suspicious)
    offenders: list[str] = []
    for path in V11.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_EXTENSIONS:
            continue
        hits = sorted(set(pattern.findall(read(path))))
        if hits:
            offenders.append(f"{path.relative_to(ROOT)} hits={hits[:10]}")
    if offenders:
        raise ValueError("Mojibake-like characters found:\n" + "\n".join(offenders))


def main() -> None:
    check_required_files()
    check_python_syntax()
    check_v11_references()
    check_no_mojibake_like_text()
    print("[check_v11_static] ok")


if __name__ == "__main__":
    main()
