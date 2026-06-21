from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm import tqdm

from src.experiment_2.run_three_model_exp2 import UsageTracker
from src.thesis_laysumm.paths import run_data_dir
from src.utils import load_jsonl, save_jsonl


class ParallelExecutionError(RuntimeError):
    def __init__(self, message: str, partial_results: list[Any]):
        super().__init__(message)
        self.partial_results = partial_results


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def mutate_sentence(sentence: str) -> str:
    text = str(sentence or "").strip()
    if not text:
        return "This statement is not supported."
    number_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if number_match:
        value = float(number_match.group(1))
        new_value = value + 5 if value < 90 else value - 5
        replacement = str(int(new_value)) if float(new_value).is_integer() else str(round(new_value, 1))
        return text[: number_match.start(1)] + replacement + text[number_match.end(1) :]
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
    for pattern, replacement in rules:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
    return "It is not true that " + text[:1].lower() + text[1:]


def chunk_words(text: str, chunk_words: int, overlap_words: int) -> list[dict[str, Any]]:
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_articles(run_name: str) -> list[dict[str, Any]]:
    path = run_data_dir(run_name) / "00_articles" / "articles.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing articles file: {path}")
    return load_jsonl(path)


def _usage_csv_path(run_name: str) -> Path:
    run_dir = run_data_dir(run_name)
    direct = run_dir / "api_usage_calls.csv"
    if direct.is_file():
        return direct
    nested = direct / "api_usage_calls.csv"
    if nested.is_file():
        return nested
    return direct


def usage_tracker(run_name: str, *, resume: bool) -> UsageTracker:
    path = _usage_csv_path(run_name)
    if not resume or not path.is_file():
        return UsageTracker(rows=[])
    try:
        return UsageTracker.from_csv(path)
    except OSError as exc:
        print(
            f"[usage] WARNING: could not read {path} ({exc}); "
            "starting a fresh in-memory usage log for this run."
        )
        return UsageTracker(rows=[])


def save_usage(run_name: str, usage: UsageTracker) -> None:
    run_dir = run_data_dir(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    direct = run_dir / "api_usage_calls.csv"
    out_dir = direct if direct.is_dir() else run_dir
    try:
        usage.save(out_dir)
    except OSError as exc:
        print(f"[usage] WARNING: could not write usage under {out_dir} ({exc}).")


def run_parallel(
    items: Iterable[Any],
    worker: Callable[[Any], Any],
    *,
    max_workers: int,
    desc: str,
) -> list[Any]:
    item_list = list(items)
    if max_workers <= 1:
        out: list[Any] = []
        for item in tqdm(item_list, desc=desc, total=len(item_list)):
            try:
                out.append(worker(item))
            except Exception as exc:
                raise ParallelExecutionError(
                    f"{desc} failed after {len(out)} completed items: {exc}",
                    out,
                ) from exc
        return out
    out: list[Any] = []
    ex = ThreadPoolExecutor(max_workers=max_workers)
    futures = [ex.submit(worker, item) for item in item_list]
    failed = False
    try:
        for fut in tqdm(as_completed(futures), desc=desc, total=len(futures)):
            try:
                out.append(fut.result())
            except Exception as exc:
                failed = True
                for pending in futures:
                    pending.cancel()
                ex.shutdown(wait=False, cancel_futures=True)
                raise ParallelExecutionError(
                    f"{desc} failed after {len(out)} completed items: {exc}",
                    out,
                ) from exc
    finally:
        if not failed:
            ex.shutdown(wait=True, cancel_futures=False)
    return out
