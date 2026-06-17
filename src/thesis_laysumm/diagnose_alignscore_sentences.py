from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
alignscore_src = ROOT_DIR / "AlignScore" / "src"
if str(alignscore_src) not in sys.path:
    sys.path.insert(0, str(alignscore_src))

from nltk.tokenize import sent_tokenize

from src.thesis_laysumm.llm_utils import load_articles
from src.thesis_laysumm.paths import run_data_dir, run_results_dir
from src.thesis_laysumm_v8.common import find_mojibake, readability_stats, safe_word_count
from src.utils import load_jsonl, save_json, save_jsonl


VARIANTS = {
    "generated": {
        "summary_key": "generated_summary",
        "path": lambda run_name, model_key: run_data_dir(run_name)
        / "04_summaries"
        / model_key
        / "generated_summaries.jsonl",
    },
    "rewritten": {
        "summary_key": "rewritten_summary",
        "path": lambda run_name, model_key: run_data_dir(run_name)
        / "06_rewritten"
        / model_key
        / "rewritten_summaries.jsonl",
    },
}


def _load_scorer(*, ckpt_path: str | None, device: str, batch_size: int, verbose: bool):
    from alignscore import AlignScore
    from huggingface_hub import hf_hub_download

    checkpoint = ckpt_path or hf_hub_download(repo_id="yzha/AlignScore", filename="AlignScore-base.ckpt")
    return AlignScore(
        model="roberta-base",
        batch_size=batch_size,
        device=device,
        ckpt_path=checkpoint,
        evaluation_mode="nli_sp",
        verbose=verbose,
    )


def _summary_rows(run_name: str, model_key: str, variant: str) -> tuple[list[dict[str, Any]], str]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
    path = VARIANTS[variant]["path"](run_name, model_key)
    if not path.exists():
        raise FileNotFoundError(f"Missing {variant} summaries: {path}")
    rows = load_jsonl(path)
    summary_key = VARIANTS[variant]["summary_key"]
    missing = [str(row.get("article_id")) for row in rows if not str(row.get(summary_key, "")).strip()]
    if missing:
        raise ValueError(f"{variant} summaries missing text for article_ids={missing}")
    return rows, summary_key


def _sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in sent_tokenize(str(text or "")) if s.strip()]
    return sentences or [str(text or "").strip()]


def _risk_tags(sentence: str) -> list[str]:
    lower = sentence.lower()
    tags: list[str] = []
    if lower.startswith(("this ", "these ", "it ", "they ", "this change", "this finding", "this means", "this shows")):
        tags.append("deictic_subject")
    if any(x in lower for x in ("means that", "shows that", "confirms", "ensures", "causes", "allows", "makes", "drives", "explains")):
        tags.append("causal_or_inferential")
    if any(x in lower for x in ("like ", "as if", "similar to", "magnet", "sponge", "machine")):
        tags.append("analogy_or_metaphor")
    if any(x in lower for x in ("most common", "half of the world", "health-seeking behavior refers", "helps scientists")):
        tags.append("background_or_definition")
    if any(x in lower for x in ("suggest", "may ", "might ", "could ", "possibly", "potential")):
        tags.append("hedged_or_implication")
    if any(x in lower for x in ("not ", " no ", "without", "only", "but not", "however")):
        tags.append("negation_or_contrast")
    if any(ch.isdigit() for ch in sentence):
        tags.append("number")
    stats = readability_stats(sentence)
    if int(stats["max_sentence_words"]) > 24:
        tags.append("long_sentence")
    if float(stats["long_word_ratio"]) >= 0.18:
        tags.append("hard_word_dense")
    if find_mojibake(sentence):
        tags.append("mojibake")
    return tags


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "article_id",
        "source_dataset",
        "original_index",
        "model_key",
        "variant",
        "sentence_index",
        "alignscore",
        "word_count",
        "risk_tags",
        "sentence",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def diagnose(
    *,
    run_name: str,
    model_key: str,
    variant: str,
    device: str,
    batch_size: int,
    ckpt_path: str | None,
    threshold: float,
    limit_articles: int | None,
    verbose: bool,
) -> dict[str, Any]:
    articles = load_articles(run_name)
    articles_by_id = {str(article["id"]): article for article in articles}
    summary_rows, summary_key = _summary_rows(run_name, model_key, variant)
    if limit_articles is not None:
        summary_rows = summary_rows[:limit_articles]

    scorer = _load_scorer(ckpt_path=ckpt_path, device=device, batch_size=batch_size, verbose=verbose)
    out_rows: list[dict[str, Any]] = []
    article_scores: list[dict[str, Any]] = []

    for row in summary_rows:
        aid = str(row["article_id"])
        article = articles_by_id.get(aid)
        if article is None:
            raise ValueError(f"Summary article_id not found in articles: {aid}")
        summary = str(row[summary_key]).strip()
        sentences = _sentences(summary)
        scores = scorer.score(contexts=[str(article["document"])] * len(sentences), claims=sentences)
        sentence_rows = []
        for idx, (sentence, score) in enumerate(zip(sentences, scores), start=1):
            sent_row = {
                "article_id": aid,
                "source_dataset": row.get("source_dataset"),
                "original_index": int(row.get("original_index") or article.get("original_index") or 0),
                "model_key": model_key,
                "variant": variant,
                "sentence_index": idx,
                "alignscore": round(float(score), 6),
                "word_count": safe_word_count(sentence),
                "risk_tags": "|".join(_risk_tags(sentence)),
                "sentence": sentence,
            }
            sentence_rows.append(sent_row)
            out_rows.append(sent_row)
        article_scores.append(
            {
                "article_id": aid,
                "source_dataset": row.get("source_dataset"),
                "original_index": int(row.get("original_index") or article.get("original_index") or 0),
                "sentence_count": len(sentence_rows),
                "mean_sentence_alignscore": round(statistics.mean(r["alignscore"] for r in sentence_rows), 6),
                "min_sentence_alignscore": round(min(r["alignscore"] for r in sentence_rows), 6),
                "low_sentence_count": sum(1 for r in sentence_rows if float(r["alignscore"]) < threshold),
            }
        )

    out_dir = run_results_dir(run_name) / model_key
    prefix = f"sentence_alignscore_{variant}"
    low_rows = [row for row in out_rows if float(row["alignscore"]) < threshold]
    low_rows.sort(key=lambda r: (float(r["alignscore"]), str(r["article_id"]), int(r["sentence_index"])))
    article_scores.sort(key=lambda r: (int(r["low_sentence_count"]) * -1, float(r["mean_sentence_alignscore"])))

    save_jsonl(out_rows, out_dir / f"{prefix}.jsonl")
    save_jsonl(low_rows, out_dir / f"{prefix}_low.jsonl")
    save_jsonl(article_scores, out_dir / f"{prefix}_articles.jsonl")
    _write_csv(out_rows, out_dir / f"{prefix}.csv")
    _write_csv(low_rows, out_dir / f"{prefix}_low.csv")

    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "model_key": model_key,
        "variant": variant,
        "threshold": threshold,
        "article_count": len(summary_rows),
        "sentence_count": len(out_rows),
        "low_sentence_count": len(low_rows),
        "mean_sentence_alignscore": round(statistics.mean(float(r["alignscore"]) for r in out_rows), 6) if out_rows else None,
        "median_sentence_alignscore": round(statistics.median(float(r["alignscore"]) for r in out_rows), 6) if out_rows else None,
        "min_sentence_alignscore": round(min(float(r["alignscore"]) for r in out_rows), 6) if out_rows else None,
        "lowest_sentences": low_rows[:30],
        "worst_articles": article_scores[:20],
        "outputs": {
            "sentences_jsonl": str(out_dir / f"{prefix}.jsonl"),
            "sentences_csv": str(out_dir / f"{prefix}.csv"),
            "low_jsonl": str(out_dir / f"{prefix}_low.jsonl"),
            "low_csv": str(out_dir / f"{prefix}_low.csv"),
            "article_jsonl": str(out_dir / f"{prefix}_articles.jsonl"),
        },
    }
    save_json(payload, out_dir / f"{prefix}_metadata.json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sentence-level AlignScore diagnostics for a LaySumm run.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ckpt-path", default=None)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--limit-articles", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    diagnose(
        run_name=args.run_name,
        model_key=args.model_key,
        variant=args.variant,
        device=args.device,
        batch_size=args.batch_size,
        ckpt_path=args.ckpt_path,
        threshold=args.threshold,
        limit_articles=args.limit_articles,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
