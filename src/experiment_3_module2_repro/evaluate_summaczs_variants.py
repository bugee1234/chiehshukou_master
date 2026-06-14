from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "AlignScore" / "src"))
sys.path.insert(0, str(ROOT_DIR / "summac"))

DEFAULT_RUN_NAME = "pilot_val_n10_judge_gemini31_flash_lite"
DEFAULT_MODEL_KEY = "gemini3_flash_preview_minimal"
VARIANTS = ("full_article", "expert_extract", "expert_as_document")
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "build_expert_conditioned_article.txt"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_dir(run_name: str) -> Path:
    return ROOT_DIR / "data" / "experiment_3_module2_repro" / "runs" / run_name


def output_dir(run_name: str) -> Path:
    return ROOT_DIR / "results" / "experiment_3_module2_repro" / run_name / "summaczs_variants"


def read_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Gemini response must be a JSON object")
    return obj


def split_sentences(text: str) -> list[str]:
    try:
        import nltk

        sentences = nltk.tokenize.sent_tokenize(text)
    except (ImportError, LookupError):
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", str(text or "").strip())
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 10]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_expert_extract(args: argparse.Namespace) -> None:
    from src.experiment_2.run_three_model_exp2 import ProviderClient, UsageTracker

    rows = load_jsonl(run_dir(args.run_name) / "00_inputs" / "articles.jsonl")
    if args.limit is not None:
        rows = rows[: args.limit]
    out = output_dir(args.run_name) / "expert_extract"
    output_path = out / "documents.jsonl"
    existing = {
        str(row["id"]): row for row in load_jsonl(output_path)
    } if args.resume and output_path.exists() else {}

    usage_path = out / "api_usage_calls.csv"
    usage = UsageTracker.from_csv(usage_path) if args.resume else UsageTracker(rows=[])
    client = ProviderClient(args.model_key, usage)
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    results: list[dict[str, Any]] = list(existing.values())

    for index, row in enumerate(rows, start=1):
        item_id = str(row["id"])
        if item_id in existing:
            print(f"[expert-extract {index}/{len(rows)}] skip {item_id}")
            continue
        prompt = prompt_template.replace("{expert_summary}", str(row["expert_summary"]))
        prompt = prompt.replace("{article}", str(row["article"]))
        started = time.perf_counter()
        response = client.chat(
            stage="summaczs_expert_extract",
            item_id=item_id,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        elapsed = time.perf_counter() - started
        obj = read_json_object(response)
        document = str(obj.get("expert_conditioned_article", "") or "").strip()
        if not document:
            raise ValueError(f"Gemini returned an empty expert-conditioned article for {item_id}")
        sentence_count = len(split_sentences(document))
        if sentence_count > 100:
            raise ValueError(f"{item_id}: expert extract has {sentence_count} sentences; expected <= 100")
        results.append(
            {
                "id": item_id,
                "source_dataset": row.get("source_dataset"),
                "model_key": args.model_key,
                "document": document,
                "sentence_count": sentence_count,
                "word_count": len(document.split()),
                "generation_seconds": round(elapsed, 6),
            }
        )
        save_jsonl(results, output_path)
        usage.save(out)
        print(f"[expert-extract {index}/{len(rows)}] {item_id}: {sentence_count} sentences, {elapsed:.2f}s")

    usage.save(out)
    ordered = {str(row["id"]): row for row in results}
    final_rows = [ordered[str(row["id"])] for row in rows]
    save_jsonl(final_rows, output_path)
    save_json(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_name": args.run_name,
            "model_key": args.model_key,
            "prompt": str(PROMPT_PATH.relative_to(ROOT_DIR)),
            "article_count": len(final_rows),
            "total_generation_seconds": round(sum(float(row["generation_seconds"]) for row in final_rows), 6),
            "max_sentence_count": max(int(row["sentence_count"]) for row in final_rows),
        },
        out / "metadata.json",
    )


def build_variant_documents(
    articles: list[dict[str, Any]], extracts: dict[str, dict[str, Any]], variant: str
) -> list[str]:
    if variant == "full_article":
        return [str(row["article"]) for row in articles]
    if variant == "expert_as_document":
        return [str(row["expert_summary"]) for row in articles]
    missing = [str(row["id"]) for row in articles if str(row["id"]) not in extracts]
    if missing:
        raise ValueError(f"Missing expert extracts for: {', '.join(missing)}. Run prepare-expert-extract first.")
    return [str(extracts[str(row["id"])]["document"]) for row in articles]


def score_alignscore(model: Any, documents: list[str], summaries: list[str]) -> tuple[list[float], list[float]]:
    import torch

    scores: list[float] = []
    elapsed: list[float] = []
    for document, summary in zip(documents, summaries):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        score = float(model.score(contexts=[document], claims=[summary])[0])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - started)
        scores.append(score)
    return scores, elapsed


def score_summaczs_blockwise(
    model: Any, document: str, summary: str, block_sentences: int
) -> tuple[float, int, int]:
    import numpy as np

    sentences = split_sentences(document)
    blocks = [sentences[i : i + block_sentences] for i in range(0, len(sentences), block_sentences)]
    if not blocks:
        blocks = [[document]]
    images = [model.imager.build_image(" ".join(block), summary) for block in blocks]
    full_image = np.concatenate(images, axis=1)
    return float(model.image2score(full_image)), len(sentences), len(blocks)


def score_summaczs(
    model: Any, documents: list[str], summaries: list[str], block_sentences: int
) -> tuple[list[float], list[float], list[int], list[int]]:
    import torch

    scores: list[float] = []
    elapsed: list[float] = []
    sentence_counts: list[int] = []
    block_counts: list[int] = []
    for document, summary in zip(documents, summaries):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        score, sentence_count, block_count = score_summaczs_blockwise(
            model, document, summary, block_sentences
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - started)
        scores.append(score)
        sentence_counts.append(sentence_count)
        block_counts.append(block_count)
    return scores, elapsed, sentence_counts, block_counts


def evaluate(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download

    articles = load_jsonl(run_dir(args.run_name) / "00_inputs" / "articles.jsonl")
    if args.limit is not None:
        articles = articles[: args.limit]
    extract_path = output_dir(args.run_name) / "expert_extract" / "documents.jsonl"
    extract_metadata_path = output_dir(args.run_name) / "expert_extract" / "metadata.json"
    extracts = {str(row["id"]): row for row in load_jsonl(extract_path)} if extract_path.exists() else {}
    extract_metadata = (
        json.loads(extract_metadata_path.read_text(encoding="utf-8"))
        if extract_metadata_path.exists()
        else {}
    )
    summaries = [str(row["expert_summary"]) for row in articles]
    variants = list(args.variants)
    out = output_dir(args.run_name) / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    overall_started = time.perf_counter()

    print("[evaluate] loading AlignScore...")
    align_load_started = time.perf_counter()
    from alignscore import AlignScore

    ckpt_path = hf_hub_download(repo_id="yzha/AlignScore", filename="AlignScore-base.ckpt")
    align_model = AlignScore(
        model="roberta-base",
        batch_size=args.align_batch_size,
        device=args.device,
        ckpt_path=ckpt_path,
        evaluation_mode="nli_sp",
        verbose=False,
    )
    align_load_seconds = time.perf_counter() - align_load_started

    align_results: dict[str, tuple[list[float], list[float]]] = {}
    for variant in variants:
        documents = build_variant_documents(articles, extracts, variant)
        print(f"[evaluate] AlignScore: {variant}")
        align_results[variant] = score_alignscore(align_model, documents, summaries)
    del align_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[evaluate] loading SummaCZS...")
    summaczs_load_started = time.perf_counter()
    from summac.model_summac import SummaCZS

    summaczs_model = SummaCZS(
        model_name="vitc",
        granularity="sentence",
        op1="max",
        op2="mean",
        use_ent=True,
        use_con=True,
        imager_load_cache=False,
        max_doc_sents=args.block_sentences,
        device=args.device,
    )
    summaczs_load_seconds = time.perf_counter() - summaczs_load_started

    summaczs_results: dict[str, tuple[list[float], list[float], list[int], list[int]]] = {}
    for variant in variants:
        documents = build_variant_documents(articles, extracts, variant)
        print(f"[evaluate] SummaCZS: {variant}")
        summaczs_results[variant] = score_summaczs(
            summaczs_model, documents, summaries, args.block_sentences
        )

    detail_rows: list[dict[str, Any]] = []
    summaries_out: dict[str, Any] = {}
    for variant in variants:
        align_scores, align_times = align_results[variant]
        zs_scores, zs_times, sentence_counts, block_counts = summaczs_results[variant]
        documents = build_variant_documents(articles, extracts, variant)
        for index, article in enumerate(articles):
            detail_rows.append(
                {
                    "variant": variant,
                    "id": article["id"],
                    "source_dataset": article.get("source_dataset"),
                    "document_sentences": sentence_counts[index],
                    "document_words": len(documents[index].split()),
                    "summary_sentences": len(split_sentences(summaries[index])),
                    "summaczs_blocks": block_counts[index],
                    "alignscore": align_scores[index],
                    "alignscore_seconds": align_times[index],
                    "summaczs": zs_scores[index],
                    "summaczs_seconds": zs_times[index],
                }
            )
        preparation_seconds = (
            float(extract_metadata.get("total_generation_seconds", 0.0))
            if variant == "expert_extract"
            else 0.0
        )
        inference_seconds = float(sum(align_times) + sum(zs_times))
        summaries_out[variant] = {
            "article_count": len(articles),
            "alignscore_mean": float(np.mean(align_scores)),
            "alignscore_inference_seconds": float(sum(align_times)),
            "summaczs_mean": float(np.mean(zs_scores)),
            "summaczs_inference_seconds": float(sum(zs_times)),
            "combined_inference_seconds": inference_seconds,
            "preparation_generation_seconds": preparation_seconds,
            "end_to_end_seconds": preparation_seconds + inference_seconds,
        }

    method = {
        "generated_summary": "expert_summary for every variant",
        "full_article": {
            "document": "original article",
            "alignscore": "native nli_sp full-document sentence/chunk search",
            "summaczs": "split the article into consecutive blocks of at most 100 sentences, compute one NLI image per block, concatenate all images along the document-sentence axis, then apply image2score once",
        },
        "expert_extract": {
            "document": "Gemini 3 Flash Preview expert-conditioned article capped at 100 sentences",
            "generation_condition": "Gemini sees the original article and expert summary; the expert summary selects relevant information, while every output claim must be supported by the article",
            "alignscore": "native nli_sp scoring against the generated expert-conditioned document",
            "summaczs": "native ZS aggregation against the generated expert-conditioned document",
        },
        "expert_as_document": {
            "document": "expert summary",
            "generated_summary": "the same expert summary",
            "purpose": "identity sanity check, not a formal mathematical upper bound",
        },
        "summaczs_configuration": "vitc, sentence granularity, op1=max, op2=mean, entailment minus contradiction",
        "timing": "per-article inference excludes model loading; preparation_generation_seconds applies only to expert_extract; end_to_end_seconds is preparation plus AlignScore and SummaCZS inference",
    }
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": args.run_name,
        "method": method,
        "configuration": {
            "device": args.device,
            "block_sentences": args.block_sentences,
            "align_batch_size": args.align_batch_size,
            "variants": variants,
        },
        "model_load_seconds": {
            "alignscore": align_load_seconds,
            "summaczs": summaczs_load_seconds,
        },
        "variants": summaries_out,
        "evaluation_wall_seconds": time.perf_counter() - overall_started,
    }
    save_json(result, out / "summary.json")
    save_json(method, out / "method_manifest.json")
    save_jsonl(detail_rows, out / "per_article.jsonl")
    write_csv(detail_rows, out / "per_article.csv")
    print(json.dumps(result["variants"], indent=2))
    print(f"[evaluate] results -> {out}")


def evaluate_summacconv(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from summac.model_summac import SummaCConv

    articles = load_jsonl(run_dir(args.run_name) / "00_inputs" / "articles.jsonl")
    if args.limit is not None:
        articles = articles[: args.limit]
    extract_path = output_dir(args.run_name) / "expert_extract" / "documents.jsonl"
    if not extract_path.exists():
        raise FileNotFoundError(
            f"Missing {extract_path}. Run prepare-expert-extract first."
        )
    extracts = {str(row["id"]): row for row in load_jsonl(extract_path)}
    documents = build_variant_documents(articles, extracts, "expert_extract")
    summaries = [str(row["expert_summary"]) for row in articles]

    for article, document in zip(articles, documents):
        sentence_count = len(split_sentences(document))
        if sentence_count > 100:
            raise ValueError(
                f"{article['id']}: expert extract has {sentence_count} sentences; expected <= 100"
            )

    start_file = ROOT_DIR / "summac" / "summac_conv_vitc_sent_perc_e.bin"
    if not start_file.exists():
        raise FileNotFoundError(f"Missing SummaCConv weights: {start_file}")

    print("[evaluate-summacconv] loading SummaCConv...")
    load_started = time.perf_counter()
    model = SummaCConv(
        models=["vitc"],
        bins="percentile",
        granularity="sentence",
        nli_labels="e",
        device=args.device,
        start_file=str(start_file),
        agg="mean",
        imager_load_cache=False,
        max_doc_sents=100,
    )
    model_load_seconds = time.perf_counter() - load_started

    scores: list[float] = []
    elapsed_seconds: list[float] = []
    detail_rows: list[dict[str, Any]] = []
    inference_started = time.perf_counter()
    for index, (article, document, summary) in enumerate(
        zip(articles, documents, summaries), start=1
    ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        score = float(model.score([document], [summary])["scores"][0])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        scores.append(score)
        elapsed_seconds.append(elapsed)
        detail_rows.append(
            {
                "variant": "expert_extract",
                "id": article["id"],
                "source_dataset": article.get("source_dataset"),
                "document_sentences": len(split_sentences(document)),
                "document_words": len(document.split()),
                "summary_sentences": len(split_sentences(summary)),
                "summary_sentences_used_max": 10,
                "summacconv": score,
                "summacconv_seconds": elapsed,
            }
        )
        print(
            f"[evaluate-summacconv {index}/{len(articles)}] "
            f"{article['id']}: score={score:.6f}, {elapsed:.2f}s"
        )

    inference_wall_seconds = time.perf_counter() - inference_started
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": args.run_name,
        "variant": "expert_extract",
        "article_count": len(articles),
        "document": "Gemini 3 Flash Preview expert-conditioned article capped at 100 sentences",
        "generated_summary": "expert_summary",
        "configuration": {
            "metric": "SummaCConv",
            "models": ["vitc"],
            "bins": "percentile",
            "granularity": "sentence",
            "nli_labels": "e",
            "agg": "mean",
            "max_doc_sents": 100,
            "max_summary_chunks": 10,
            "device": args.device,
        },
        "summacconv_mean": float(np.mean(scores)),
        "summacconv_inference_seconds": float(sum(elapsed_seconds)),
        "summacconv_inference_wall_seconds": inference_wall_seconds,
        "model_load_seconds": model_load_seconds,
    }
    out = output_dir(args.run_name) / "summacconv_expert_extract"
    save_json(result, out / "summary.json")
    save_jsonl(detail_rows, out / "per_article.jsonl")
    write_csv(detail_rows, out / "per_article.csv")
    print(json.dumps(result, indent=2))
    print(f"[evaluate-summacconv] results -> {out}")


def evaluate_retrieval_summacconv(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from summac.model_summac import SummaCConv

    articles = load_jsonl(run_dir(args.run_name) / "00_inputs" / "articles.jsonl")
    if args.limit is not None:
        articles = articles[: args.limit]

    start_file = ROOT_DIR / "summac" / "summac_conv_vitc_sent_perc_e.bin"
    if not start_file.exists():
        raise FileNotFoundError(f"Missing SummaCConv weights: {start_file}")

    overall_started = time.perf_counter()
    print(f"[evaluate-retrieval-summacconv] loading {args.embedding_model}...")
    embedding_load_started = time.perf_counter()
    embedding_model = SentenceTransformer(args.embedding_model, device=args.device)
    embedding_load_seconds = time.perf_counter() - embedding_load_started

    print("[evaluate-retrieval-summacconv] loading SummaCConv...")
    summac_load_started = time.perf_counter()
    model = SummaCConv(
        models=["vitc"],
        bins="percentile",
        granularity="sentence",
        nli_labels="e",
        device=args.device,
        start_file=str(start_file),
        agg="mean",
        imager_load_cache=False,
        max_doc_sents=args.max_document_sentences,
    )
    summac_load_seconds = time.perf_counter() - summac_load_started
    nli_imager = model.imagers[0]

    detail_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    selected_document_rows: list[dict[str, Any]] = []
    scores: list[float] = []
    embedding_seconds_all: list[float] = []
    rerank_seconds_all: list[float] = []
    summac_seconds_all: list[float] = []
    article_wall_seconds_all: list[float] = []

    for article_index, article in enumerate(articles, start=1):
        article_started = time.perf_counter()
        item_id = str(article["id"])
        article_sentences = split_sentences(str(article["article"]))
        summary = str(article["expert_summary"])
        summary_sentences = split_sentences(summary)
        if not article_sentences or not summary_sentences:
            raise ValueError(f"{item_id}: article or expert summary has no usable sentences")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        embedding_started = time.perf_counter()
        article_embeddings = embedding_model.encode(
            article_sentences,
            batch_size=args.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        summary_embeddings = embedding_model.encode(
            summary_sentences,
            batch_size=args.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarities = summary_embeddings @ article_embeddings.T
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        embedding_seconds = time.perf_counter() - embedding_started

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        rerank_started = time.perf_counter()
        selected_sentence_scores: dict[int, float] = {}
        per_summary_evidence: list[dict[str, Any]] = []
        embedding_k = min(args.embedding_top_k, len(article_sentences))
        rerank_k = min(args.nli_top_k, embedding_k)
        for summary_index, summary_sentence in enumerate(summary_sentences):
            candidate_indices = np.argsort(-similarities[summary_index])[:embedding_k].tolist()
            candidate_document = " ".join(article_sentences[i] for i in candidate_indices)
            image = nli_imager.build_image(candidate_document, summary_sentence)
            if image.shape[1] != len(candidate_indices):
                raise ValueError(
                    f"{item_id} summary sentence {summary_index}: NLI tokenizer produced "
                    f"{image.shape[1]} candidates from {len(candidate_indices)} retrieved sentences"
                )
            entailment_scores = image[0, :, 0]
            ranked_positions = np.argsort(-entailment_scores)[:rerank_k].tolist()
            selected_indices = [candidate_indices[position] for position in ranked_positions]

            top20 = [
                {
                    "article_sentence_index": article_sentence_index,
                    "embedding_score": float(similarities[summary_index, article_sentence_index]),
                    "nli_entailment": float(entailment_scores[position]),
                    "sentence": article_sentences[article_sentence_index],
                }
                for position, article_sentence_index in enumerate(candidate_indices)
            ]
            top5 = [top20[position] for position in ranked_positions]
            for evidence in top5:
                sentence_index = int(evidence["article_sentence_index"])
                selected_sentence_scores[sentence_index] = max(
                    selected_sentence_scores.get(sentence_index, float("-inf")),
                    float(evidence["nli_entailment"]),
                )
            per_summary_evidence.append(
                {
                    "summary_sentence_index": summary_index,
                    "summary_sentence": summary_sentence,
                    "embedding_top_k": top20,
                    "nli_rerank_top_k": top5,
                }
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        rerank_seconds = time.perf_counter() - rerank_started

        ranked_unique_indices = sorted(
            selected_sentence_scores,
            key=lambda index: (-selected_sentence_scores[index], index),
        )
        capped_indices = ranked_unique_indices[: args.max_document_sentences]
        selected_indices_in_article_order = sorted(capped_indices)
        selected_sentences = [article_sentences[index] for index in selected_indices_in_article_order]
        retrieved_document = " ".join(selected_sentences)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        summac_started = time.perf_counter()
        score = float(model.score([retrieved_document], [summary])["scores"][0])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        summac_seconds = time.perf_counter() - summac_started
        article_wall_seconds = time.perf_counter() - article_started

        scores.append(score)
        embedding_seconds_all.append(embedding_seconds)
        rerank_seconds_all.append(rerank_seconds)
        summac_seconds_all.append(summac_seconds)
        article_wall_seconds_all.append(article_wall_seconds)
        detail_rows.append(
            {
                "id": item_id,
                "source_dataset": article.get("source_dataset"),
                "article_sentences": len(article_sentences),
                "summary_sentences": len(summary_sentences),
                "summary_sentences_used_by_retrieval": len(summary_sentences),
                "summary_sentences_used_by_summacconv_max": 10,
                "unique_sentences_before_cap": len(selected_sentence_scores),
                "selected_document_sentences": len(selected_sentences),
                "selected_document_words": len(retrieved_document.split()),
                "summacconv": score,
                "embedding_seconds": embedding_seconds,
                "nli_rerank_seconds": rerank_seconds,
                "summacconv_seconds": summac_seconds,
                "article_wall_seconds": article_wall_seconds,
            }
        )
        retrieval_rows.append(
            {
                "id": item_id,
                "source_dataset": article.get("source_dataset"),
                "summary_evidence": per_summary_evidence,
            }
        )
        selected_document_rows.append(
            {
                "id": item_id,
                "source_dataset": article.get("source_dataset"),
                "document": retrieved_document,
                "selected_article_sentence_indices": selected_indices_in_article_order,
                "sentence_count": len(selected_sentences),
                "word_count": len(retrieved_document.split()),
            }
        )
        print(
            f"[evaluate-retrieval-summacconv {article_index}/{len(articles)}] "
            f"{item_id}: selected={len(selected_sentences)}, score={score:.6f}, "
            f"total={article_wall_seconds:.2f}s"
        )

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": args.run_name,
        "article_count": len(articles),
        "method": {
            "document_source": "extractive selection from original article sentences",
            "query": "every expert-summary sentence",
            "embedding_retrieval": f"top {args.embedding_top_k} by normalized cosine similarity",
            "nli_reranking": f"top {args.nli_top_k} by vitc entailment probability",
            "deduplication": "union by original article sentence index",
            "document_cap": f"top {args.max_document_sentences} unique sentences by best NLI entailment, restored to original article order",
            "final_metric": "official SummaCConv configuration",
        },
        "configuration": {
            "embedding_model": args.embedding_model,
            "embedding_top_k": args.embedding_top_k,
            "nli_model": "vitc",
            "nli_top_k": args.nli_top_k,
            "max_document_sentences": args.max_document_sentences,
            "summacconv_max_summary_chunks": 10,
            "device": args.device,
        },
        "summacconv_mean": float(np.mean(scores)),
        "timing_seconds": {
            "embedding_model_load": embedding_load_seconds,
            "summacconv_model_load": summac_load_seconds,
            "embedding_total": float(sum(embedding_seconds_all)),
            "nli_rerank_total": float(sum(rerank_seconds_all)),
            "summacconv_total": float(sum(summac_seconds_all)),
            "ten_article_processing_total": float(sum(article_wall_seconds_all)),
            "overall_wall_including_model_load": time.perf_counter() - overall_started,
        },
    }
    out = output_dir(args.run_name) / "retrieval_summacconv"
    save_json(result, out / "summary.json")
    save_jsonl(detail_rows, out / "per_article.jsonl")
    write_csv(detail_rows, out / "per_article.csv")
    save_jsonl(retrieval_rows, out / "retrieval_evidence.jsonl")
    save_jsonl(selected_document_rows, out / "selected_documents.jsonl")
    print(json.dumps(result, indent=2))
    print(f"[evaluate-retrieval-summacconv] results -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and evaluate three expert-summary factuality diagnostics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-expert-extract")
    prepare.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    prepare.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    prepare.set_defaults(func=prepare_expert_extract)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    eval_parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    eval_parser.add_argument("--device", default="cuda")
    eval_parser.add_argument("--block-sentences", type=int, default=100)
    eval_parser.add_argument("--align-batch-size", type=int, default=16)
    eval_parser.add_argument("--limit", type=int)
    eval_parser.set_defaults(func=evaluate)

    conv_parser = subparsers.add_parser(
        "evaluate-summacconv",
        help="Evaluate SummaCConv on the Gemini expert-extract documents.",
    )
    conv_parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    conv_parser.add_argument("--device", default="cuda")
    conv_parser.add_argument("--limit", type=int)
    conv_parser.set_defaults(func=evaluate_summacconv)

    retrieval_parser = subparsers.add_parser(
        "evaluate-retrieval-summacconv",
        help="Retrieve original article evidence and evaluate it with SummaCConv.",
    )
    retrieval_parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    retrieval_parser.add_argument("--device", default="cuda")
    retrieval_parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    retrieval_parser.add_argument("--embedding-batch-size", type=int, default=64)
    retrieval_parser.add_argument("--embedding-top-k", type=int, default=20)
    retrieval_parser.add_argument("--nli-top-k", type=int, default=5)
    retrieval_parser.add_argument("--max-document-sentences", type=int, default=100)
    retrieval_parser.add_argument("--limit", type=int)
    retrieval_parser.set_defaults(func=evaluate_retrieval_summacconv)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
