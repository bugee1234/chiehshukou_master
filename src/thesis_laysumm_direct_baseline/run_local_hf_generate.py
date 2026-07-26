from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT_DIR / "data" / "laysumm_direct_baseline" / "runs"
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "direct_lay_summary.txt"

MODEL_CONFIGS = {
    "qwen25_7b_instruct_off_the_shelf": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "official_baseline_label": "Baseline-qwen2.5-7B-sft",
    },
    "llama3_8b_instruct_off_the_shelf": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "official_baseline_label": "Baseline-llama3-8B-sft",
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text or "").strip()))


def split_template(template: str) -> tuple[str, str]:
    if template.count("{article}") != 1:
        raise ValueError("Prompt template must contain {article} exactly once")
    return tuple(template.split("{article}", 1))  # type: ignore[return-value]


def prepare_chat_input(
    *, tokenizer: Any, article: str, template: str, max_input_tokens: int
) -> tuple[Any, dict[str, Any]]:
    """Preserve the instruction and truncate only the article from its tail.

    Token limits are applied after each model's native chat template. This is
    necessary because the original Llama 3 checkpoint has an 8,192-token
    context window. The exact token counts and truncation are recorded per row.
    """
    import torch

    prefix, suffix = split_template(template)
    article_ids = tokenizer(article, add_special_tokens=False)["input_ids"]
    original_count = len(article_ids)

    low, high = 0, original_count
    best_ids = None
    best_used = 0
    while low <= high:
        used = (low + high) // 2
        visible_article = tokenizer.decode(article_ids[:used], skip_special_tokens=True)
        content = prefix + visible_article + suffix
        candidate = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        length = int(candidate.shape[-1])
        if length <= max_input_tokens:
            best_ids = candidate
            best_used = used
            low = used + 1
        else:
            high = used - 1
    if best_ids is None:
        raise ValueError("Prompt wrapper alone exceeds --max-input-tokens")
    return best_ids.to(dtype=torch.long), {
        "original_article_tokens": original_count,
        "visible_article_tokens": best_used,
        "input_tokens_with_chat_template": int(best_ids.shape[-1]),
        "article_truncated_for_context": best_used < original_count,
        "truncation_policy": "tail_truncation_after_model_tokenization",
    }


def quantization_config(mode: str) -> Any:
    import torch
    from transformers import BitsAndBytesConfig

    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def generate(
    *,
    run_name: str,
    model_key: str,
    quantization: str,
    max_input_tokens: int,
    max_new_tokens: int,
    resume: bool,
) -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}; valid={sorted(MODEL_CONFIGS)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; local baseline generation requires the NVIDIA GPU")

    cfg = MODEL_CONFIGS[model_key]
    model_id = str(cfg["model_id"])
    input_path = DATA_ROOT / run_name / "00_inputs" / "articles.jsonl"
    output_path = DATA_ROOT / run_name / "01_direct_summaries" / model_key / "generated_summaries.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Prepare the direct-baseline run first: {input_path}")
    articles = load_jsonl(input_path)
    order = {str(row["id"]): i for i, row in enumerate(articles)}
    if len(order) != len(articles):
        raise ValueError("Input articles contain duplicate IDs")

    template = PROMPT_PATH.read_text(encoding="utf-8")
    prompt_hash = sha256_text(template)
    existing = load_jsonl(output_path) if resume and output_path.exists() else []
    existing_by_id = {str(row["article_id"]): row for row in existing}
    if len(existing_by_id) != len(existing) or set(existing_by_id) - set(order):
        raise ValueError("Existing local summaries have duplicate or out-of-run IDs")
    for row in existing:
        expected = {
            "model_key": model_key,
            "model": model_id,
            "quantization": quantization,
            "max_input_tokens": max_input_tokens,
            "max_new_tokens": max_new_tokens,
            "prompt_sha256": prompt_hash,
        }
        mismatches = {
            key: {"existing": row.get(key), "requested": value}
            for key, value in expected.items()
            if row.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Cannot resume {model_key} with changed settings: {mismatches}. "
                "Use a new run name or explicitly rerun with --no-resume."
            )

    print(f"[local load] {model_key} -> {model_id} ({quantization})")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config(quantization),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_context = int(getattr(model.config, "max_position_embeddings", max_input_tokens + max_new_tokens))
    if max_input_tokens + max_new_tokens > model_context:
        raise ValueError(
            f"Requested {max_input_tokens}+{max_new_tokens} tokens exceeds {model_id} context {model_context}"
        )

    rows = list(existing)
    for article in tqdm(articles, desc=f"local baseline | {model_key}"):
        article_id = str(article["id"])
        if article_id in existing_by_id:
            continue
        input_ids, token_meta = prepare_chat_input(
            tokenizer=tokenizer,
            article=str(article.get("article") or ""),
            template=template,
            max_input_tokens=max_input_tokens,
        )
        input_ids = input_ids.to(model.device)
        attention_mask = torch.ones_like(input_ids)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        elapsed = time.perf_counter() - started
        new_ids = generated[0, input_ids.shape[-1] :]
        summary = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if not summary:
            raise RuntimeError(f"Empty generation for {article_id}")
        row = {
            "article_id": article_id,
            "source_dataset": article.get("source_dataset"),
            "original_index": article.get("original_index"),
            "model_key": model_key,
            "provider": "huggingface_local",
            "model": model_id,
            "official_baseline_label": cfg["official_baseline_label"],
            "baseline_design": "off_the_shelf_instruct_zero_shot_not_official_sft",
            "task_specific_fine_tuning": False,
            "prompt_version": PROMPT_PATH.stem,
            "prompt_file": str(PROMPT_PATH),
            "prompt_sha256": prompt_hash,
            "visible_input_fields": ["article"],
            "temperature": 0.0,
            "decoding": "greedy",
            "quantization": quantization,
            "max_input_tokens": max_input_tokens,
            "max_new_tokens": max_new_tokens,
            "generated_summary": summary,
            "word_count": word_count(summary),
            "max_summary_words": None,
            "truncated_by_word_limit": False,
            "raw_response": summary,
            "parse_error": False,
            "generated_tokens": int(new_ids.shape[-1]),
            "hit_max_new_tokens": int(new_ids.shape[-1]) >= max_new_tokens,
            "generation_seconds": round(elapsed, 3),
            **token_meta,
        }
        rows.append(row)
        rows.sort(key=lambda item: order[str(item["article_id"])])
        save_jsonl(rows, output_path)

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "model_key": model_key,
            "model_id": model_id,
            "official_baseline_label": cfg["official_baseline_label"],
            "experimental_label": "off-the-shelf instruction checkpoint; no BioLaySumm fine-tuning",
            "article_count": len(rows),
            "by_source": {
                source: sum(row.get("source_dataset") == source for row in rows)
                for source in ("PLOS", "eLife")
            },
            "quantization": quantization,
            "max_input_tokens": max_input_tokens,
            "max_new_tokens": max_new_tokens,
            "truncated_article_count": sum(bool(row.get("article_truncated_for_context")) for row in rows),
            "hit_max_new_tokens_count": sum(bool(row.get("hit_max_new_tokens")) for row in rows),
            "prompt_sha256": prompt_hash,
            "visible_input_fields": ["article"],
            "task_specific_fine_tuning": False,
        },
        output_path.parent / "generation_metadata.json",
    )
    print(f"[local generate] {model_key}: rows={len(rows)} -> {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run off-the-shelf BioLaySumm backbone checkpoints locally.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-key", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--quantization", choices=["8bit", "4bit"], default="8bit")
    parser.add_argument("--max-input-tokens", type=int, default=7680)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate(
        run_name=args.run_name,
        model_key=args.model_key,
        quantization=args.quantization,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
