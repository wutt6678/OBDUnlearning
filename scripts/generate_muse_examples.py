from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.models import load_model_and_tokenizer, model_device


def row_text(row: dict, text_key: str = "text") -> str:
    if text_key in row and row[text_key] is not None:
        return str(row[text_key])
    if "prompt" in row and "gt" in row:
        return f"{row['prompt']} {row['gt']}"
    values = [str(value) for value in row.values() if isinstance(value, str)]
    return "\n".join(values)


def format_prompt(tokenizer, prompt: str, prompt_format: str) -> str:
    if prompt_format == "chat_template" and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


@torch.no_grad()
def embedding_vocab_size(model) -> int:
    return model.get_input_embeddings().weight.shape[0]


def check_token_ids(model, input_ids: torch.Tensor, context: str) -> None:
    max_id = int(input_ids.max().detach().cpu()) if input_ids.numel() else -1
    vocab_size = embedding_vocab_size(model)
    if max_id >= vocab_size:
        raise ValueError(
            f"Token id out of embedding range in {context}: max_id={max_id}, "
            f"embedding_vocab_size={vocab_size}. Check tokenizer/model mismatch."
        )


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_length: int, max_new_tokens: int, prompt_format: str) -> str:
    device = model_device(model)
    prompt_text = format_prompt(tokenizer, prompt, prompt_format)
    inputs = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    check_token_ids(model, inputs["input_ids"], "generation prompt")
    inputs = inputs.to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def build_raw_examples(
    dataset_name: str,
    config_name: str,
    split: str,
    tokenizer,
    n: int,
    text_key: str,
    prompt_fraction: float,
    text_chunk_chars: int = 2048,
):
    rows = load_dataset(dataset_name, config_name, split=split)
    examples = []
    for idx, row in enumerate(rows):
        text = row_text(row, text_key).strip()
        if not text:
            continue
        for chunk_start in range(0, len(text), text_chunk_chars):
            text_piece = text[chunk_start : chunk_start + text_chunk_chars].strip()
            if not text_piece:
                continue
            token_ids = tokenizer(text_piece, add_special_tokens=False, truncation=False, return_tensors=None)["input_ids"]
            if len(token_ids) < 32:
                continue
            pivot = max(1, min(len(token_ids) - 8, int(len(token_ids) * prompt_fraction)))
            prompt = tokenizer.decode(token_ids[:pivot], skip_special_tokens=True).strip()
            gold_continuation = tokenizer.decode(token_ids[pivot : min(len(token_ids), pivot + 128)], skip_special_tokens=True).strip()
            if not prompt or not gold_continuation:
                continue
            examples.append(
            {
                    "idx": idx,
                    "chunk_start": chunk_start,
                    "source": f"{dataset_name}/{config_name}/{split}",
                    "prompt": prompt,
                    "gold_continuation_prefix": gold_continuation,
                }
            )
            if len(examples) >= n:
                break
        if len(examples) >= n:
            break
    return examples


def build_qa_examples(dataset_name: str, config_name: str, split: str, n: int):
    rows = load_dataset(dataset_name, config_name, split=split)
    examples = []
    for idx, row in enumerate(rows.select(range(min(n, len(rows))))):
        examples.append(
            {
                "idx": idx,
                "source": f"{dataset_name}/{config_name}/{split}",
                "question": str(row["question"]),
                "gold_answer": str(row["answer"]),
            }
        )
    return examples


def write_markdown(path: Path, result: dict) -> None:
    lines = ["# MUSE Generation Examples", ""]
    lines.append(f"model: `{result['model']['name_or_path']}`")
    lines.append("")
    lines.append("## Raw Generation Examples")
    for item in result["raw_generation_examples"]:
        lines.extend(
            [
                "",
                f"### Raw {item['idx']}",
                "",
                "**Prompt**",
                "",
                item["prompt"],
                "",
                "**Gold continuation prefix**",
                "",
                item["gold_continuation_prefix"],
                "",
                "**Model generation**",
                "",
                item["generation"],
            ]
        )
    lines.append("")
    lines.append("## KnowMem QA Generation Examples")
    for item in result["knowmem_qa_generation_examples"]:
        lines.extend(
            [
                "",
                f"### QA {item['idx']}",
                "",
                f"**Question:** {item['question']}",
                "",
                f"**Gold answer:** {item['gold_answer']}",
                "",
                "**Model generation**",
                "",
                item["generation"],
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune/configs/qwen3_5_2b_muse_books_eval.yaml")
    parser.add_argument("--dataset-name", default="muse-bench/MUSE-Books")
    parser.add_argument("--raw-config", default="raw")
    parser.add_argument("--raw-split", default="forget")
    parser.add_argument("--qa-config", default="knowmem")
    parser.add_argument("--qa-split", default="forget_qa")
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--raw-max-new-tokens", type=int, default=128)
    parser.add_argument("--qa-max-new-tokens", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--text-chunk-chars", type=int, default=2048)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    model, tokenizer = load_model_and_tokenizer(cfg["model"])
    model.eval()
    print(
        f"tokenizer_len={len(tokenizer)} embedding_vocab_size={embedding_vocab_size(model)} "
        f"pad={tokenizer.pad_token_id} eos={tokenizer.eos_token_id}"
    )

    data_cfg = cfg.get("data", {})
    max_length = args.max_length or data_cfg.get("max_length", 2048)
    prompt_format = data_cfg.get("prompt_format", "chat_template")
    text_key = data_cfg.get("text_key", "text")
    prompt_fraction = data_cfg.get("text_prompt_fraction", 0.35)

    raw_examples = build_raw_examples(
        args.dataset_name,
        args.raw_config,
        args.raw_split,
        tokenizer,
        args.num_examples,
        text_key,
        prompt_fraction,
        args.text_chunk_chars,
    )
    qa_examples = build_qa_examples(args.dataset_name, args.qa_config, args.qa_split, args.num_examples)

    for item in raw_examples:
        item["generation"] = generate(
            model,
            tokenizer,
            item["prompt"],
            max_length,
            args.raw_max_new_tokens,
            prompt_format,
        )
    for item in qa_examples:
        item["generation"] = generate(
            model,
            tokenizer,
            item["question"],
            max_length,
            args.qa_max_new_tokens,
            prompt_format,
        )

    output_dir = Path(args.output_dir or Path(cfg["output_dir"]) / "generation_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "config": args.config,
        "model": cfg["model"],
        "dataset_name": args.dataset_name,
        "raw_generation_examples": raw_examples,
        "knowmem_qa_generation_examples": qa_examples,
    }
    (output_dir / "muse_generation_examples.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(output_dir / "muse_generation_examples.md", result)
    print(f"saved generation examples to {output_dir}")


if __name__ == "__main__":
    main()
