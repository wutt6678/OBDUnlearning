from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from .evaluate import save_json
from .models import model_device
from .open_tofu_eval import rouge_l_f1


@dataclass
class MUSESetConfig:
    name: str
    dataset_name: str
    config_name: str | None = None
    split: str = "test"
    prompt_key: str = "question"
    answer_key: str = "answer"
    text_key: str | None = None
    max_samples: int | None = None


@dataclass
class MUSEConfig:
    sets: list[MUSESetConfig]
    max_length: int = 512
    max_new_tokens: int = 64
    text_prompt_fraction: float = 0.35
    prompt_format: str = "plain_qa"


def evaluate_muse(model, tokenizer, cfg: MUSEConfig, output_dir: str | Path | None = None) -> dict:
    model.eval()
    results: dict[str, dict] = {}
    for set_cfg in cfg.sets:
        rows = load_muse_rows(set_cfg)
        losses: list[float] = []
        rouge_values: list[float] = []
        exact_values: list[float] = []
        for row in rows:
            prompt, answer = extract_prompt_answer(row, set_cfg, cfg.text_prompt_fraction)
            if not prompt or not answer:
                continue
            losses.append(score_completion_loss(model, tokenizer, prompt, answer, cfg.max_length, cfg.prompt_format))
            generation = generate_completion(
                model,
                tokenizer,
                prompt,
                cfg.max_length,
                cfg.max_new_tokens,
                cfg.prompt_format,
            )
            rouge_values.append(rouge_l_f1(generation, answer))
            exact_values.append(float(answer.strip().lower() in generation.strip().lower()))
        results[set_cfg.name] = {
            "num_examples": len(losses),
            "answer_loss_lower_is_better": float(np.mean(losses)) if losses else 0.0,
            "answer_probability_higher_is_better": float(np.mean([math.exp(-x) for x in losses])) if losses else 0.0,
            "rouge_l_higher_is_better": float(np.mean(rouge_values)) if rouge_values else 0.0,
            "exact_containment_higher_is_better": float(np.mean(exact_values)) if exact_values else 0.0,
        }
    results["summary"] = summarize_muse(results)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(output_dir / "muse_eval.json", results)
        write_muse_markdown(output_dir / "muse_eval.md", results)
    return results


def load_muse_rows(cfg: MUSESetConfig):
    args = [cfg.dataset_name]
    if cfg.config_name:
        args.append(cfg.config_name)
    rows = load_dataset(*args, split=cfg.split)
    if cfg.max_samples:
        rows = rows.select(range(min(cfg.max_samples, len(rows))))
    return rows


def extract_prompt_answer(row: dict, cfg: MUSESetConfig, text_prompt_fraction: float) -> tuple[str, str]:
    if cfg.text_key:
        text = str(row[cfg.text_key])
        if not text.strip():
            return "", ""
        pivot = max(1, min(len(text) - 1, int(len(text) * text_prompt_fraction)))
        return text[:pivot], text[pivot:]
    return str(row[cfg.prompt_key]), str(row[cfg.answer_key])


def format_prompt_and_full(tokenizer, prompt: str, answer: str | None, prompt_format: str) -> tuple[str, str | None]:
    if prompt_format == "chat_template" and tokenizer.chat_template is not None:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if answer is None:
            return prompt_text, None
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        return prompt_text, full_text

    prompt_text = prompt
    if answer is None:
        return prompt_text, None
    separator = "" if prompt.endswith((" ", "\n", "\t")) else " "
    return prompt_text, f"{prompt}{separator}{answer}"


@torch.no_grad()
def score_completion_loss(model, tokenizer, prompt: str, answer: str, max_length: int, prompt_format: str) -> float:
    device = model_device(model)
    prompt_text, full_text = format_prompt_and_full(tokenizer, prompt, answer, prompt_format)
    enc = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )["input_ids"][0]
    labels = enc["input_ids"].clone()
    labels[:, : min(prompt_ids.numel(), labels.shape[1])] = -100
    if torch.all(labels == -100):
        return 1e9
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
    loss = out.loss.item()
    return loss if math.isfinite(loss) else 1e9


@torch.no_grad()
def generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_length: int,
    max_new_tokens: int,
    prompt_format: str,
) -> str:
    device = model_device(model)
    prompt_text, _ = format_prompt_and_full(tokenizer, prompt, None, prompt_format)
    inputs = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    generated_ids = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def summarize_muse(results: dict) -> dict:
    eval_sets = [value for key, value in results.items() if key != "summary"]
    if not eval_sets:
        return {}
    return {
        "mean_answer_loss_lower_is_better": float(np.mean([x["answer_loss_lower_is_better"] for x in eval_sets])),
        "mean_answer_probability_higher_is_better": float(np.mean([x["answer_probability_higher_is_better"] for x in eval_sets])),
        "mean_rouge_l_higher_is_better": float(np.mean([x["rouge_l_higher_is_better"] for x in eval_sets])),
        "mean_exact_containment_higher_is_better": float(np.mean([x["exact_containment_higher_is_better"] for x in eval_sets])),
    }


def write_muse_markdown(path: Path, results: dict) -> None:
    lines = ["# MUSE-Compatible Evaluation", ""]
    lines.append("| set | examples | loss ↓ | prob ↑ | ROUGE-L ↑ | exact containment ↑ |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key, value in results.items():
        if key == "summary":
            continue
        lines.append(
            f"| {key} | {value['num_examples']} | {value['answer_loss_lower_is_better']:.4f} | "
            f"{value['answer_probability_higher_is_better']:.4f} | {value['rouge_l_higher_is_better']:.4f} | "
            f"{value['exact_containment_higher_is_better']:.4f} |"
        )
    lines.extend(["", "## Summary", "", "```json", json.dumps(results.get("summary", {}), indent=2), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
