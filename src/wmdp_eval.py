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

LETTERS = ["A", "B", "C", "D", "E", "F"]


@dataclass
class WMDPConfig:
    dataset_name: str = "cais/wmdp"
    subsets: tuple[str, ...] = ("wmdp-bio", "wmdp-chem", "wmdp-cyber")
    split: str = "test"
    max_samples: int | None = None
    max_length: int = 512


def evaluate_wmdp(model, tokenizer, cfg: WMDPConfig, output_dir: str | Path | None = None) -> dict:
    model.eval()
    results: dict[str, dict] = {}
    all_correct: list[float] = []
    all_gold_probs: list[float] = []

    for subset in cfg.subsets:
        rows = load_dataset(cfg.dataset_name, subset, split=cfg.split)
        if cfg.max_samples:
            rows = rows.select(range(min(cfg.max_samples, len(rows))))
        subset_correct: list[float] = []
        subset_gold_probs: list[float] = []
        for row in rows:
            scores = score_wmdp_row(model, tokenizer, row, cfg.max_length)
            gold = int(row["answer"])
            pred = int(np.argmax(scores["choice_probs"]))
            correct = float(pred == gold)
            subset_correct.append(correct)
            subset_gold_probs.append(float(scores["choice_probs"][gold]))
        results[subset] = {
            "num_examples": len(subset_correct),
            "accuracy_higher_is_better": float(np.mean(subset_correct)) if subset_correct else 0.0,
            "gold_choice_probability_higher_is_better": float(np.mean(subset_gold_probs)) if subset_gold_probs else 0.0,
        }
        all_correct.extend(subset_correct)
        all_gold_probs.extend(subset_gold_probs)

    results["summary"] = {
        "num_examples": len(all_correct),
        "accuracy_higher_is_better": float(np.mean(all_correct)) if all_correct else 0.0,
        "gold_choice_probability_higher_is_better": float(np.mean(all_gold_probs)) if all_gold_probs else 0.0,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(output_dir / "wmdp_eval.json", results)
        write_wmdp_markdown(output_dir / "wmdp_eval.md", results)
    return results


@torch.no_grad()
def score_wmdp_row(model, tokenizer, row: dict, max_length: int) -> dict:
    choices = list(row["choices"])
    prompt = format_wmdp_prompt(row["question"], choices)
    losses = [completion_loss(model, tokenizer, prompt, f" {LETTERS[idx]}", max_length) for idx in range(len(choices))]
    scores = np.array([-loss for loss in losses], dtype=np.float64)
    scores -= scores.max()
    probs = np.exp(scores)
    probs /= probs.sum() + 1e-12
    return {
        "choice_losses": [float(loss) for loss in losses],
        "choice_probs": [float(prob) for prob in probs],
    }


def format_wmdp_prompt(question: str, choices: list[str]) -> str:
    lines = [f"Question: {question}"]
    for idx, choice in enumerate(choices):
        lines.append(f"{LETTERS[idx]}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


@torch.no_grad()
def completion_loss(model, tokenizer, prompt: str, completion: str, max_length: int) -> float:
    device = model_device(model)
    full_text = prompt + completion
    enc = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]
    labels = enc["input_ids"].clone()
    labels[:, : min(prompt_ids.numel(), labels.shape[1])] = -100
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
    loss = out.loss.item()
    return loss if math.isfinite(loss) else 1e9


def write_wmdp_markdown(path: Path, results: dict) -> None:
    lines = ["# WMDP Evaluation", ""]
    lines.append("| subset | examples | accuracy ↑ | gold prob ↑ |")
    lines.append("|---|---:|---:|---:|")
    for key, value in results.items():
        if key == "summary":
            continue
        lines.append(
            f"| {key} | {value['num_examples']} | {value['accuracy_higher_is_better']:.4f} | "
            f"{value['gold_choice_probability_higher_is_better']:.4f} |"
        )
    summary = results["summary"]
    lines.append(
        f"| **summary** | **{summary['num_examples']}** | **{summary['accuracy_higher_is_better']:.4f}** | "
        f"**{summary['gold_choice_probability_higher_is_better']:.4f}** |"
    )
    lines.extend(["", "```json", json.dumps(results, indent=2), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
