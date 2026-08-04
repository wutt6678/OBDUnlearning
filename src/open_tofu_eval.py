from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .data import PROMPT_TEMPLATE, load_tofu_raw_config
from .evaluate import save_json
from .models import model_device


@dataclass
class EvalConfig:
    dataset_name: str
    forget_perturbed_split: str
    retain_perturbed_split: str
    holdout_split: str
    real_authors_split: str
    world_facts_split: str
    max_samples: int | None
    batch_size: int
    max_length: int
    max_new_tokens: int


def evaluate_open_tofu(model, tokenizer, cfg: EvalConfig, output_dir: str | Path | None = None) -> dict:
    model.eval()
    results = {}

    forget = load_tofu_raw_config(cfg.dataset_name, cfg.forget_perturbed_split, max_samples=cfg.max_samples)
    retain = load_tofu_raw_config(cfg.dataset_name, cfg.retain_perturbed_split, max_samples=cfg.max_samples)
    holdout = load_tofu_raw_config(cfg.dataset_name, cfg.holdout_split, max_samples=cfg.max_samples)
    real_authors = load_tofu_raw_config(cfg.dataset_name, cfg.real_authors_split, max_samples=cfg.max_samples)
    world_facts = load_tofu_raw_config(cfg.dataset_name, cfg.world_facts_split, max_samples=cfg.max_samples)

    forget_correct = qa_probability_scores(model, tokenizer, forget, cfg)
    forget_para = qa_probability_scores(model, tokenizer, forget, cfg, answer_key="paraphrased_answer")
    forget_wrong = qa_multi_answer_scores(model, tokenizer, forget, cfg, answer_key="perturbed_answer")
    retain_correct = qa_probability_scores(model, tokenizer, retain, cfg)
    retain_wrong = qa_multi_answer_scores(model, tokenizer, retain, cfg, answer_key="perturbed_answer")
    holdout_correct = qa_probability_scores(model, tokenizer, holdout, cfg)

    results["forget"] = {
        "answer_probability": mean_key(forget_correct, "prob"),
        "answer_loss": mean_key(forget_correct, "avg_loss"),
        "paraphrased_probability": mean_key(forget_para, "prob"),
        "rouge_l": qa_rouge_l(model, tokenizer, forget, cfg),
        "truth_ratio": truth_ratio(forget_correct, forget_wrong, mode="forget"),
        "exact_memorization": mean_key(forget_correct, "exact_memorization"),
        "extraction_strength": mean_key(forget_correct, "extraction_strength"),
    }
    results["retain"] = {
        "answer_probability": mean_key(retain_correct, "prob"),
        "answer_loss": mean_key(retain_correct, "avg_loss"),
        "rouge_l": qa_rouge_l(model, tokenizer, retain, cfg),
        "truth_ratio": truth_ratio(retain_correct, retain_wrong, mode="retain"),
    }
    results["privacy"] = {
        "loss_mia_auc_forget_vs_holdout": loss_mia_auc(forget_correct, holdout_correct),
    }
    results["utility"] = {
        "real_authors_mc_prob": mc_normalized_probability(model, tokenizer, real_authors, cfg),
        "world_facts_mc_prob": mc_normalized_probability(model, tokenizer, world_facts, cfg),
    }
    results["summary"] = summarize_open_tofu(results)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(output_dir / "open_tofu_eval.json", results)
        write_eval_markdown(output_dir / "open_tofu_eval.md", results)
    return results


def qa_probability_scores(model, tokenizer, rows, cfg: EvalConfig, answer_key: str = "answer") -> list[dict]:
    scores = []
    for row in rows:
        scores.append(score_answer(model, tokenizer, row["question"], row[answer_key], cfg.max_length))
    return scores


def qa_multi_answer_scores(model, tokenizer, rows, cfg: EvalConfig, answer_key: str) -> list[dict]:
    scores = []
    for row in rows:
        answers = row[answer_key]
        answer_scores = [score_answer(model, tokenizer, row["question"], ans, cfg.max_length) for ans in answers]
        scores.append(
            {
                "prob": float(np.mean([s["prob"] for s in answer_scores])),
                "avg_loss": float(np.mean([s["avg_loss"] for s in answer_scores])),
            }
        )
    return scores


@torch.no_grad()
def score_answer(model, tokenizer, question: str, answer: str, max_length: int) -> dict:
    device = model_device(model)
    prompt = PROMPT_TEMPLATE.format(question=question)
    full_text = f"{prompt} {answer}"
    enc = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]
    labels = enc["input_ids"].clone()
    labels[:, : min(prompt_ids.numel(), labels.shape[1])] = -100
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
    logits = out.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid = shifted_labels != -100
    if valid.sum().item() == 0:
        return {"prob": 0.0, "avg_loss": float("inf"), "exact_memorization": 0.0, "extraction_strength": 0.0}
    token_losses = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2),
        shifted_labels,
        ignore_index=-100,
        reduction="none",
    )
    avg_loss = (token_losses * valid).sum() / valid.sum()
    pred_ids = logits.argmax(dim=-1)
    valid_preds = pred_ids[valid]
    valid_labels = shifted_labels[valid]
    token_match = (valid_preds == valid_labels).float()
    exact_mem = token_match.mean().item()
    extraction = suffix_match_strength(valid_preds, valid_labels)
    return {
        "prob": math.exp(-avg_loss.item()),
        "avg_loss": avg_loss.item(),
        "exact_memorization": exact_mem,
        "extraction_strength": extraction,
    }


def suffix_match_strength(preds: torch.Tensor, labels: torch.Tensor) -> float:
    n = labels.numel()
    if n == 0:
        return 0.0
    for k in range(n):
        if torch.equal(preds[k:], labels[k:]):
            return 1.0 - (k / n)
    return 0.0


@torch.no_grad()
def qa_rouge_l(model, tokenizer, rows, cfg: EvalConfig) -> float:
    values = []
    device = model_device(model)
    for row in rows:
        prompt = PROMPT_TEMPLATE.format(question=row["question"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_length).to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        generation = text[len(prompt) :].strip()
        values.append(rouge_l_f1(generation, row["answer"]))
    return float(np.mean(values)) if values else 0.0


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred = prediction.lower().split()
    ref = reference.lower().split()
    if not pred or not ref:
        return 0.0
    lcs = lcs_len(pred, ref)
    prec = lcs / len(pred)
    rec = lcs / len(ref)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def lcs_len(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def truth_ratio(correct_scores: list[dict], wrong_scores: list[dict], mode: str) -> float:
    ratios = []
    for correct, wrong in zip(correct_scores, wrong_scores):
        correct_prob = math.exp(-correct["avg_loss"])
        wrong_prob = math.exp(-wrong["avg_loss"])
        ratio = wrong_prob / (correct_prob + 1e-10)
        if mode == "forget":
            ratios.append(min(ratio, 1.0 / (ratio + 1e-10)))
        else:
            ratios.append(max(0.0, 1.0 - ratio))
    return float(np.mean(ratios)) if ratios else 0.0


def loss_mia_auc(member_scores: list[dict], nonmember_scores: list[dict]) -> float:
    y_true = [1] * len(member_scores) + [0] * len(nonmember_scores)
    # Lower loss means more likely to be a member.
    y_score = [-x["avg_loss"] for x in member_scores] + [-x["avg_loss"] for x in nonmember_scores]
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.5


def mc_normalized_probability(model, tokenizer, rows, cfg: EvalConfig) -> float:
    values = []
    for row in rows:
        options = [row[f"option{i}"] for i in range(1, 5) if f"option{i}" in row]
        if not options and "perturbed_answer" in row:
            options = [row["answer"], *row["perturbed_answer"]]
        option_scores = [score_answer(model, tokenizer, row["question"], opt, cfg.max_length)["prob"] for opt in options]
        answer_idx = options.index(row["answer"]) if row["answer"] in options else 0
        denom = sum(option_scores) + 1e-10
        values.append(option_scores[answer_idx] / denom)
    return float(np.mean(values)) if values else 0.0


def summarize_open_tofu(results: dict) -> dict:
    memorization = 1.0 - np.mean(
        [
            results["forget"]["answer_probability"],
            results["forget"]["paraphrased_probability"],
            results["forget"]["rouge_l"],
            results["forget"]["exact_memorization"],
            results["forget"]["extraction_strength"],
        ]
    )
    utility = harmonic_mean(
        [
            results["retain"]["answer_probability"],
            max(results["retain"]["rouge_l"], 1e-8),
            results["utility"]["real_authors_mc_prob"],
            results["utility"]["world_facts_mc_prob"],
        ]
    )
    privacy = 1.0 - abs(results["privacy"]["loss_mia_auc_forget_vs_holdout"] - 0.5) * 2
    aggregate = harmonic_mean([max(memorization, 1e-8), max(utility, 1e-8), max(privacy, 1e-8)])
    return {
        "memorization_score_higher_is_better": float(memorization),
        "utility_score_higher_is_better": float(utility),
        "privacy_score_higher_is_better": float(privacy),
        "aggregate_score_higher_is_better": float(aggregate),
    }


def harmonic_mean(values: Iterable[float]) -> float:
    vals = [max(float(v), 1e-8) for v in values]
    return len(vals) / sum(1.0 / v for v in vals)


def mean_key(rows: list[dict], key: str) -> float:
    values = [row[key] for row in rows if np.isfinite(row[key])]
    return float(np.mean(values)) if values else 0.0


def write_eval_markdown(path: Path, results: dict):
    lines = ["# Open-Style TOFU Evaluation", "", "```json", json.dumps(results, indent=2), "```"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
