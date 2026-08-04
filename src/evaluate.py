from __future__ import annotations

import json
from pathlib import Path

import torch

from .data import PROMPT_TEMPLATE
from .models import model_device
from .saliency import move_batch


@torch.no_grad()
def average_loss(model, loader) -> float:
    model.eval()
    device = model_device(model)
    losses = []
    for batch in loader:
        batch = move_batch(batch, device)
        loss = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        ).loss
        losses.append(loss.item())
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def generation_probe(model, tokenizer, examples, max_new_tokens: int, limit: int):
    model.eval()
    device = model_device(model)
    rows = []
    for ex in examples[:limit]:
        prompt = PROMPT_TEMPLATE.format(question=ex.question)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated = text[len(prompt) :].strip()
        answer_hit = _rough_hit(generated, ex.answer)
        rows.append(
            {
                "profile_id": ex.profile_id,
                "question": ex.question,
                "gold_answer": ex.answer,
                "generated": generated,
                "rough_answer_hit": answer_hit,
            }
        )
    return rows


def _rough_hit(generated: str, answer: str) -> bool:
    answer_tokens = [tok.strip(".,;:!?()[]{}\"'").lower() for tok in answer.split()]
    answer_tokens = [tok for tok in answer_tokens if len(tok) >= 5]
    if not answer_tokens:
        return False
    generated_l = generated.lower()
    hits = sum(tok in generated_l for tok in answer_tokens[:12])
    return hits >= max(2, min(4, len(answer_tokens) // 3))


def save_json(path: str | Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
