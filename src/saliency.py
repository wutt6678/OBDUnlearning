from __future__ import annotations

from itertools import islice

import torch

from .models import model_device


def trainable_named_parameters(model):
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def compute_gradient_saliency(
    model,
    loader,
    max_batches: int,
    mode: str = "grad_abs",
) -> dict[str, torch.Tensor]:
    model.train()
    device = model_device(model)
    params = trainable_named_parameters(model)
    accum = {name: torch.zeros_like(p, device="cpu") for name, p in params}
    seen = 0

    for batch in islice(loader, max_batches):
        model.zero_grad(set_to_none=True)
        batch = move_batch(batch, device)
        loss = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        ).loss
        loss.backward()
        for name, p in params:
            if p.grad is None:
                continue
            if mode == "weight_grad":
                score = (p.detach() * p.grad.detach()).abs()
            elif mode == "grad_abs":
                score = p.grad.detach().abs()
            elif mode == "fisher":
                score = p.grad.detach().float().pow(2)
            else:
                raise ValueError(f"Unknown saliency mode: {mode}")
            score = score.float().cpu()
            accum[name].add_(score)
        seen += 1

    if seen == 0:
        raise ValueError("Cannot compute saliency from an empty dataloader.")
    for name in accum:
        accum[name].div_(seen)
    model.zero_grad(set_to_none=True)
    return accum


def saliency_threshold(scores: dict[str, torch.Tensor], quantile: float) -> float:
    flat = torch.cat([v.flatten() for v in scores.values()])
    return torch.quantile(flat, quantile).item()
