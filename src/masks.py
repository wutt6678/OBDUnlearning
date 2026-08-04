from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DualMasks:
    retain: dict[str, torch.Tensor]
    forget: dict[str, torch.Tensor]
    retain_only: dict[str, torch.Tensor]
    forget_only: dict[str, torch.Tensor]
    conflict: dict[str, torch.Tensor]
    neutral: dict[str, torch.Tensor]


def build_dual_masks(
    retain_scores: dict[str, torch.Tensor],
    forget_scores: dict[str, torch.Tensor],
    retain_threshold: float,
    forget_threshold: float,
) -> DualMasks:
    retain = {}
    forget = {}
    retain_only = {}
    forget_only = {}
    conflict = {}
    neutral = {}
    for name in retain_scores:
        r = retain_scores[name] > retain_threshold
        f = forget_scores[name] > forget_threshold
        retain[name] = r
        forget[name] = f
        retain_only[name] = r & ~f
        forget_only[name] = ~r & f
        conflict[name] = r & f
        neutral[name] = ~r & ~f
    return DualMasks(retain, forget, retain_only, forget_only, conflict, neutral)


def build_masks_from_groups(
    retain_only: dict[str, torch.Tensor],
    forget_only: dict[str, torch.Tensor],
    conflict: dict[str, torch.Tensor],
    neutral: dict[str, torch.Tensor],
) -> DualMasks:
    retain = {}
    forget = {}
    for name in retain_only:
        retain[name] = retain_only[name] | conflict[name]
        forget[name] = forget_only[name] | conflict[name]
    return DualMasks(retain, forget, retain_only, forget_only, conflict, neutral)


def mask_stats(masks: DualMasks) -> dict[str, float]:
    total = 0
    counts = {
        "retain_only": 0,
        "forget_only": 0,
        "conflict": 0,
        "neutral": 0,
        "retain": 0,
        "forget": 0,
    }
    for name in masks.retain:
        n = masks.retain[name].numel()
        total += n
        for key in counts:
            counts[key] += int(getattr(masks, key)[name].sum().item())
    return {key: value / max(total, 1) for key, value in counts.items()} | {"total": total}
