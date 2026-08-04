from __future__ import annotations

from collections import defaultdict
from itertools import islice

import torch

from .masks import DualMasks, build_masks_from_groups
from .models import model_device
from .saliency import move_batch, trainable_named_parameters


def compute_behavior_probe_scores(
    model,
    loader,
    max_batches: int,
    module_filter: str = "trainable",
) -> dict[str, torch.Tensor]:
    """Measure output-neuron behavior with mean absolute activation.

    The probe is forward-only and independent from gradient saliency. It hooks
    modules that own trainable descendants, averages each output channel over
    batch/sequence positions, and returns one score vector per module.
    """
    model.eval()
    device = model_device(model)
    module_names = select_probe_modules(model, module_filter)
    sums: dict[str, torch.Tensor] = {}
    counts = defaultdict(int)
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            tensor = unpack_tensor(output)
            if tensor is None or tensor.ndim < 2:
                return
            values = tensor.detach().float().abs()
            if values.ndim == 2:
                per_neuron = values.mean(dim=0)
            else:
                per_neuron = values.reshape(-1, values.shape[-1]).mean(dim=0)
            per_neuron = per_neuron.cpu()
            if name not in sums:
                sums[name] = torch.zeros_like(per_neuron)
            if sums[name].numel() == per_neuron.numel():
                sums[name].add_(per_neuron)
                counts[name] += 1

        return hook

    module_lookup = dict(model.named_modules())
    for name in module_names:
        handles.append(module_lookup[name].register_forward_hook(make_hook(name)))

    seen = 0
    try:
        with torch.no_grad():
            for batch in islice(loader, max_batches):
                batch = move_batch(batch, device)
                model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                seen += 1
    finally:
        for handle in handles:
            handle.remove()

    if seen == 0:
        raise ValueError("Cannot compute probe scores from an empty dataloader.")

    scores = {}
    for name, value in sums.items():
        if counts[name] > 0:
            scores[name] = value / counts[name]
    if not scores:
        raise ValueError("Probe did not collect any module activations.")
    return scores


def select_probe_modules(model, module_filter: str) -> list[str]:
    if module_filter != "trainable":
        raise ValueError(f"Unknown probe module_filter: {module_filter}")

    modules = []
    for name, module in model.named_modules():
        if not name:
            continue
        child_names = {child_name for child_name, _child in module.named_children()}
        has_lora_children = bool({"lora_A", "lora_B"} & child_names)
        has_direct_trainable = any(
            param.requires_grad for param in module.parameters(recurse=False)
        )
        if has_lora_children or has_direct_trainable:
            modules.append(name)
    return prune_nested_modules(modules)


def prune_nested_modules(names: list[str]) -> list[str]:
    selected = []
    for name in sorted(names, key=lambda value: (value.count("."), value)):
        if any(name.startswith(parent + ".") for parent in selected):
            continue
        selected.append(name)
    return selected


def unpack_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        return unpack_tensor(output[0])
    if isinstance(output, dict):
        for value in output.values():
            tensor = unpack_tensor(value)
            if tensor is not None:
                return tensor
    return None


def probe_threshold(scores: dict[str, torch.Tensor], quantile: float) -> float:
    flat = torch.cat([value.flatten() for value in scores.values()])
    return torch.quantile(flat, quantile).item()


def build_probe_masks(
    model,
    retain_scores: dict[str, torch.Tensor],
    forget_scores: dict[str, torch.Tensor],
    retain_threshold: float,
    forget_threshold: float,
    unmapped: str = "neutral",
) -> DualMasks:
    neuron_groups = build_probe_neuron_groups(
        retain_scores,
        forget_scores,
        retain_threshold,
        forget_threshold,
    )
    return map_neuron_groups_to_parameter_masks(model, neuron_groups, unmapped=unmapped)


def build_relative_probe_masks(
    model,
    retain_scores: dict[str, torch.Tensor],
    forget_scores: dict[str, torch.Tensor],
    retain_activity_threshold: float,
    forget_activity_threshold: float,
    retain_margin_threshold: float,
    forget_margin_threshold: float,
    unmapped: str = "neutral",
) -> DualMasks:
    neuron_groups = build_relative_probe_neuron_groups(
        retain_scores,
        forget_scores,
        retain_activity_threshold,
        forget_activity_threshold,
        retain_margin_threshold,
        forget_margin_threshold,
    )
    return map_neuron_groups_to_parameter_masks(model, neuron_groups, unmapped=unmapped)


def build_probe_neuron_groups(
    retain_scores: dict[str, torch.Tensor],
    forget_scores: dict[str, torch.Tensor],
    retain_threshold: float,
    forget_threshold: float,
) -> dict[str, dict[str, torch.Tensor]]:
    groups = {"retain_only": {}, "forget_only": {}, "conflict": {}, "neutral": {}}
    for name in sorted(set(retain_scores) & set(forget_scores)):
        r = retain_scores[name] > retain_threshold
        f = forget_scores[name] > forget_threshold
        groups["retain_only"][name] = r & ~f
        groups["forget_only"][name] = ~r & f
        groups["conflict"][name] = r & f
        groups["neutral"][name] = ~r & ~f
    return groups


def build_relative_probe_neuron_groups(
    retain_scores: dict[str, torch.Tensor],
    forget_scores: dict[str, torch.Tensor],
    retain_activity_threshold: float,
    forget_activity_threshold: float,
    retain_margin_threshold: float,
    forget_margin_threshold: float,
) -> dict[str, dict[str, torch.Tensor]]:
    groups = {"retain_only": {}, "forget_only": {}, "conflict": {}, "neutral": {}}
    for name in sorted(set(retain_scores) & set(forget_scores)):
        retain = retain_scores[name]
        forget = forget_scores[name]
        retain_active = retain > retain_activity_threshold
        forget_active = forget > forget_activity_threshold
        retain_only = retain_active & ((retain - forget) > retain_margin_threshold)
        forget_only = forget_active & ((forget - retain) > forget_margin_threshold)
        conflict = (retain_active & forget_active) & ~retain_only & ~forget_only
        neutral = ~(retain_only | forget_only | conflict)
        groups["retain_only"][name] = retain_only
        groups["forget_only"][name] = forget_only
        groups["conflict"][name] = conflict
        groups["neutral"][name] = neutral
    return groups


def map_neuron_groups_to_parameter_masks(
    model,
    neuron_groups: dict[str, dict[str, torch.Tensor]],
    unmapped: str = "neutral",
) -> DualMasks:
    retain_only = {}
    forget_only = {}
    conflict = {}
    neutral = {}
    module_names = sorted(neuron_groups["neutral"], key=len, reverse=True)

    for param_name, param in trainable_named_parameters(model):
        module_name = matching_probe_module(param_name, module_names)
        masks = None
        if module_name is not None:
            masks = expand_neuron_group_for_param(
                param,
                {group: neuron_groups[group][module_name] for group in neuron_groups},
            )
        if masks is None:
            masks = default_param_groups(param, unmapped)
        retain_only[param_name] = masks["retain_only"]
        forget_only[param_name] = masks["forget_only"]
        conflict[param_name] = masks["conflict"]
        neutral[param_name] = masks["neutral"]

    return build_masks_from_groups(retain_only, forget_only, conflict, neutral)


def matching_probe_module(param_name: str, module_names: list[str]) -> str | None:
    for module_name in module_names:
        if param_name.startswith(module_name + "."):
            return module_name
    return None


def expand_neuron_group_for_param(
    param: torch.nn.Parameter,
    groups: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor] | None:
    shape = tuple(param.shape)
    if not shape:
        return None
    neuron_count = groups["neutral"].numel()
    if shape[0] != neuron_count:
        return None

    expanded = {}
    view_shape = (neuron_count,) + (1,) * (param.ndim - 1)
    for group, mask in groups.items():
        expanded[group] = mask.reshape(view_shape).expand(shape).clone()
    return expanded


def default_param_groups(param: torch.nn.Parameter, group_name: str) -> dict[str, torch.Tensor]:
    if group_name not in {"retain_only", "forget_only", "conflict", "neutral"}:
        raise ValueError(f"Unknown unmapped probe group: {group_name}")
    groups = {}
    for name in ("retain_only", "forget_only", "conflict", "neutral"):
        groups[name] = torch.zeros_like(param, dtype=torch.bool, device="cpu")
    groups[group_name] = torch.ones_like(param, dtype=torch.bool, device="cpu")
    return groups
