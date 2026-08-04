from __future__ import annotations

from itertools import cycle

import torch
import torch.nn.functional as F

from .models import model_device
from .saliency import move_batch, trainable_named_parameters


BASE_OBJECTIVES = {"grad_ascent", "grad_diff", "npo", "dpo", "simnpo", "rmu", "kl", "retrain"}
WRAPPERS = {
    "none",
    "forget_only",
    "dual_mask_naive",
    "cadmu",
    "orthogonal_neutral",
    "cadmu_freeze_forget",
    "cadmu_freeze_retain",
    "cadmu_freeze_conflict",
    "cadmu_neutral_grad_diff",
}


def run_unlearning(model, forget_loader, retain_loader, masks, cfg: dict, refresh_masks=None):
    method = cfg.get("method", "cadmu_grad_diff")
    objective, wrapper = parse_method(method, cfg)

    model.train()
    device = model_device(model)
    named_params = trainable_named_parameters(model)
    opt = torch.optim.AdamW([p for _, p in named_params], lr=cfg["lr"])
    retain_iter = cycle(retain_loader)
    forget_iter = cycle(forget_loader)
    params = dict(named_params)
    dynamic_masking = bool(cfg.get("dynamic_masking", False)) and wrapper != "none"
    mask_update_every = int(cfg.get("mask_update_every", 0) or 0)
    if dynamic_masking and (refresh_masks is None or mask_update_every <= 0):
        raise ValueError("dynamic_masking requires refresh_masks and mask_update_every > 0")

    for step in range(cfg["steps"]):
        if dynamic_masking and step > 0 and step % mask_update_every == 0:
            print(f"refreshing dynamic masks at step {step}/{cfg['steps']}")
            masks = refresh_masks(model, step=step)
            model.train()

        forget_batch = move_batch(next(forget_iter), device)
        retain_batch = move_batch(next(retain_iter), device)

        raw = collect_objective_grads(model, forget_batch, retain_batch, objective, cfg)
        opt.zero_grad(set_to_none=True)
        for name, p in params.items():
            grad = build_wrapped_gradient(
                name=name,
                param=p,
                raw=raw,
                masks=masks,
                cfg=cfg,
                wrapper=wrapper,
                device=device,
            )
            if grad is not None:
                p.grad = grad

        opt.step()

        if (step + 1) % 5 == 0 or step == 0:
            print(
                f"{method} step {step + 1}/{cfg['steps']} "
                f"forget_loss={raw['forget_loss_value']:.4f} "
                f"retain_loss={raw['retain_loss_value']:.4f} "
                f"objective_loss={raw['objective_loss_value']:.4f}"
            )


def parse_method(method: str, cfg: dict) -> tuple[str, str]:
    if method in BASE_OBJECTIVES:
        return method, "none"

    if method.startswith("cadmu_freeze_forget_"):
        objective = method.removeprefix("cadmu_freeze_forget_")
        wrapper = "cadmu_freeze_forget"
    elif method.startswith("cadmu_freeze_retain_"):
        objective = method.removeprefix("cadmu_freeze_retain_")
        wrapper = "cadmu_freeze_retain"
    elif method.startswith("cadmu_freeze_conflict_"):
        objective = method.removeprefix("cadmu_freeze_conflict_")
        wrapper = "cadmu_freeze_conflict"
    elif method.startswith("cadmu_neutral_grad_diff_"):
        objective = method.removeprefix("cadmu_neutral_grad_diff_")
        wrapper = "cadmu_neutral_grad_diff"
    elif method.startswith("cadmu_"):
        objective = method.removeprefix("cadmu_")
        wrapper = "cadmu"
    elif method.startswith("dual_mask_naive_"):
        objective = method.removeprefix("dual_mask_naive_")
        wrapper = "dual_mask_naive"
    elif method.startswith("forget_only_"):
        objective = method.removeprefix("forget_only_")
        wrapper = "forget_only"
    elif method.startswith("orthogonal_neutral_"):
        objective = method.removeprefix("orthogonal_neutral_")
        wrapper = "orthogonal_neutral"
    else:
        legacy = {
            "no_mask": ("grad_diff", "none"),
            "forget_only": ("grad_ascent", "forget_only"),
            "dual_mask_naive": ("grad_diff", "dual_mask_naive"),
            "cadmu_lite": ("grad_diff", "cadmu"),
        }
        if method in legacy:
            return legacy[method]
        raise ValueError(f"Unknown unlearning method: {method}")

    if objective not in BASE_OBJECTIVES:
        raise ValueError(f"Unknown base objective in method {method}: {objective}")
    if wrapper not in WRAPPERS:
        raise ValueError(f"Unknown wrapper in method {method}: {wrapper}")
    return objective, wrapper


def collect_objective_grads(model, forget_batch, retain_batch, objective: str, cfg: dict):
    forget_loss = model(
        input_ids=forget_batch["input_ids"],
        attention_mask=forget_batch["attention_mask"],
        labels=forget_batch["labels"],
    ).loss
    retain_loss = model(
        input_ids=retain_batch["input_ids"],
        attention_mask=retain_batch["attention_mask"],
        labels=retain_batch["labels"],
    ).loss

    alpha = cfg.get("alpha_forget", 1.0)
    beta = cfg.get("beta_retain", 1.0)
    npo_beta = cfg.get("npo_beta", 1.0)
    dpo_beta = cfg.get("dpo_beta", npo_beta)
    simnpo_beta = cfg.get("simnpo_beta", npo_beta)

    if objective == "grad_ascent":
        objective_loss = -alpha * forget_loss
        forget_component = -alpha * forget_loss
        # Keep pure ascent as the no-mask objective, but expose retain gradients to wrappers.
        retain_component = beta * retain_loss
    elif objective == "grad_diff":
        objective_loss = -alpha * forget_loss + beta * retain_loss
        forget_component = -alpha * forget_loss
        retain_component = beta * retain_loss
    elif objective == "npo":
        npo_loss = npo_forget_loss(model, forget_batch, beta=npo_beta)
        objective_loss = alpha * npo_loss + beta * retain_loss
        forget_component = alpha * npo_loss
        retain_component = beta * retain_loss
    elif objective == "simnpo":
        simnpo_loss = simnpo_forget_loss(model, forget_batch, beta=simnpo_beta)
        objective_loss = alpha * simnpo_loss + beta * retain_loss
        forget_component = alpha * simnpo_loss
        retain_component = beta * retain_loss
    elif objective == "dpo":
        dpo_loss = dpo_forget_retain_loss(model, forget_batch, retain_batch, beta=dpo_beta)
        objective_loss = alpha * dpo_loss + beta * retain_loss
        forget_component = alpha * dpo_loss
        retain_component = beta * retain_loss
    elif objective == "rmu":
        rmu_loss = rmu_forget_loss(model, forget_batch, cfg)
        objective_loss = alpha * rmu_loss + beta * retain_loss
        forget_component = alpha * rmu_loss
        retain_component = beta * retain_loss
    elif objective == "kl":
        kl_loss = retain_kl_loss(model, retain_batch)
        objective_loss = -alpha * forget_loss + beta * kl_loss
        forget_component = -alpha * forget_loss
        retain_component = beta * kl_loss
    elif objective == "retrain":
        objective_loss = beta * retain_loss
        forget_component = forget_loss * 0.0
        retain_component = beta * retain_loss
    else:
        raise ValueError(f"Unknown objective: {objective}")

    return {
        "objective": collect_grads(model, objective_loss),
        "forget": collect_grads(model, forget_component),
        "retain": collect_grads(model, retain_component) if retain_component is not None else None,
        "forget_loss_value": float(forget_loss.detach().item()),
        "retain_loss_value": float(retain_loss.detach().item()),
        "objective_loss_value": float(objective_loss.detach().item()),
    }


def collect_grads(model, loss):
    if loss is None:
        return None
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    grads = {}
    for name, p in trainable_named_parameters(model):
        grads[name] = None if p.grad is None else p.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    return grads


def npo_forget_loss(model, batch, beta: float):
    current_nll = batch_sequence_nll(model, batch)
    with torch.no_grad():
        ref_nll = batch_sequence_nll_with_reference(model, batch)
    lose_log_ratio = -(current_nll - ref_nll)
    return -2.0 / beta * F.logsigmoid(-beta * lose_log_ratio).mean()


def simnpo_forget_loss(model, batch, beta: float):
    current_nll = batch_sequence_nll(model, batch)
    return -2.0 / beta * F.logsigmoid(beta * current_nll).mean()


def dpo_forget_retain_loss(model, forget_batch, retain_batch, beta: float):
    chosen_logp = batch_sequence_logp(model, retain_batch)
    rejected_logp = batch_sequence_logp(model, forget_batch)
    with torch.no_grad():
        ref_chosen_logp = batch_sequence_logp_with_reference(model, retain_batch)
        ref_rejected_logp = batch_sequence_logp_with_reference(model, forget_batch)
    policy_log_ratio = chosen_logp.mean() - rejected_logp.mean()
    ref_log_ratio = ref_chosen_logp.mean() - ref_rejected_logp.mean()
    return -F.logsigmoid(beta * (policy_log_ratio - ref_log_ratio))


def rmu_forget_loss(model, batch, cfg: dict):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        output_hidden_states=True,
    )
    hidden = outputs.hidden_states[-1].float()
    labels = batch["labels"]
    token_mask = labels.ne(-100)
    if not token_mask.any():
        token_mask = batch["attention_mask"].bool()
    direction = deterministic_rmu_direction(
        hidden.shape[-1],
        hidden.device,
        hidden.dtype,
        scale=cfg.get("rmu_target_scale", 1.0),
    )
    target = direction.view(1, 1, -1).expand_as(hidden)
    return F.mse_loss(hidden[token_mask], target[token_mask])


def deterministic_rmu_direction(width: int, device, dtype, scale: float):
    values = torch.arange(width, device=device, dtype=torch.float32)
    direction = torch.sin(values * 12.9898 + 78.233)
    direction = direction / (direction.norm() + 1.0e-12)
    return (direction * scale).to(dtype)


def retain_kl_loss(model, batch):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    logits = outputs.logits[..., :-1, :].contiguous()
    with torch.no_grad():
        ref_logits = reference_logits(model, batch)[..., :-1, :].contiguous()
    labels = batch["labels"][..., 1:].contiguous()
    mask = labels.ne(-100)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    ref_probs = F.softmax(ref_logits.float(), dim=-1)
    kl = F.kl_div(log_probs, ref_probs, reduction="none").sum(dim=-1)
    if mask.any():
        return kl[mask].mean()
    return kl.mean()


def batch_sequence_nll(model, batch):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    logits = outputs.logits[..., :-1, :].contiguous()
    labels = batch["labels"][..., 1:].contiguous()
    loss = F.cross_entropy(
        logits.transpose(-1, -2),
        labels,
        ignore_index=-100,
        reduction="none",
    ).sum(dim=-1)
    return loss


def batch_sequence_logp(model, batch):
    return -batch_sequence_nll(model, batch)


def batch_sequence_logp_with_reference(model, batch):
    return -batch_sequence_nll_with_reference(model, batch)


def reference_logits(model, batch):
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            return model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    ).logits.detach()


def batch_sequence_nll_with_reference(model, batch):
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            return batch_sequence_nll(model, batch)
    return batch_sequence_nll(model, batch)


def build_wrapped_gradient(name, param, raw, masks, cfg: dict, wrapper: str, device):
    objective_grad = get_grad(raw["objective"], name, param)
    forget_grad = get_grad(raw["forget"], name, param)
    retain_grad = get_grad(raw["retain"], name, param)

    if wrapper == "none":
        return objective_grad

    ro = masks.retain_only[name].to(device)
    fo = masks.forget_only[name].to(device)
    cf = masks.conflict[name].to(device)
    mf = masks.forget[name].to(device)
    nt = masks.neutral[name].to(device)

    if wrapper == "forget_only":
        return mf * forget_grad

    grad = torch.zeros_like(param)
    if wrapper == "dual_mask_naive":
        grad = grad + ro * retain_grad
        grad = grad + fo * forget_grad
        grad = grad + cf * forget_grad
        return grad

    if wrapper in {
        "cadmu",
        "orthogonal_neutral",
        "cadmu_freeze_forget",
        "cadmu_freeze_retain",
        "cadmu_freeze_conflict",
        "cadmu_neutral_grad_diff",
    }:
        projected = project_away(forget_grad, retain_grad, cfg["projection_eps"])
        if wrapper != "cadmu_freeze_retain":
            grad = grad + ro * retain_grad
        if wrapper != "cadmu_freeze_forget":
            grad = grad + fo * forget_grad
        if wrapper != "cadmu_freeze_conflict":
            grad = grad + cf * projected
        if wrapper in {"orthogonal_neutral", "cadmu_neutral_grad_diff"}:
            grad = grad + nt * objective_grad
        return grad

    raise ValueError(f"Unknown wrapper: {wrapper}")


def get_grad(grads: dict[str, torch.Tensor] | None, name: str, param):
    if grads is None or grads.get(name) is None:
        return torch.zeros_like(param)
    return grads[name]


def project_away(source: torch.Tensor, basis: torch.Tensor, eps: float) -> torch.Tensor:
    denom = torch.sum(basis * basis) + eps
    return source - (torch.sum(source * basis) / denom) * basis
