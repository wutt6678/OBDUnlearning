from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn.utils import clip_grad_norm_
from transformers import set_seed

from src.config import load_config
from src.data import load_tofu_config_examples, make_loader
from src.models import load_model_and_tokenizer, model_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llama3_1b_tofu_sft.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    model, tokenizer = load_model_and_tokenizer(cfg["model"])
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    examples = []
    for split_cfg in data_cfg["splits"]:
        split_examples = load_tofu_config_examples(
            data_cfg["dataset_name"],
            split_cfg["name"],
            data_cfg.get("split", "train"),
            split_cfg.get("max_samples"),
        )
        examples.extend(split_examples)
        print(f"loaded {len(split_examples)} examples from {split_cfg['name']}")
    print(f"total fine-tune examples: {len(examples)}")

    loader = make_loader(
        examples,
        tokenizer,
        data_cfg["max_length"],
        train_cfg["batch_size"],
        shuffle=True,
    )
    device = model_device(model)
    model.train()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    epochs = train_cfg.get("epochs", 1)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    log_every = train_cfg.get("log_every", 20)
    total_updates = math.ceil(len(loader) / grad_accum) * epochs
    update = 0

    for epoch in range(epochs):
        running = 0.0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            loss = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            ).loss
            (loss / grad_accum).backward()
            running += float(loss.detach().item())

            if step % grad_accum == 0 or step == len(loader):
                clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
                update += 1
                if update == 1 or update % log_every == 0 or update == total_updates:
                    avg = running / min(log_every * grad_accum, step)
                    print(
                        f"epoch {epoch + 1}/{epochs} update {update}/{total_updates} "
                        f"loss={loss.detach().item():.4f} avg_loss={avg:.4f}"
                    )
                    running = 0.0

    output_dir = Path(cfg["output_dir"])
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"saved fine-tuned adapter to {model_dir}")

if __name__ == "__main__":
    main()
