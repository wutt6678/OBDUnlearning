# OBDUnlearning

A research codebase for **machine unlearning in large language models**. It provides a unified pipeline for fine-tuning surrogate models on knowledge benchmarks (TOFU, MUSE, WMDP), applying a variety of unlearning objectives and masking-based wrappers, and evaluating forget/retain trade-offs.

## Overview

The experimental flow is:

1. **Fine-tune** a base model on a knowledge benchmark (produces a model that "knows" the forget set).
2. **Unlearn** using an objective (`grad_diff`, `npo`, `grad_ascent`, `dpo`, `simnpo`, `rmu`, `kl`, `retrain`) combined with a wrapper (`none`, `forget_only`, `dual_mask_naive`, `cadmu` and variants, `orthogonal_neutral`, ...).
3. **Evaluate** on the benchmark's forget/retain splits plus utility metrics.

## Repository Layout

```
├── src/                  # Core library
│   ├── config.py         # YAML config loading
│   ├── models.py         # Model/tokenizer loading, LoRA inference, compat patches
│   ├── data.py           # Dataset loading & tokenization
│   ├── masks.py          # Parameter masks (saliency/probe-based)
│   ├── probe.py          # Probe construction utilities
│   ├── saliency.py       # Saliency scoring helpers
│   ├── unlearn.py        # Unlearning objectives + wrappers training loop
│   ├── evaluate.py       # Generic evaluation entry
│   ├── open_tofu_eval.py # TOFU evaluation
│   ├── muse_eval.py      # MUSE evaluation
│   └── wmdp_eval.py      # WMDP evaluation
├── configs/              # Experiment configs (main runs + alpha/beta sweeps)
├── scripts/              # Run/materialize/plot scripts for unlearning sweeps
├── finetune/             # SFT pipeline (see below)
│   ├── finetune_tofu.py  # Main SFT script (chat templates, FSDP-ready)
│   ├── evaluate_tofu.py  # OpenTOFU evaluation of fine-tuned models
│   ├── configs/          # Per-model/task SFT & eval configs
│   └── scripts/          # Per-model run scripts (single-GPU & FSDP)
├── environment.yml       # Conda environment (Python 3.13)
└── requirements-pip.txt  # Extra pip dependencies
```

## Setup

```bash
conda env create -f environment.yml
conda activate obdunlearning
pip install -r requirements-pip.txt
```

## Fine-Tuning (SFT)

Per-model scripts live in `finetune/scripts/` following the pattern
`run_<model>_<task>_{finetune,finetune_fsdp,eval,finetune_and_eval}.sh`, with tasks:
`tofu`, `muse_books[_qa]`, `muse_news[_qa]`, `wmdp`.

Single-GPU:

```bash
python finetune/finetune_tofu.py --config finetune/configs/<model>_<task>_sft.yaml
```

Multi-GPU FSDP (thin wrappers that runtime-patch the config and launch `torchrun`):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash finetune/scripts/run_<model>_<task>_finetune_fsdp.sh
```

Outputs are saved to `outputs/finetune/<model>_<task>_sft/model` and can be referenced as `model.name_or_path` in downstream unlearning/eval configs.

See [`finetune/README.md`](finetune/README.md) for per-model examples.

## Unlearning

Unlearning experiments are driven by YAML configs in `configs/`. Each config specifies the fine-tuned model, forget/retain data, the unlearning `method` (objective + wrapper), mask construction, and hyperparameters. Sweep configs (alpha/beta/main) are materialized by the `scripts/materialize_*.py` helpers and executed via `scripts/run_*.sh`, which skip already-completed method/output combinations.

```bash
bash scripts/run_<model>_probe_relative_main_sweeps.sh
```

Results (metrics + summaries) are written under the config's `output_dir`. Sweep results can be visualized with the corresponding `scripts/plot_*.py` scripts.

## Evaluation

```bash
# TOFU
python finetune/evaluate_tofu.py --config finetune/configs/<model>_tofu_sft_eval.yaml

# MUSE / WMDP
python scripts/evaluate_benchmark.py --config <config>.yaml
```

## Supported Models

The fine-tuning configs cover a wide range of model families: LLaMA 3.x (1B–8B), Qwen2.5/3/3.5/3.6 (0.5B–27B), Mistral/Ministral, Gemma, Phi-4-mini, SmoLLM, DeepSeek-R1 distills, and ChatGLM/GLM (with dedicated compatibility patches for tokenizer, LoRA targets, and FSDP gradient checkpointing).

## Hardware

Experiments were run on a node with 4× NVIDIA A5000 (24 GB) GPUs; FSDP with CPU offload, `use_orig_params`, and activation checkpointing is used for models that exceed single-GPU memory.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) — free for non-commercial research use with attribution. See [LICENSE](LICENSE) for details.
