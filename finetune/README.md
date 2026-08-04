# TOFU Fine-Tuning

This directory contains a copied and slightly extended TOFU SFT script.

## Files

- `finetune_tofu.py`: recommended script. Supports plain QA prompts and tokenizer chat templates.
- `finetune_tofu_basic.py`: direct copy of the original `scripts/finetune_tofu.py`.
- `configs/*.yaml`: model-family templates.

## Run

```bash
python finetune/finetune_tofu.py --config finetune/configs/llama3_2_1b_tofu_sft.yaml
```

The output is saved to:

```text
outputs/finetune/<name>/model
```

That path can be used later as `model.name_or_path` in unlearning configs.


## Qwen3.5-2B TOFU SFT + Eval

Fine-tune:

```bash
bash finetune/scripts/run_qwen3_5_2b_tofu_finetune.sh
```

Evaluate the fine-tuned model:

```bash
bash finetune/scripts/run_qwen3_5_2b_tofu_eval.sh
```

Run both sequentially:

```bash
bash finetune/scripts/run_qwen3_5_2b_tofu_finetune_and_eval.sh
```

The fine-tuned adapter/model is saved to:

```text
outputs/finetune/qwen3_5_2b_tofu_sft/model
```

The evaluation is saved to:

```text
outputs/finetune/qwen3_5_2b_tofu_sft_eval/open_tofu_eval.json
outputs/finetune/qwen3_5_2b_tofu_sft_eval/open_tofu_eval.md
```


## Additional Instruct SFT Scripts

Ministral3-3B-Instruct:

```bash
bash finetune/scripts/run_ministral3_3b_instruct_tofu_finetune.sh
bash finetune/scripts/run_ministral3_3b_instruct_tofu_eval.sh
bash finetune/scripts/run_ministral3_3b_instruct_tofu_finetune_and_eval.sh
```

Gemma4-E2B:

```bash
bash finetune/scripts/run_gemma4_e2b_tofu_finetune.sh
bash finetune/scripts/run_gemma4_e2b_tofu_eval.sh
bash finetune/scripts/run_gemma4_e2b_tofu_finetune_and_eval.sh
```

Phi-4-mini-instruct:

```bash
bash finetune/scripts/run_phi4_mini_instruct_tofu_finetune.sh
bash finetune/scripts/run_phi4_mini_instruct_tofu_eval.sh
bash finetune/scripts/run_phi4_mini_instruct_tofu_finetune_and_eval.sh
```

The model repository ids are stored in the corresponding YAML files under `finetune/configs/`; update `model.name_or_path` and `model.tokenizer_name_or_path` there if your local/HuggingFace id differs.


## Multi-GPU Fine-Tuning with FSDP

For limited GPU memory, use the multi-GPU fine-tuning scripts. They launch one process per GPU with PyTorch FSDP instead of single-process `device_map=auto`; this avoids the forward-loss mismatch observed with `device_map=auto` during full fine-tuning.

By default, the FSDP launcher treats `train.batch_size` as the per-GPU batch size (`train.fsdp_batch_size_mode: per_rank`). If you need to preserve an exact global micro-batch size, set `train.fsdp_batch_size_mode: global`; in that mode `train.batch_size` must be divisible by the number of GPUs.

Examples:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash finetune/scripts/run_phi4_mini_instruct_tofu_finetune_model_parallel.sh
CUDA_VISIBLE_DEVICES=0,1 bash finetune/scripts/run_ministral3_3b_instruct_tofu_finetune_model_parallel.sh
CUDA_VISIBLE_DEVICES=0,1 bash finetune/scripts/run_gemma4_e2b_tofu_finetune_model_parallel.sh
CUDA_VISIBLE_DEVICES=0,1 bash finetune/scripts/run_qwen3_5_2b_tofu_finetune_model_parallel.sh
```

You can also call the generic launcher directly:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash finetune/scripts/run_tofu_finetune_fsdp.sh finetune/configs/qwen3_5_2b_tofu_sft.yaml 2
```

Memory notes:

- The launcher enables CPU offload, fine-grained FSDP wrapping, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default. Activation checkpointing is disabled by default because it can conflict with FSDP execution order on some decoder-only models.
- If memory is comfortable and you want speed, set `train.fsdp_cpu_offload: false` in the config. If memory is still too tight, lower `train.batch_size` or `data.max_length` before trying `train.gradient_checkpointing: true`.
- With `train.fsdp_batch_size_mode: global`, `train.batch_size` is split across GPUs. With `per_rank`, it is used on every GPU.


## Supported Model Families

The script uses HuggingFace `AutoModelForCausalLM`, `AutoTokenizer`, and PEFT LoRA. It supports any compatible local path or HF repo id, including:

- Llama 3.2 Instruct, e.g. `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`
- Llama 3.3 Instruct, e.g. `meta-llama/Llama-3.3-70B-Instruct`
- Qwen Instruct models. Use `qwen_template_tofu_sft.yaml` and replace `Qwen/QWEN_MODEL_INSTRUCT` with the exact Qwen 3.5/3.6 repo id or a local model path.

For Qwen-style repositories, keep `trust_remote_code: true` if the model requires custom code.

## Prompt Format

Use chat templates for Instruct models:

```yaml
data:
  prompt_format: chat_template
```

Use plain TOFU QA format for base models or tokenizers without chat templates:

```yaml
data:
  prompt_format: plain_qa
```

## Suggested Additional Models To Consider

These are not added as fixed configs yet, because they depend on your compute and exact comparison goal:

- Qwen2.5/Qwen3 small Instruct models: good Llama-size comparison points.
- Mistral or Ministral Instruct: useful non-Llama architecture comparison.
- Gemma 2/3 Instruct: another strong open family, but tokenizer/license/runtime details should be checked first.
- Phi-3/Phi-4 mini Instruct: cheap sanity-check models for fast ablations.
