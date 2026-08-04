from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


PROMPT_TEMPLATE = "Question: {question}\nAnswer:"


@dataclass(frozen=True)
class TofuExample:
    question: str
    answer: str
    profile_id: int


class TofuQADataset(Dataset):
    def __init__(self, examples: list[TofuExample], tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        prompt = PROMPT_TEMPLATE.format(question=ex.question)
        full_text = f"{prompt} {ex.answer}"
        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        labels = list(enc["input_ids"])
        prompt_len = min(len(prompt_enc["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "question": ex.question,
            "answer": ex.answer,
            "profile_id": ex.profile_id,
        }


def load_tofu_examples(
    dataset_name: str,
    split: str,
    profile_size: int,
    profile_ids: list[int],
    max_samples: int | None = None,
) -> list[TofuExample]:
    raw = load_dataset(dataset_name, split=split)
    wanted = set(profile_ids)
    examples: list[TofuExample] = []
    for idx, row in enumerate(raw):
        profile_id = idx // profile_size
        if profile_id not in wanted:
            continue
        examples.append(
            TofuExample(
                question=row["question"],
                answer=row["answer"],
                profile_id=profile_id,
            )
        )
        if max_samples and len(examples) >= max_samples:
            break
    return examples


def load_tofu_config_examples(
    dataset_name: str,
    config_name: str,
    split: str = "train",
    max_samples: int | None = None,
    question_key: str = "question",
    answer_key: str = "answer",
) -> list[TofuExample]:
    raw = load_dataset(dataset_name, config_name, split=split)
    examples: list[TofuExample] = []
    for idx, row in enumerate(raw):
        examples.append(
            TofuExample(
                question=row[question_key],
                answer=row[answer_key],
                profile_id=idx,
            )
        )
        if max_samples and len(examples) >= max_samples:
            break
    return examples


def load_tofu_raw_config(
    dataset_name: str,
    config_name: str,
    split: str = "train",
    max_samples: int | None = None,
):
    raw = load_dataset(dataset_name, config_name, split=split)
    if max_samples:
        raw = raw.select(range(min(max_samples, len(raw))))
    return raw


LETTERS = ["A", "B", "C", "D", "E", "F"]


def format_mcq_question(question: str, choices: list[str]) -> str:
    lines = [question]
    for idx, choice in enumerate(choices):
        lines.append(f"{LETTERS[idx]}. {choice}")
    return "\n".join(lines)


def load_wmdp_examples(
    dataset_name: str,
    config_name: str,
    split: str = "test",
    max_samples: int | None = None,
) -> list[TofuExample]:
    raw = load_dataset(dataset_name, config_name, split=split)
    examples: list[TofuExample] = []
    for idx, row in enumerate(raw):
        choices = list(row["choices"])
        answer_idx = int(row["answer"])
        examples.append(
            TofuExample(
                question=format_mcq_question(row["question"], choices),
                answer=LETTERS[answer_idx],
                profile_id=idx,
            )
        )
        if max_samples and len(examples) >= max_samples:
            break
    return examples


TEXT_KEYS = ("text", "document", "article", "content", "completion")


def row_to_text(row: dict, text_key: str | None = None) -> str:
    if text_key and text_key in row:
        return str(row[text_key]).strip()
    if "prompt" in row and "gt" in row:
        return f"{row['prompt']} {row['gt']}".strip()
    key = text_key or first_existing_key(row, TEXT_KEYS)
    if key is None:
        raise KeyError(f"Could not find a text field. Available fields: {list(row.keys())}")
    return str(row[key]).strip()


def load_text_completion_examples(
    dataset_name: str,
    config_name: str,
    split: str,
    max_samples: int | None = None,
    text_key: str | None = None,
    prompt_fraction: float = 0.35,
    min_answer_chars: int = 32,
    tokenizer=None,
    block_tokens: int | None = None,
    block_overlap_tokens: int = 0,
    min_answer_tokens: int = 8,
    text_chunk_chars: int | None = None,
) -> list[TofuExample]:
    raw = load_dataset(dataset_name, config_name, split=split)
    examples: list[TofuExample] = []
    for idx, row in enumerate(raw):
        try:
            text = row_to_text(row, text_key)
        except KeyError as exc:
            raise KeyError(
                f"Could not find a text field in {dataset_name}/{config_name}/{split}. "
                f"Available fields: {list(row.keys())}"
            ) from exc
        if tokenizer is not None and block_tokens:
            chunk_chars = text_chunk_chars or max(4096, block_tokens * 8)
            block_idx = 0
            stop = False
            for text_start in range(0, len(text), chunk_chars):
                text_piece = text[text_start : text_start + chunk_chars]
                if not text_piece.strip():
                    continue
                token_ids = tokenizer(
                    text_piece,
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                    return_tensors=None,
                )["input_ids"]
                if len(token_ids) <= min_answer_tokens + 1:
                    continue
                stride = max(1, block_tokens - max(0, block_overlap_tokens))
                for start in range(0, len(token_ids), stride):
                    block = token_ids[start : start + block_tokens]
                    if len(block) <= min_answer_tokens + 1:
                        continue
                    pivot = max(1, min(len(block) - min_answer_tokens, int(len(block) * prompt_fraction)))
                    prompt = tokenizer.decode(block[:pivot], skip_special_tokens=True).strip()
                    answer = tokenizer.decode(block[pivot:], skip_special_tokens=True).strip()
                    if not prompt or not answer:
                        continue
                    examples.append(
                        TofuExample(
                            question=prompt,
                            answer=answer,
                            profile_id=idx * 100000 + block_idx,
                        )
                    )
                    block_idx += 1
                    if max_samples and len(examples) >= max_samples:
                        stop = True
                        break
                if stop:
                    break
        else:
            if len(text) <= min_answer_chars:
                continue
            pivot = max(1, min(len(text) - min_answer_chars, int(len(text) * prompt_fraction)))
            examples.append(
                TofuExample(
                    question=text[:pivot].strip(),
                    answer=text[pivot:].strip(),
                    profile_id=idx,
                )
            )
        if max_samples and len(examples) >= max_samples:
            break
    return examples


def first_existing_key(row: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in row:
            return key
    return None


def load_config_examples(data_cfg: dict, split_cfg: dict, tokenizer=None) -> list[TofuExample]:
    dataset_format = split_cfg.get("dataset_format", data_cfg.get("dataset_format", "qa"))
    dataset_name = split_cfg.get("dataset_name", data_cfg["dataset_name"])
    config_name = split_cfg.get("config_name", split_cfg.get("name"))
    split = split_cfg.get("split", data_cfg.get("split", "train"))
    max_samples = split_cfg.get("max_samples")
    if dataset_format == "wmdp":
        return load_wmdp_examples(dataset_name, config_name, split, max_samples)
    if dataset_format in {"text", "muse_text"}:
        return load_text_completion_examples(
            dataset_name,
            config_name,
            split,
            max_samples,
            text_key=split_cfg.get("text_key", data_cfg.get("text_key")),
            prompt_fraction=split_cfg.get("text_prompt_fraction", data_cfg.get("text_prompt_fraction", 0.35)),
            min_answer_chars=split_cfg.get("min_answer_chars", data_cfg.get("min_answer_chars", 32)),
            tokenizer=tokenizer,
            block_tokens=split_cfg.get(
                "text_block_tokens",
                data_cfg.get("text_block_tokens", data_cfg.get("max_length") if dataset_format == "muse_text" else None),
            ),
            block_overlap_tokens=split_cfg.get("text_block_overlap_tokens", data_cfg.get("text_block_overlap_tokens", 0)),
            min_answer_tokens=split_cfg.get("min_answer_tokens", data_cfg.get("min_answer_tokens", 8)),
            text_chunk_chars=split_cfg.get("text_chunk_chars", data_cfg.get("text_chunk_chars", 2048)),
        )
    if dataset_format in {"qa", "tofu", "muse"}:
        return load_tofu_config_examples(
            dataset_name,
            config_name,
            split,
            max_samples,
            question_key=split_cfg.get("question_key", data_cfg.get("question_key", "question")),
            answer_key=split_cfg.get("answer_key", data_cfg.get("answer_key", "answer")),
        )
    raise ValueError(f"Unsupported dataset_format: {dataset_format}")


def make_collator(tokenizer):
    def collate(batch: list[dict]) -> dict:
        input_rows = [
            {
                "input_ids": row["input_ids"],
                "attention_mask": row["attention_mask"],
            }
            for row in batch
        ]
        padded = tokenizer.pad(input_rows, padding=True, return_tensors="pt")
        max_len = padded["input_ids"].shape[1]
        labels = []
        for row in batch:
            label = row["labels"]
            labels.append(label + [-100] * (max_len - len(label)))
        import torch

        padded["labels"] = torch.tensor(labels, dtype=torch.long)
        padded["questions"] = [row["question"] for row in batch]
        padded["answers"] = [row["answer"] for row in batch]
        padded["profile_ids"] = [row["profile_id"] for row in batch]
        return padded

    return collate


def make_loader(examples, tokenizer, max_length: int, batch_size: int, shuffle: bool):
    dataset = TofuQADataset(examples, tokenizer, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collator(tokenizer),
    )
