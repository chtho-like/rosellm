from typing import List

import torch
from torch.utils.data import Dataset

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


def build_tokenizer(
    model_name_or_path: str,
    use_fast: bool = True,
):
    if AutoTokenizer is None:
        raise ImportError("need to install transformers")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class TextDatasetForCausalLM(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        tokenizer,
        seq_len: int,
        add_eos: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.add_eos = add_eos
        texts: List[str] = []
        for path in file_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    texts.append(line)
        if not texts:
            raise ValueError("the file is full of empty lines")
        all_ids: List[int] = []
        for text in texts:
            if add_eos and hasattr(tokenizer, "eos_token_id"):
                encoded = tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
                all_ids.extend(encoded)
                all_ids.append(tokenizer.eos_token_id)
            else:
                encoded = tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
                all_ids.extend(encoded)
        if len(all_ids) < seq_len:
            raise ValueError("the total number of tokens is less than seq_len")
        self.all_ids = all_ids
        self.num_samples = len(all_ids) // seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end = start + self.seq_len
        ids = self.all_ids[start:end]
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
