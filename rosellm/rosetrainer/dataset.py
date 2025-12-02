import random
from typing import List, Optional

import numpy as np
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
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        eos_id: Optional[int] = None
        if add_eos and hasattr(tokenizer, "eos_token_id"):
            eos_id = tokenizer.eos_token_id
        all_ids: List[int] = []
        total_tokens = 0
        total_files = 0
        for path in file_paths:
            total_files += 1
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            if not text.strip():
                continue
            ids = tokenizer.encode(
                text,
                add_special_tokens=False,
                verbose=False,
            )
            if not ids:
                continue
            all_ids.extend(ids)
            total_tokens += len(ids)
            if eos_id is not None:
                all_ids.append(eos_id)
                total_tokens += 1
            if max_tokens is not None and total_tokens >= max_tokens:
                all_ids = all_ids[:max_tokens]
                total_tokens = len(all_ids)
                break
        print(f"total files: {total_files}")
        print(f"total tokens: {total_tokens}")
        if len(all_ids) < seq_len:
            raise ValueError("the total number of tokens is less than seq_len")
        self.all_ids = all_ids
        self.total_tokens = len(all_ids)
        max_start = self.total_tokens - seq_len
        num_samples = self.total_tokens // seq_len
        if num_samples > max_start + 1:
            num_samples = max_start + 1
        if seed is None:
            rng = random.Random()
        else:
            rng = random.Random(seed)
        candidates = list(range(max_start + 1))
        if num_samples < len(candidates):
            start_indices = rng.sample(candidates, num_samples)
        else:
            rng.shuffle(candidates)
            start_indices = candidates
        self.start_indices = start_indices
        self.num_samples = len(self.start_indices)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        start = self.start_indices[idx]
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


class FineWebNPYDataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        seq_len: int,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        random_start: bool = True,
    ) -> None:
        super().__init__()
        if not file_paths:
            raise ValueError("file_paths is empty")
        self.seq_len = seq_len
        arrays: List[np.ndarray] = []
        total = 0
        for path in file_paths:
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 1:
                raise ValueError(f"array {path} is not 1D")
            arr = arr.astype(np.int64, copy=False)
            if max_tokens is not None:
                remaining = max_tokens - total
                if remaining <= 0:
                    break
                if arr.shape[0] > remaining:
                    arr = arr[:remaining]
            arrays.append(arr)
            total += arr.shape[0]
        if not arrays:
            raise ValueError("no arrays loaded")
        self.tokens = np.concatenate(arrays, axis=0)
        self.total_tokens = int(self.tokens.shape[0])
        if self.total_tokens < seq_len:
            raise ValueError("the total number of tokens is less than seq_len")
        max_id = int(self.tokens.max())
        self.vocab_size = max_id + 1
        max_start = self.total_tokens - seq_len
        num_samples = self.total_tokens // seq_len
        if num_samples > max_start + 1:
            num_samples = max_start + 1
        if seed is None:
            rng = random.Random()
        else:
            rng = random.Random(seed)
        if random_start:
            candidates = list(range(max_start + 1))
            if num_samples < len(candidates):
                start_indices = rng.sample(candidates, num_samples)
            else:
                rng.shuffle(candidates)
                start_indices = candidates
        else:
            start_indices = [i * seq_len for i in range(num_samples)]
        self.start_indices = np.array(start_indices, dtype=np.int64)
        self.num_samples = int(self.start_indices.shape[0])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        start = int(self.start_indices[idx])
        end = start + self.seq_len
        ids = self.tokens[start:end]
        input_ids = torch.from_numpy(ids).long()
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
