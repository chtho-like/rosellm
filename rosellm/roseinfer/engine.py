import os
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, NamedTuple, Optional

import torch
from torch.profiler import record_function

from rosellm.roseinfer.detokenizer import (
    BaseDetokenizer,
    GPT2ByteDetokenizer,
    PrefixDiffDetokenizer,
)
from rosellm.rosetrainer.config import GPTConfig
from rosellm.rosetrainer.dataset import build_tokenizer
from rosellm.rosetrainer.model import GPTModel

try:
    import tiktoken
except ImportError:
    tiktoken = None


@contextmanager
def _maybe_nvtx_range(name: str, enabled: bool) -> Iterator[None]:
    if enabled:
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


class InferenceEngine:
    def __init__(
        self,
        checkpoint_path: str | None = None,
        tokenizer_name: str = "gpt2",
        device: Optional[str] = None,
        use_amp: bool = True,
        max_position_embeddings: Optional[int] = None,
        bf16: bool = False,
        kv_cache_max_concurrency: int = 256,
        prefix_cache_max_entries: int = 256,
        use_paged_attention: bool = False,
        model: GPTModel | None = None,
        config: GPTConfig | None = None,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.use_paged_attention = use_paged_attention
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.use_amp = use_amp and self.device.type == "cuda"
        if self.use_amp and self.device.type == "cuda":
            if bf16:
                self.amp_dtype: torch.dtype | None = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
        else:
            self.amp_dtype = None
        if model is None:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path must be provided when model is None")
            ckpt = torch.load(checkpoint_path, map_location=self.device.type)
            cfg_dict = ckpt.get("config")
            if cfg_dict is None:
                print("cannot find config from checkpoints, use GPTConfig")
                config = GPTConfig()
            else:
                config = GPTConfig(**cfg_dict)
            if max_position_embeddings is not None:
                if max_position_embeddings > config.max_position_embeddings:
                    raise ValueError(
                        "max_position_embeddings cannot exceed model max_position_embeddings "
                        f"({max_position_embeddings} > {config.max_position_embeddings})"
                    )
                config.max_position_embeddings = max_position_embeddings
            self.config = config
            self.model = GPTModel(config).to(self.device)
            self.model.load_state_dict(ckpt["model"])
        else:
            if config is None:
                raise ValueError("config must be provided when model is not None")
            if max_position_embeddings is not None:
                if max_position_embeddings > config.max_position_embeddings:
                    raise ValueError(
                        "max_position_embeddings cannot exceed model max_position_embeddings "
                        f"({max_position_embeddings} > {config.max_position_embeddings})"
                    )
                config.max_position_embeddings = max_position_embeddings
            self.config = config
            self.model = model.to(self.device)
        self.model.eval()
        if tokenizer is None:
            self.tokenizer = build_tokenizer(tokenizer_name)
        else:
            self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id

        def make_detok() -> BaseDetokenizer:
            tok_name = tokenizer_name
            if tok_name is None:
                tok_name = getattr(self.tokenizer, "name_or_path", "")
            if (
                isinstance(tok_name, str)
                and tok_name.startswith("gpt2")
                and tiktoken is not None
            ):
                try:
                    return GPT2ByteDetokenizer(self.tokenizer)
                except Exception as e:
                    print(f"failed to create GPT2ByteDetokenizer: {e}")
            return PrefixDiffDetokenizer(self.tokenizer)

        self._make_detok = make_detok
        block_size = 64
        max_context = max_position_embeddings or self.config.max_position_embeddings
        max_concurrency = max(1, kv_cache_max_concurrency)
        max_total_tokens = max_context * max_concurrency
        max_blocks_per_layer = (max_total_tokens + block_size - 1) // block_size
        self.block_size = block_size
        self.max_context = max_context
        self.max_blocks_per_seq = (max_context + block_size - 1) // block_size
        model_dtype = next(self.model.parameters()).dtype
        self.kv_manager = KVBlockManager(
            num_layers=self.config.n_layers,
            num_heads=self.config.n_heads,
            head_dim=self.config.d_model // self.config.n_heads,
            block_size=block_size,
            max_blocks_per_layer=max_blocks_per_layer,
            device=self.device,
            dtype=self.amp_dtype if self.use_amp else model_dtype,
        )
        self.prefix_cache = PrefixCache(
            self.kv_manager,
            max_entries=prefix_cache_max_entries,
        )
        self._paged_block_tables_buf: torch.Tensor | None = None
        self._paged_block_tables_capacity: int = 0
        self._paged_block_tables_cpu_buf: torch.Tensor | None = None
        self._paged_block_tables_cpu_capacity: int = 0
        self._paged_global_block_tables: torch.Tensor | None = None
        self._paged_slot_capacity: int = 0
        self._paged_free_slots: list[int] = []
        self._paged_dirty_rows_buf: torch.Tensor | None = None
        self._paged_dirty_rows_capacity: int = 0
        self._paged_dirty_rows_cpu_buf: torch.Tensor | None = None
        self._paged_dirty_rows_cpu_capacity: int = 0

        if self.config.vocab_size < self.tokenizer.vocab_size:
            raise ValueError("the model vocab_size is less than tokenizer vocab_size")

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            ids = [self.eos_token_id]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        return input_ids  # [1, T0]

    def _encode_prompt_token_ids(self, token_ids: list[int]) -> torch.Tensor:
        ids = list(token_ids)
        if not ids:
            ids = [self.eos_token_id]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        return input_ids  # [1, T0]

    def _encode_prompt_token_ids_batch(
        self,
        token_ids_list: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[list[int]]]:
        if not token_ids_list:
            raise ValueError("token_ids_list must be non-empty")
        max_pos = int(self.config.max_position_embeddings)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.eos_token_id

        truncated: list[list[int]] = []
        lengths: list[int] = []
        max_len = 0
        for ids0 in token_ids_list:
            ids = list(ids0)
            if not ids:
                ids = [self.eos_token_id]
            if len(ids) > max_pos:
                ids = ids[-max_pos:]
            truncated.append(ids)
            lengths.append(len(ids))
            max_len = max(max_len, len(ids))

        batch: list[list[int]] = []
        masks: list[list[int]] = []
        for ids in truncated:
            pad_len = max_len - len(ids)
            batch.append([pad_id] * pad_len + ids)
            masks.append([0] * pad_len + [1] * len(ids))

        input_ids = torch.tensor(
            batch,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.tensor(
            masks,
            dtype=torch.long,
            device=self.device,
        )
        return input_ids, attention_mask, lengths, truncated

    def _encode_prompts_batch(
        self,
        prompts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(prompts) > 0
        all_ids: list[list[int]] = []
        max_len = 0
        for text in prompts:
            ids = self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
            if not ids:
                ids = [self.eos_token_id]
            all_ids.append(ids)
            if len(ids) > max_len:
                max_len = len(ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.eos_token_id
        batch = []
        masks = []
        for ids in all_ids:
            pad_len = max_len - len(ids)
            batch.append([pad_id] * pad_len + ids)
            masks.append([0] * pad_len + [1] * len(ids))
        input_ids = torch.tensor(
            batch,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.tensor(
            masks,
            dtype=torch.long,
            device=self.device,
        )
        input_ids = self._maybe_truncate(input_ids)
        if input_ids.size(1) < attention_mask.size(1):
            attention_mask = attention_mask[:, -input_ids.size(1) :]
        return input_ids, attention_mask

    def _decode_tokens(self, token_ids: torch.Tensor) -> str:
        ids = token_ids.tolist()
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return text

    def _maybe_truncate(self, input_ids: torch.Tensor) -> torch.Tensor:
        max_pos = self.config.max_position_embeddings
        if input_ids.size(1) > max_pos:
            input_ids = input_ids[:, -max_pos:]
        return input_ids

    def _maybe_prefill_with_prefix_cache(
        self,
        session: "InferenceSession",
        prompt: str,
        use_prefix_cache: bool,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
        stop_on_eos: bool,
        prompt_token_ids: Optional[list[int]] = None,
    ) -> None:
        if prompt_token_ids is None:
            input_ids = self._encode_prompt(prompt)
        else:
            input_ids = self._encode_prompt_token_ids(prompt_token_ids)
        input_ids = self._maybe_truncate(input_ids)
        session.input_ids = input_ids
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens > 0:
            available = self.config.max_position_embeddings - input_ids.size(1)
            if available <= 0:
                session.finished = True
                return
            if max_new_tokens > available:
                print(
                    f"[warn] max_new_tokens clamped from {max_new_tokens} to {available} "
                    f"(max_position_embeddings={self.config.max_position_embeddings})"
                )
                max_new_tokens = available
        session.set_generation_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            stop_on_eos=stop_on_eos,
        )
        cached_logits = None
        if use_prefix_cache:
            cached_logits = self.prefix_cache.attach(prompt, session)
        if cached_logits is None:
            logits = session.prefill(input_ids)
            last_logits = logits[:, -1, :]
            session.kv_cache = None
            if use_prefix_cache:
                self.prefix_cache.put(prompt, session, last_logits)
        else:
            last_logits = cached_logits

        next_token = self._sample_next_token(
            last_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )
        token_id = int(next_token)
        session.generated_ids.append(token_id)
        session.step_count = 1
        if stop_on_eos:
            eos_id = self.eos_token_id
            if eos_id is not None and token_id == eos_id:
                session.finished = True
        if max_new_tokens > 0 and session.step_count >= max_new_tokens:
            session.finished = True

    @torch.no_grad()
    def _prefill_register_kv_batch(
        self,
        sessions: list["InferenceSession"],
        input_ids: torch.Tensor,  # [B, T]
        attention_mask: torch.Tensor,  # [B, T]
        lengths: list[int],  # [B]
    ) -> torch.Tensor:
        if len(sessions) != input_ids.size(0) or len(lengths) != input_ids.size(0):
            raise ValueError("batch size mismatch")
        from torch.amp import autocast

        position_ids = attention_mask.to(dtype=torch.long).cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        with record_function("roseinfer.prefill_batch.model_forward"):
            if self.use_amp:
                with autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                ):
                    logits, _, presents = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None,
                        past_key_values=None,
                        use_cache=True,
                        position_ids=position_ids,
                    )
            else:
                logits, _, presents = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                    past_key_values=None,
                    use_cache=True,
                    position_ids=position_ids,
                )
        kvm = self.kv_manager
        with record_function("roseinfer.prefill_batch.register_kv"):
            for layer_idx, layer_past in enumerate(presents):
                if layer_idx >= kvm.num_layers:
                    break
                k_layer, v_layer = layer_past  # [B, H, T, D]
                for b, sess in enumerate(sessions):
                    seq_len = int(lengths[b])
                    sess.prompt_length = seq_len
                    k = k_layer[b : b + 1, :, -seq_len:, :]
                    v = v_layer[b : b + 1, :, -seq_len:, :]
                    block_ids = kvm.register_prefill_layer(
                        layer_idx,
                        k,
                        v,
                    )
                    sess.block_ids_per_layer[layer_idx] = block_ids
        last_logits = logits[:, -1, :]  # [B, V]
        return last_logits

    def _top_k_logits(
        self,
        logits: torch.Tensor,  # [..., vocab]
        top_k: int,
    ) -> torch.Tensor:
        if top_k <= 0:
            return logits
        values, _ = torch.topk(logits, top_k)  # [..., k]
        min_values = values[..., -1, None]  # [..., 1]
        return torch.where(  # [..., vocab]
            logits < min_values,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    def _top_p_logits(
        self,
        logits: torch.Tensor,  # [..., vocab]
        top_p: float,
    ) -> torch.Tensor:
        if top_p <= 0.0 or top_p >= 1.0:
            return logits
        sorted_logits, sorted_idx = torch.sort(  # [..., vocab]
            logits,
            descending=True,
        )
        probs = torch.softmax(sorted_logits, dim=-1)  # [..., vocab]
        cum_probs = torch.cumsum(probs, dim=-1)  # [..., vocab]
        mask = cum_probs > top_p  # [..., vocab]
        mask[..., 0] = False  # keep at least one token
        sorted_logits = sorted_logits.masked_fill(
            mask,
            float("-inf"),
        )
        _, inv_idx = torch.sort(
            sorted_idx,
            dim=-1,
        )
        logits_filtered = torch.gather(
            sorted_logits,
            dim=-1,
            index=inv_idx,
        )
        return logits_filtered

    def _sample_next_token(
        self,
        logits: torch.Tensor,  # [..., vocab]
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
    ) -> int:
        if not do_sample or temperature <= 0.0:
            next_token = torch.argmax(logits, dim=-1)  # [..., 1]
            return int(next_token.item())
        scaled = logits / float(temperature)
        filtered = self._top_k_logits(scaled, top_k)
        filtered = self._top_p_logits(filtered, top_p)
        probs = torch.softmax(filtered, dim=-1)  # [..., vocab]
        probs = probs.clamp_min(1e-9)
        next_token = torch.multinomial(probs, num_samples=1)[:, 0]  # [..., 1]
        return int(next_token.item())

    def _sample_next_token_batch(
        self,
        logits: torch.Tensor,  # [batch, vocab]
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
    ) -> torch.Tensor:
        batch_size = logits.size(0)
        next_ids = []
        for i in range(batch_size):
            next_id = self._sample_next_token(
                logits=logits[i : i + 1],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            next_ids.append(next_id)
        return torch.tensor(
            next_ids,
            dtype=torch.long,
            device=self.device,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_on_eos: bool = True,
        do_sample: bool = False,
    ) -> str:
        self.model.eval()
        session = InferenceSession(self)
        try:
            input_ids = self._encode_prompt(prompt)  # [1, T0]
            input_ids = self._maybe_truncate(input_ids)  # [1, T]
            max_new_tokens = int(max_new_tokens)
            if max_new_tokens > 0:
                available = self.config.max_position_embeddings - input_ids.size(1)
                if available <= 0:
                    max_new_tokens = 0
                elif max_new_tokens > available:
                    max_new_tokens = available
            logits = session.prefill(input_ids)  # [1, T, V]
            last_logits = logits[:, -1, :]  # [1, V]
            generated_ids = input_ids[0].tolist()
            if max_new_tokens <= 0:
                generated = torch.tensor(
                    generated_ids,
                    dtype=torch.long,
                    device=self.device,
                )
                return self._decode_tokens(generated)
            next_id = self._sample_next_token(
                logits=last_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            generated_ids.append(next_id)
            last_token_id = next_id
            if (
                stop_on_eos
                and self.eos_token_id is not None
                and next_id == self.eos_token_id
            ):
                generated = torch.tensor(
                    generated_ids,
                    dtype=torch.long,
                    device=self.device,
                )
                return self._decode_tokens(generated)

            for _ in range(max_new_tokens - 1):
                next_logits = session.decode_step(last_token_id)
                next_id = self._sample_next_token(
                    logits=next_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                )
                generated_ids.append(next_id)
                last_token_id = next_id
                if (
                    stop_on_eos
                    and self.eos_token_id is not None
                    and next_id == self.eos_token_id
                ):
                    break
            generated = torch.tensor(
                generated_ids,
                dtype=torch.long,
                device=self.device,
            )
            text = self._decode_tokens(generated)
            return text
        finally:
            session.release_kv_blocks()

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_on_eos: bool = True,
        do_sample: bool = False,
    ) -> list[str]:
        assert len(prompts) > 0
        self.model.eval()
        session = InferenceSession(self)
        try:
            input_ids, attn_mask = self._encode_prompts_batch(prompts)
            batch_size = input_ids.size(0)
            last_logits = session.prefill_batch(
                input_ids,
                attention_mask=attn_mask,
            )
            lengths = attn_mask.sum(dim=1).tolist()
            generated_ids = [
                input_ids[b, -lengths[b] :].tolist() for b in range(batch_size)
            ]
            if max_new_tokens <= 0:
                outputs = []
                for ids in generated_ids:
                    t = torch.tensor(
                        ids,
                        dtype=torch.long,
                        device=self.device,
                    )
                    text = self._decode_tokens(t)
                    outputs.append(text)
                return outputs
            next_ids = self._sample_next_token_batch(
                logits=last_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            eos_positions: list[Optional[int]] = [None for _ in range(batch_size)]
            for b in range(batch_size):
                token_id = int(next_ids[b].item())
                generated_ids[b].append(token_id)
                if (
                    stop_on_eos
                    and self.eos_token_id is not None
                    and eos_positions[b] is None
                    and token_id == self.eos_token_id
                ):
                    eos_positions[b] = len(generated_ids[b]) - 1
            last_token_ids = next_ids
            for _ in range(max_new_tokens - 1):
                if (
                    stop_on_eos
                    and self.eos_token_id is not None
                    and all(pos is not None for pos in eos_positions)
                ):
                    break
                next_logits = session.decode_step_batch(last_token_ids)
                next_ids = self._sample_next_token_batch(
                    logits=next_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                )
                for b in range(batch_size):
                    token_id = int(next_ids[b].item())
                    if (
                        stop_on_eos
                        and self.eos_token_id is not None
                        and eos_positions[b] is not None
                    ):
                        continue
                    generated_ids[b].append(token_id)
                    if (
                        stop_on_eos
                        and self.eos_token_id is not None
                        and token_id == self.eos_token_id
                    ):
                        eos_positions[b] = len(generated_ids[b]) - 1
                last_token_ids = next_ids
            outputs: list[str] = []
            for b in range(batch_size):
                ids = generated_ids[b]
                if stop_on_eos and self.eos_token_id is not None:
                    pos = eos_positions[b]
                    if pos is not None:
                        ids = ids[: pos + 1]
                t = torch.tensor(
                    ids,
                    dtype=torch.long,
                    device=self.device,
                )
                text = self._decode_tokens(t)
                outputs.append(text)
            return outputs
        finally:
            session.release_kv_blocks()

    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_on_eos: bool = True,
        do_sample: bool = False,
    ) -> Iterator[str]:
        self.model.eval()
        session = InferenceSession(self)
        try:
            token_ids = self.tokenizer.encode(
                prompt,
                add_special_tokens=False,
            )
            if not token_ids:
                token_ids = [self.eos_token_id]
            ids_tensor = torch.tensor(
                [token_ids],
                dtype=torch.long,
                device=self.device,
            )
            detok = self._make_detok()
            detok.start_prompt(token_ids)
            prefill_logits = session.prefill(ids_tensor)  # [1, T, V]
            last_logits = prefill_logits[:, -1, :]  # [1, V]
            if max_new_tokens <= 0:
                piece = detok.flush()
                if piece:
                    yield piece
                return
            next_id = self._sample_next_token(
                logits=last_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            piece = detok.on_token(next_id)
            if piece:
                yield piece
            if (
                stop_on_eos
                and self.eos_token_id is not None
                and next_id == self.eos_token_id
            ):
                tail = detok.flush()
                if tail:
                    yield tail
                return
            last_token_id = next_id
            for _ in range(max_new_tokens - 1):
                next_logits = session.decode_step(last_token_id)  # [1, V]
                next_id = self._sample_next_token(
                    logits=next_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                )
                piece = detok.on_token(next_id)
                if piece:
                    yield piece
                last_token_id = next_id
                if (
                    stop_on_eos
                    and self.eos_token_id is not None
                    and next_id == self.eos_token_id
                ):
                    break
            tail = detok.flush()
            if tail:
                yield tail
        finally:
            session.release_kv_blocks()

    @torch.no_grad()
    def stream_generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_on_eos: bool = True,
        do_sample: bool = True,
    ) -> Iterator[list[str]]:
        self.model.eval()
        session = InferenceSession(self)
        try:
            batch_size = len(prompts)
            if batch_size == 0:
                return
            all_prompt_ids: list[list[int]] = []
            for p in prompts:
                ids = self.tokenizer.encode(
                    p,
                    add_special_tokens=False,
                )
                if not ids:
                    ids = [self.eos_token_id]
                all_prompt_ids.append(ids)
            detoks: list[BaseDetokenizer] = []
            for ids in all_prompt_ids:
                d = self._make_detok()
                d.start_prompt(ids)
                detoks.append(d)
            max_len = max(len(ids) for ids in all_prompt_ids)
            pad_id = self.eos_token_id
            batch_ids = []
            masks = []
            for ids in all_prompt_ids:
                pad_len = max_len - len(ids)
                batch_ids.append([pad_id] * pad_len + ids)
                masks.append([0] * pad_len + [1] * len(ids))
            input_ids = torch.tensor(
                batch_ids,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.tensor(
                masks,
                dtype=torch.long,
                device=self.device,
            )
            last_logits = session.prefill_batch(
                input_ids,
                attention_mask=attention_mask,
            )  # [B, V]
            if max_new_tokens <= 0:
                first_pieces = []
                for d in detoks:
                    tail = d.flush()
                    first_pieces.append(tail)
                if any(first_pieces):
                    yield first_pieces
                return
            next_ids: list[int] = []
            first_pieces: list[str] = []
            finished = [False for _ in range(batch_size)]
            for b in range(batch_size):
                logits_b = last_logits[b : b + 1]  # [1, V]
                tok_id = self._sample_next_token(
                    logits=logits_b,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                )
                next_ids.append(tok_id)
                piece = detoks[b].on_token(tok_id)
                if piece:
                    first_pieces.append(piece)
                else:
                    first_pieces.append("")
                if stop_on_eos and tok_id == self.eos_token_id:
                    finished[b] = True
            yield first_pieces
            last_token_ids = torch.tensor(
                next_ids,
                dtype=torch.long,
                device=self.device,
            )
            for _ in range(max_new_tokens - 1):
                next_logits = session.decode_step_batch(last_token_ids)
                new_ids: list[int] = []
                pieces: list[str] = []
                for b in range(batch_size):
                    logits_b = next_logits[b : b + 1]  # [1, V]
                    tok_id = self._sample_next_token(
                        logits=logits_b,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=do_sample,
                    )
                    new_ids.append(tok_id)
                    if stop_on_eos and finished[b]:
                        pieces.append("")
                        continue
                    piece = detoks[b].on_token(tok_id)
                    if piece:
                        pieces.append(piece)
                    else:
                        pieces.append("")
                    if stop_on_eos and tok_id == self.eos_token_id:
                        finished[b] = True
                last_token_ids = torch.tensor(
                    new_ids,
                    dtype=torch.long,
                    device=self.device,
                )
                yield pieces
                if all(finished):
                    break
            tails = []
            for b in range(batch_size):
                tail = detoks[b].flush()
                tails.append(tail)
            if any(tails):
                yield tails
        finally:
            session.release_kv_blocks()

    def _get_paged_block_tables_buf(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        if (
            self._paged_block_tables_buf is None
            or self._paged_block_tables_capacity < batch_size
        ):
            cap = max(batch_size, self._paged_block_tables_capacity * 2, 16)
            self._paged_block_tables_buf = torch.empty(
                (
                    self.config.n_layers,
                    cap,
                    self.max_blocks_per_seq,
                ),
                device=self.device,
                dtype=torch.int32,
            )
            self._paged_block_tables_capacity = cap
        return self._paged_block_tables_buf

    def _get_paged_block_tables_cpu_buf(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        if (
            self._paged_block_tables_cpu_buf is None
            or self._paged_block_tables_cpu_capacity < batch_size
        ):
            cap = max(batch_size, self._paged_block_tables_cpu_capacity * 2, 16)
            self._paged_block_tables_cpu_buf = torch.empty(
                (
                    self.config.n_layers,
                    cap,
                    self.max_blocks_per_seq,
                ),
                device="cpu",
                dtype=torch.int32,
                pin_memory=(self.device.type == "cuda"),
            )
            self._paged_block_tables_cpu_capacity = cap
        return self._paged_block_tables_cpu_buf

    def _ensure_paged_slot_capacity(
        self,
        min_capacity: int,
    ) -> None:
        if (
            self._paged_global_block_tables is not None
            and self._paged_slot_capacity >= min_capacity
        ):
            return
        min_capacity = max(1, int(min_capacity))
        new_cap = max(min_capacity, self._paged_slot_capacity * 2, 128)
        new_tables = torch.zeros(
            (self.config.n_layers, new_cap, self.max_blocks_per_seq),
            device=self.device,
            dtype=torch.int32,
        )
        if (
            self._paged_global_block_tables is not None
            and self._paged_slot_capacity > 0
        ):
            new_tables[:, : self._paged_slot_capacity].copy_(
                self._paged_global_block_tables
            )
        self._paged_global_block_tables = new_tables
        self._paged_free_slots.extend(range(self._paged_slot_capacity, new_cap))
        self._paged_slot_capacity = new_cap

    def _alloc_paged_slot(self) -> int:
        if not self._paged_free_slots:
            self._ensure_paged_slot_capacity(self._paged_slot_capacity + 1)
        return int(self._paged_free_slots.pop())

    def _free_paged_slot(self, slot_id: int) -> None:
        self._paged_free_slots.append(int(slot_id))

    def _get_paged_global_block_tables(self) -> torch.Tensor:
        if self._paged_global_block_tables is None:
            self._ensure_paged_slot_capacity(128)
        assert self._paged_global_block_tables is not None
        return self._paged_global_block_tables

    def _get_paged_dirty_rows_buf(
        self,
        n_rows: int,
    ) -> torch.Tensor:
        if (
            self._paged_dirty_rows_buf is None
            or self._paged_dirty_rows_capacity < n_rows
        ):
            cap = max(n_rows, self._paged_dirty_rows_capacity * 2, 16)
            self._paged_dirty_rows_buf = torch.empty(
                (cap, self.max_blocks_per_seq),
                device=self.device,
                dtype=torch.int32,
            )
            self._paged_dirty_rows_capacity = cap
        return self._paged_dirty_rows_buf

    def _get_paged_dirty_rows_cpu_buf(
        self,
        n_rows: int,
    ) -> torch.Tensor:
        if (
            self._paged_dirty_rows_cpu_buf is None
            or self._paged_dirty_rows_cpu_capacity < n_rows
        ):
            cap = max(n_rows, self._paged_dirty_rows_cpu_capacity * 2, 16)
            self._paged_dirty_rows_cpu_buf = torch.empty(
                (cap, self.max_blocks_per_seq),
                device="cpu",
                dtype=torch.int32,
                pin_memory=(self.device.type == "cuda"),
            )
            self._paged_dirty_rows_cpu_capacity = cap
        return self._paged_dirty_rows_cpu_buf

    @torch.no_grad()
    def decode_step_sessions(
        self,
        sessions: list["InferenceSession"],
    ) -> torch.Tensor:
        with record_function("roseinfer.decode_step_sessions.total"):
            assert sessions
            from torch.amp import autocast

            device = self.device
            batch_size = len(sessions)
            kvm = self.kv_manager

            last_ids: list[int] = []
            seq_lens: list[int] = []
            for sess in sessions:
                if sess.finished:
                    continue
                assert sess.generated_ids
                last_ids.append(sess.generated_ids[-1])
                seq_len = sess.prompt_length + sess.step_count - 1
                seq_lens.append(seq_len)
            assert len(last_ids) == batch_size
            lens = torch.tensor(seq_lens, device=device, dtype=torch.long)

            input_ids = torch.tensor(  # [B, 1]
                last_ids,
                dtype=torch.long,
                device=device,
            ).view(batch_size, 1)
            position_ids = lens.view(batch_size, 1)
            num_layers = kvm.num_layers
            num_heads = kvm.num_heads
            head_dim = kvm.head_dim
            if self.use_paged_attention:
                from rosellm.rosetrainer.paged_attention import PagedKVCache

                nvtx = device.type == "cuda" and os.environ.get("ROSEINFER_NVTX") == "1"
                block_size = kvm.block_size
                max_blocks_per_layer = kvm.max_blocks_per_layer
                slot_ids: list[int] = []
                for sess in sessions:
                    if sess.paged_slot_id is None:
                        sess.paged_slot_id = self._alloc_paged_slot()
                        sess.clear_paged_block_table_cache()
                    assert sess.paged_slot_id is not None
                    slot_ids.append(sess.paged_slot_id)
                global_block_tables = self._get_paged_global_block_tables()

                with _maybe_nvtx_range(
                    "roseinfer.decode_step_sessions.sync_global_block_tables",
                    nvtx,
                ), record_function(
                    "roseinfer.decode_step_sessions.sync_global_block_tables"
                ):
                    dirty_idx: list[int] = []
                    for idx, sess in enumerate(sessions):
                        _, dirty = sess.get_paged_block_table_row_cpu_and_dirty(
                            layer_idx=0,
                            offset=0,
                        )
                        if dirty:
                            dirty_idx.append(idx)
                    if dirty_idx:
                        dirty_slot_ids = [slot_ids[idx] for idx in dirty_idx]
                        dirty_slot_ids_t = torch.tensor(
                            dirty_slot_ids,
                            device=device,
                            dtype=torch.long,
                        )
                        n_dirty = len(dirty_idx)
                        rows_cpu = self._get_paged_dirty_rows_cpu_buf(n_dirty)[:n_dirty]
                        rows_buf = self._get_paged_dirty_rows_buf(n_dirty)[:n_dirty]
                        for layer_idx in range(num_layers):
                            offset = layer_idx * max_blocks_per_layer
                            rows = [
                                sessions[idx].get_paged_block_table_row_cpu(
                                    layer_idx=layer_idx,
                                    offset=offset,
                                )
                                for idx in dirty_idx
                            ]
                            torch.stack(rows, dim=0, out=rows_cpu)
                            rows_buf.copy_(rows_cpu, non_blocking=True)
                            global_block_tables[layer_idx].index_copy_(
                                0,
                                dirty_slot_ids_t,
                                rows_buf,
                            )

                slot_mapping = torch.tensor(
                    slot_ids,
                    device=device,
                    dtype=torch.int32,
                )
                block_tables = [
                    global_block_tables[layer_idx] for layer_idx in range(num_layers)
                ]
                paged = PagedKVCache(
                    k_cache=kvm._k_cache,
                    v_cache=kvm._v_cache,
                    block_tables=block_tables,
                    slot_mapping=slot_mapping,
                    context_lens=lens.to(torch.int32),
                    block_size=block_size,
                )
                with _maybe_nvtx_range(
                    "roseinfer.model.forward", nvtx
                ), record_function(
                    "roseinfer.model.forward",
                ):
                    if self.use_amp:
                        with autocast(device_type=device.type, dtype=self.amp_dtype):
                            logits, _, presents = self.model(
                                input_ids=input_ids,
                                attention_mask=None,
                                labels=None,
                                past_key_values=None,
                                use_cache=True,
                                position_ids=position_ids,
                                paged_kv_cache=paged,
                            )
                    else:
                        logits, _, presents = self.model(
                            input_ids=input_ids,
                            attention_mask=None,
                            labels=None,
                            past_key_values=None,
                            use_cache=True,
                            position_ids=position_ids,
                            paged_kv_cache=paged,
                        )
                last_logits = logits[:, -1, :]  # [B, V]
                with _maybe_nvtx_range(
                    "roseinfer.kv.append_token", nvtx
                ), record_function("roseinfer.kv.append_token"):
                    for layer_idx in range(num_layers):
                        k_step, v_step = presents[layer_idx]  # [B, H, 1, D]
                        k_step = k_step.squeeze(2)  # [B, H, D]
                        v_step = v_step.squeeze(2)
                        block_ids_list = [
                            sess.block_ids_per_layer[layer_idx] for sess in sessions
                        ]
                        kvm.append_token_batch(
                            layer_idx,
                            block_ids_list,
                            k_step,
                            v_step,
                        )
                return last_logits

            max_len = max(seq_lens)
            past_mask = torch.arange(
                max_len,
                device=device,
            ).unsqueeze(
                0
            ) < lens.unsqueeze(1)
            new_mask = torch.ones(
                batch_size,
                1,
                device=device,
                dtype=past_mask.dtype,
            )
            attention_mask = torch.cat(
                [past_mask, new_mask],
                dim=1,
            ).to(torch.long)

            batched_past = []
            with record_function("roseinfer.decode_step_sessions.build_batched_past"):
                for layer_idx in range(num_layers):
                    k_cat = torch.zeros(
                        [batch_size, num_heads, max_len, head_dim],
                        dtype=kvm.dtype,
                        device=device,
                    )
                    v_cat = torch.zeros_like(k_cat)
                    for idx, sess in enumerate(sessions):
                        seq_len = seq_lens[idx]
                        block_ids = sess.block_ids_per_layer[layer_idx]
                        kvm.gather_sequence_into(
                            layer_idx,
                            block_ids,
                            seq_len,
                            k_cat[idx],
                            v_cat[idx],
                        )
                    batched_past.append((k_cat, v_cat))
            with record_function("roseinfer.model.forward"):
                if self.use_amp:
                    with autocast(
                        device_type=device.type,
                        dtype=self.amp_dtype,
                    ):
                        logits, _, presents = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=None,
                            past_key_values=tuple(batched_past),
                            use_cache=True,
                            position_ids=position_ids,
                        )
                else:
                    logits, _, presents = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None,
                        past_key_values=tuple(batched_past),
                        use_cache=True,
                        position_ids=position_ids,
                    )
            last_logits = logits[:, -1, :]  # [B, V]
            with record_function("roseinfer.kv.append_token"):
                for layer_idx in range(num_layers):
                    k_b, v_b = presents[layer_idx]  # [B, H, max_len+1, D]
                    k_step = k_b.select(2, max_len)  # [B, H, D]
                    v_step = v_b.select(2, max_len)  # [B, H, D]
                    block_ids_list = [
                        sess.block_ids_per_layer[layer_idx] for sess in sessions
                    ]
                    kvm.append_token_batch(
                        layer_idx,
                        block_ids_list,
                        k_step,
                        v_step,
                    )
            return last_logits


class InferenceSession:
    def __init__(self, engine: "InferenceEngine") -> None:
        self.engine = engine
        self.kv_cache = None
        self.input_ids: torch.Tensor | None = None
        self.generated_ids: list[int] = []
        self.finished: bool = False
        self.max_new_tokens: int = 0
        self.temperature: float = 1.0
        self.top_k: int = 0
        self.top_p: float = 1.0
        self.do_sample: bool = False
        self.stop_on_eos: bool = True
        self.step_count: int = 0
        self.kv_manager = engine.kv_manager
        self.block_ids_per_layer: list[list[int]] = [
            [] for _ in range(self.kv_manager.num_layers)
        ]
        self.prompt_length: int = 0
        self.paged_slot_id: int | None = None
        self._paged_block_table_rows_cpu: list[torch.Tensor] | None = None
        self._paged_block_table_sig: list[tuple[int, int]] | None = None

    def clear_paged_block_table_cache(self) -> None:
        self._paged_block_table_rows_cpu = None
        self._paged_block_table_sig = None

    def get_paged_block_table_row_cpu_and_dirty(
        self,
        *,
        layer_idx: int,
        offset: int,
    ) -> tuple[torch.Tensor, bool]:
        max_blocks = int(self.engine.max_blocks_per_seq)
        if self._paged_block_table_rows_cpu is None:
            num_layers = int(self.kv_manager.num_layers)
            self._paged_block_table_rows_cpu = [
                torch.empty(
                    (max_blocks,),
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=(self.engine.device.type == "cuda"),
                ).zero_()
                for _ in range(num_layers)
            ]
            self._paged_block_table_sig = [(-1, -1) for _ in range(num_layers)]
        assert self._paged_block_table_sig is not None

        ids = self.block_ids_per_layer[layer_idx]
        sig = (len(ids), int(ids[-1]) if ids else -1)
        if sig != self._paged_block_table_sig[layer_idx]:
            row = self._paged_block_table_rows_cpu[layer_idx]
            row.zero_()
            if ids:
                n = min(len(ids), max_blocks)
                row[:n].copy_(
                    torch.tensor(
                        [gid - offset for gid in ids[:n]],
                        dtype=torch.int32,
                    )
                )
            self._paged_block_table_sig[layer_idx] = sig
            return row, True
        return self._paged_block_table_rows_cpu[layer_idx], False

    def get_paged_block_table_row_cpu(
        self,
        *,
        layer_idx: int,
        offset: int,
    ) -> torch.Tensor:
        row, _ = self.get_paged_block_table_row_cpu_and_dirty(
            layer_idx=layer_idx,
            offset=offset,
        )
        return row

    def set_generation_config(
        self,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
        stop_on_eos: bool,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.stop_on_eos = stop_on_eos

    def all_token_ids(self) -> list[int]:
        base_ids: list[int] = []
        if self.input_ids is not None:
            base_ids = list(self.input_ids[0].tolist())
        return base_ids + self.generated_ids

    def decode_text(self) -> str:
        token_ids = self.all_token_ids()
        return self.engine.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )

    def _register_prefill_kv(
        self,
        presents,
        seq_len: int,
    ) -> None:
        if self.kv_manager is None:
            return
        self.prompt_length = seq_len
        self.block_ids_per_layer = [[] for _ in range(self.kv_manager.num_layers)]
        for layer_idx, layer_past in enumerate(presents):
            if layer_idx >= self.kv_manager.num_layers:
                break
            key, value = layer_past  # [B, H, T, D]
            if key.size(2) != seq_len:
                continue
            block_ids = self.kv_manager.register_prefill_layer(
                layer_idx,
                key,
                value,
            )
            self.block_ids_per_layer[layer_idx] = block_ids

    @torch.no_grad()
    def prefill(
        self,
        prompt_ids: torch.Tensor,  # [..., T0]
    ):
        from torch.amp import autocast

        eng = self.engine
        input_ids = eng._maybe_truncate(prompt_ids)
        if eng.use_amp:
            with autocast(device_type=eng.device.type, dtype=eng.amp_dtype):
                logits, _, presents = eng.model(
                    input_ids=input_ids,
                    attention_mask=None,
                    labels=None,
                    past_key_values=None,
                    use_cache=True,
                )
        else:
            logits, _, presents = eng.model(
                input_ids=input_ids,
                attention_mask=None,
                labels=None,
                past_key_values=None,
                use_cache=True,
            )
        self._register_prefill_kv(presents, input_ids.size(1))
        self.kv_cache = presents
        return logits  # [..., T0, vocab]

    @torch.no_grad()
    def prefill_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from torch.amp import autocast

        eng = self.engine
        input_ids = eng._maybe_truncate(input_ids)
        if attention_mask is not None and input_ids.size(1) < attention_mask.size(1):
            attention_mask = attention_mask[:, -input_ids.size(1) :]
        if eng.use_amp:
            with autocast(
                device_type=eng.device.type,
                dtype=eng.amp_dtype,
            ):
                logits, _, presents = eng.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                    past_key_values=None,
                    use_cache=True,
                )
        else:
            logits, _, presents = eng.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                past_key_values=None,
                use_cache=True,
            )
        if input_ids.size(0) == 1:  # temporarily only support batch size 1
            self._register_prefill_kv(presents, input_ids.size(1))
        self.kv_cache = presents
        last_logits = logits[:, -1, :]  # [batch, vocab]
        return last_logits

    @torch.no_grad()
    def decode_step(self, last_token_id: int) -> torch.Tensor:
        assert self.kv_cache is not None
        from torch.amp import autocast

        eng = self.engine
        input_ids = torch.tensor(  # [1, 1]
            [[last_token_id]],
            dtype=torch.long,
            device=eng.device,
        )
        if eng.use_amp:
            with autocast(device_type=eng.device.type, dtype=eng.amp_dtype):
                logits, _, presents = eng.model(
                    input_ids=input_ids,
                    attention_mask=None,
                    labels=None,
                    past_key_values=self.kv_cache,
                    use_cache=True,
                )
        else:
            logits, _, presents = eng.model(
                input_ids=input_ids,
                attention_mask=None,
                labels=None,
                past_key_values=self.kv_cache,
                use_cache=True,
            )
        self.kv_cache = presents
        next_logits = logits[:, -1, :]  # [1, V]
        return next_logits  # [1, vocab]

    @torch.no_grad()
    def step_once(self) -> int | None:
        if self.finished:
            return None
        if not self.generated_ids:
            raise RuntimeError("no generated ids, call prefill first")
        last_token_id = self.generated_ids[-1]
        last_logits = self.decode_step(last_token_id)
        eng = self.engine
        next_token = eng._sample_next_token(
            last_logits,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=self.do_sample,
        )
        token_id = int(next_token)
        self.generated_ids.append(token_id)
        self.step_count += 1
        if self.stop_on_eos:
            eos_id = eng.eos_token_id
            if eos_id is not None and token_id == eos_id:
                self.finished = True
        if self.max_new_tokens > 0 and self.step_count >= self.max_new_tokens:
            self.finished = True
        return token_id

    @torch.no_grad()
    def apply_batch_logits(
        self,
        last_logits: torch.Tensor,
    ) -> int | None:
        if self.finished:
            return None
        eng = self.engine
        logits_2d = last_logits.view(1, -1)  # [1, V]
        next_token = eng._sample_next_token(
            logits_2d,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=self.do_sample,
        )
        token_id = int(next_token)
        self.generated_ids.append(token_id)
        self.step_count += 1
        if self.stop_on_eos:
            eos_id = eng.eos_token_id
            if eos_id is not None and token_id == eos_id:
                self.finished = True
        if self.max_new_tokens > 0 and self.step_count >= self.max_new_tokens:
            self.finished = True
        return token_id

    def release_kv_blocks(self) -> None:
        if self.paged_slot_id is not None:
            self.engine._free_paged_slot(self.paged_slot_id)
            self.paged_slot_id = None
            self.clear_paged_block_table_cache()
        self.kv_cache = None
        if self.kv_manager is None:
            return
        for layer_idx, block_ids in enumerate(self.block_ids_per_layer):
            if not block_ids:
                continue
            self.kv_manager.free_blocks(layer_idx, block_ids)
        self.block_ids_per_layer = [[] for _ in range(self.kv_manager.num_layers)]

    @torch.no_grad()
    def decode_step_batch(
        self,
        last_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        assert self.kv_cache is not None
        from torch.amp import autocast

        eng = self.engine
        input_ids = last_token_ids.view(-1, 1)  # [B, 1]
        if eng.use_amp:
            with autocast(
                device_type=eng.device.type,
                dtype=eng.amp_dtype,
            ):
                logits, _, presents = eng.model(
                    input_ids=input_ids,
                    attention_mask=None,
                    labels=None,
                    past_key_values=self.kv_cache,
                    use_cache=True,
                )
        else:
            logits, _, presents = eng.model(
                input_ids=input_ids,
                attention_mask=None,
                labels=None,
                past_key_values=self.kv_cache,
                use_cache=True,
            )
        self.kv_cache = presents
        next_logits = logits[:, -1, :]  # [B, V]
        return next_logits  # [B, V]


class PrefixCacheEntry:
    def __init__(
        self,
        prompt: str,
        prompt_length: int,
        blocks_ids_per_layer: list[list[int]],
        last_logits: torch.Tensor,
    ) -> None:
        self.prompt = prompt
        self.prompt_length = int(prompt_length)
        self.blocks_ids_per_layer = [list(ids) for ids in blocks_ids_per_layer]
        self.last_logits = last_logits.detach().to("cpu")


class PrefixCache:
    def __init__(
        self,
        kv_manager: "KVBlockManager",
        max_entries: int = 256,
    ) -> None:
        self.kv_manager = kv_manager
        self.max_entries = max(0, int(max_entries))
        self._entries: OrderedDict[str, PrefixCacheEntry] = OrderedDict()

    def _release_entry(self, entry: PrefixCacheEntry) -> None:
        for layer_idx, block_ids in enumerate(entry.blocks_ids_per_layer):
            if block_ids:
                self.kv_manager.free_blocks(layer_idx, block_ids)

    def _evict_one(self) -> None:
        if not self._entries:
            return
        _, entry = self._entries.popitem(last=False)
        self._release_entry(entry)

    def clear(self) -> None:
        while self._entries:
            _, entry = self._entries.popitem(last=False)
            self._release_entry(entry)

    def get(self, prompt: str) -> PrefixCacheEntry | None:
        return self._entries.get(prompt)

    def put(
        self,
        prompt: str,
        session: "InferenceSession",
        last_logits: torch.Tensor,
    ) -> None:
        if prompt in self._entries:
            self._entries.move_to_end(prompt)
            return
        if session.kv_manager is None:
            return
        prompt_length = session.prompt_length
        block_ids_per_layer = [list(ids) for ids in session.block_ids_per_layer]
        for block_ids in block_ids_per_layer:
            if not block_ids:
                continue
            self.kv_manager.incref_blocks(block_ids)
        entry = PrefixCacheEntry(
            prompt=prompt,
            prompt_length=prompt_length,
            blocks_ids_per_layer=block_ids_per_layer,
            last_logits=last_logits,
        )
        while self.max_entries > 0 and len(self._entries) >= self.max_entries:
            self._evict_one()
        self._entries[prompt] = entry
        self._entries.move_to_end(prompt)

    def attach(
        self,
        prompt: str,
        session: "InferenceSession",
    ) -> torch.Tensor | None:
        entry = self._entries.get(prompt)
        if entry is None:
            return None
        self._entries.move_to_end(prompt)
        session.prompt_length = entry.prompt_length
        session.block_ids_per_layer = []
        for block_ids in entry.blocks_ids_per_layer:
            if not block_ids:
                session.block_ids_per_layer.append([])
                continue
            self.kv_manager.incref_blocks(block_ids)
            session.block_ids_per_layer.append(list(block_ids))
        last_logits = entry.last_logits.to(session.engine.device)
        return last_logits


class OfflineScheduler:
    def __init__(
        self,
        engine: "InferenceEngine",
        use_prefix_cache: bool = True,
    ) -> None:
        self.engine = engine
        self.use_prefix_cache = use_prefix_cache
        self._sessions: dict[int, InferenceSession] = {}
        self._next_request_id: int = 0

    @torch.no_grad()
    def add_request(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_on_eos: bool = True,
        do_sample: bool = False,
        prompt_token_ids: Optional[list[int]] = None,
    ) -> int:
        eng = self.engine
        eng.model.eval()
        session = InferenceSession(eng)

        eng._maybe_prefill_with_prefix_cache(
            session=session,
            prompt=prompt,
            use_prefix_cache=self.use_prefix_cache,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            stop_on_eos=stop_on_eos,
            prompt_token_ids=prompt_token_ids,
        )
        request_id = self._next_request_id
        self._next_request_id += 1
        self._sessions[request_id] = session
        return request_id

    def has_unfinished(self) -> bool:
        return any(not sess.finished for sess in self._sessions.values())

    @torch.no_grad()
    def step(self) -> dict[int, int]:
        active_pairs: list[tuple[int, InferenceSession]] = [
            (rid, sess) for rid, sess in self._sessions.items() if not sess.finished
        ]
        if not active_pairs:
            return {}
        sessions = [pair[1] for pair in active_pairs]
        last_logits = self.engine.decode_step_sessions(sessions)
        step_tokens: dict[int, int] = {}
        for idx, (rid, sess) in enumerate(active_pairs):
            logits_row = last_logits[idx]
            token_id = sess.apply_batch_logits(logits_row)
            if token_id is not None:
                step_tokens[rid] = token_id
        return step_tokens

    @torch.no_grad()
    def run(self) -> dict[int, str]:
        while self.has_unfinished():
            self.step()
        outputs: dict[int, str] = {}
        for rid, session in self._sessions.items():
            outputs[rid] = session.decode_text()
        for session in self._sessions.values():
            session.release_kv_blocks()
        return outputs


@dataclass(frozen=True)
class OnlineRequest:
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    stop_on_eos: bool = True
    do_sample: bool = False
    prompt_token_ids: Optional[list[int]] = None
    request_id: Optional[int] = None


class OnlineScheduler:
    def __init__(
        self,
        engine: "InferenceEngine",
        max_batch_size: int = 8,
        use_prefix_cache: bool = True,
    ) -> None:
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.use_prefix_cache = use_prefix_cache
        self._sessions: dict[int, InferenceSession] = {}
        self._next_request_id: int = 0
        self._active_rids: deque[int] = deque()
        self._finished_ids: list[int] = []

    @torch.no_grad()
    def add_request(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_on_eos: bool = True,
        do_sample: bool = False,
        prompt_token_ids: Optional[list[int]] = None,
        request_id: Optional[int] = None,
    ) -> int:
        eng = self.engine
        eng.model.eval()
        session = InferenceSession(eng)
        eng._maybe_prefill_with_prefix_cache(
            session=session,
            prompt=prompt,
            use_prefix_cache=self.use_prefix_cache,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            stop_on_eos=stop_on_eos,
            prompt_token_ids=prompt_token_ids,
        )
        if session.finished:
            session.release_kv_blocks()
        if request_id is None:
            rid = self._next_request_id
            self._next_request_id += 1
        else:
            rid = int(request_id)
            if rid in self._sessions:
                raise ValueError(f"request_id {rid} already exists")
            if rid >= self._next_request_id:
                self._next_request_id = rid + 1
        self._sessions[rid] = session
        if not session.finished:
            self._active_rids.append(rid)
        return rid

    @torch.no_grad()
    def add_requests(
        self,
        requests: list[OnlineRequest],
    ) -> list[int]:
        if not requests:
            return []

        eng = self.engine
        eng.model.eval()

        used: set[int] = set()
        next_rid = self._next_request_id
        rids: list[int] = []

        def alloc_rid(desired: Optional[int]) -> int:
            nonlocal next_rid
            if desired is None:
                while next_rid in used or next_rid in self._sessions:
                    next_rid += 1
                rid = next_rid
                next_rid += 1
                used.add(rid)
                return rid
            rid = int(desired)
            if rid in used or rid in self._sessions:
                raise ValueError(f"request_id {rid} already exists")
            used.add(rid)
            if rid >= next_rid:
                next_rid = rid + 1
            return rid

        sessions: list[InferenceSession] = []
        token_ids_list: list[list[int]] = []
        last_logits_per_req: list[torch.Tensor | None] = [None for _ in requests]
        miss_idx: list[int] = []
        dup_of: dict[int, int] = {}
        first_idx_for_prompt: dict[str, int] = {}

        for i, req in enumerate(requests):
            rid = alloc_rid(req.request_id)
            rids.append(rid)

            if req.prompt_token_ids is None:
                ids = eng.tokenizer.encode(
                    req.prompt,
                    add_special_tokens=False,
                )
            else:
                ids = list(req.prompt_token_ids)
            if not ids:
                ids = [eng.eos_token_id]
            max_pos = int(eng.config.max_position_embeddings)
            if len(ids) > max_pos:
                ids = ids[-max_pos:]
            token_ids_list.append(ids)

            sess = InferenceSession(eng)
            sess.input_ids = eng._encode_prompt_token_ids(ids)

            max_new_tokens = int(req.max_new_tokens)
            if max_new_tokens > 0:
                available = max_pos - len(ids)
                if available <= 0:
                    sess.finished = True
                    sessions.append(sess)
                    continue
                if max_new_tokens > available:
                    max_new_tokens = available
            sess.set_generation_config(
                max_new_tokens=max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                do_sample=req.do_sample,
                stop_on_eos=req.stop_on_eos,
            )
            sessions.append(sess)

            if self.use_prefix_cache:
                src = first_idx_for_prompt.get(req.prompt)
                if src is not None:
                    dup_of[i] = src
                    continue
                first_idx_for_prompt[req.prompt] = i

                cached_logits = eng.prefix_cache.attach(req.prompt, sess)
                if cached_logits is not None:
                    last_logits_per_req[i] = cached_logits
                    continue

            miss_idx.append(i)

        if miss_idx:
            batch_token_ids = [token_ids_list[i] for i in miss_idx]
            input_ids, attn_mask, lengths, _ = eng._encode_prompt_token_ids_batch(
                batch_token_ids
            )
            batch_sessions = [sessions[i] for i in miss_idx]
            last_logits = eng._prefill_register_kv_batch(
                sessions=batch_sessions,
                input_ids=input_ids,
                attention_mask=attn_mask,
                lengths=lengths,
            )
            for b, idx in enumerate(miss_idx):
                logits = last_logits[b : b + 1]
                last_logits_per_req[idx] = logits
                if self.use_prefix_cache:
                    eng.prefix_cache.put(
                        requests[idx].prompt,
                        sessions[idx],
                        logits,
                    )

        if dup_of:
            kvm = eng.kv_manager
            for idx, src in dup_of.items():
                sess = sessions[idx]
                if sess.finished:
                    continue
                src_sess = sessions[src]
                if src_sess.finished:
                    sess.finished = True
                    continue
                sess.prompt_length = src_sess.prompt_length
                sess.block_ids_per_layer = [[] for _ in range(kvm.num_layers)]
                for layer_idx, block_ids in enumerate(src_sess.block_ids_per_layer):
                    if not block_ids:
                        continue
                    kvm.incref_blocks(block_ids)
                    sess.block_ids_per_layer[layer_idx] = list(block_ids)
                last_logits_per_req[idx] = last_logits_per_req[src]

        for idx, sess in enumerate(sessions):
            if sess.finished:
                continue
            logits = last_logits_per_req[idx]
            if logits is None:
                raise RuntimeError(f"missing prefill logits for request {rids[idx]}")
            token_id = eng._sample_next_token(
                logits=logits,
                temperature=sess.temperature,
                top_k=sess.top_k,
                top_p=sess.top_p,
                do_sample=sess.do_sample,
            )
            sess.generated_ids.append(int(token_id))
            sess.step_count = 1
            if sess.stop_on_eos:
                eos_id = eng.eos_token_id
                if eos_id is not None and int(token_id) == eos_id:
                    sess.finished = True
            if sess.max_new_tokens > 0 and sess.step_count >= sess.max_new_tokens:
                sess.finished = True
            if sess.finished:
                sess.release_kv_blocks()

        for rid, sess in zip(rids, sessions):
            self._sessions[rid] = sess
            if not sess.finished:
                self._active_rids.append(rid)

        self._next_request_id = next_rid
        return rids

    def has_unfinished(self) -> bool:
        while self._active_rids:
            rid = self._active_rids[0]
            sess = self._sessions.get(rid)
            if sess is None or sess.finished:
                self._active_rids.popleft()
                continue
            return True
        return False

    def num_unfinished(self) -> int:
        return sum(1 for sess in self._sessions.values() if not sess.finished)

    def is_finished(self, request_id: int) -> bool:
        session = self._sessions.get(request_id)
        if session is None:
            return True
        return session.finished

    def get_generated_ids(self, request_id: int) -> list[int]:
        session = self._sessions.get(request_id)
        if session is None:
            return []
        return list(session.generated_ids)

    def get_step_count(self, request_id: int) -> int:
        session = self._sessions.get(request_id)
        if session is None:
            return 0
        return int(session.step_count)

    @torch.no_grad()
    def step(self) -> dict[int, int]:
        if not self._active_rids:
            return {}
        selected_pairs: list[tuple[int, InferenceSession]] = []
        max_examine = len(self._active_rids)
        while (
            len(selected_pairs) < self.max_batch_size
            and self._active_rids
            and max_examine > 0
        ):
            max_examine -= 1
            rid = self._active_rids.popleft()
            sess = self._sessions.get(rid)
            if sess is None or sess.finished:
                continue
            selected_pairs.append((rid, sess))
        if not selected_pairs:
            return {}
        sessions = [sess for _, sess in selected_pairs]
        last_logits = self.engine.decode_step_sessions(sessions)
        step_tokens: dict[int, int] = {}
        just_finished: list[int] = []
        for idx, (rid, sess) in enumerate(selected_pairs):
            logits_row = last_logits[idx]
            token_id = sess.apply_batch_logits(logits_row)
            if token_id is not None:
                step_tokens[rid] = token_id
                if sess.finished:
                    just_finished.append(rid)
                    sess.release_kv_blocks()
                else:
                    self._active_rids.append(rid)
        if just_finished:
            self._finished_ids.extend(just_finished)
        return step_tokens

    def pop_finished_ids(self) -> list[int]:
        ids, self._finished_ids = self._finished_ids, []
        return ids

    def get_response(self, request_id: int) -> str:
        session = self._sessions[request_id]
        return session.decode_text()

    def pop_response(self, request_id: int) -> str:
        session = self._sessions.pop(request_id)
        session.release_kv_blocks()
        return session.decode_text()

    def discard_request(self, request_id: int) -> None:
        session = self._sessions.pop(request_id, None)
        if session is not None:
            session.release_kv_blocks()


class KVBlockInfo(NamedTuple):
    layer: int
    block_index: int
    start: int
    length: int


class KVBlockManager:
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int,
        max_blocks_per_layer: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks_per_layer = max_blocks_per_layer
        self.device = device
        self.dtype = dtype
        self._next_block_index: list[int] = [0 for _ in range(num_layers)]
        self._free_block_indices: list[list[int]] = [[] for _ in range(num_layers)]
        self._block_infos: dict[int, KVBlockInfo] = {}
        self._k_cache = torch.empty(
            (num_layers, max_blocks_per_layer, num_heads, block_size, head_dim),
            device=device,
            dtype=dtype,
        )
        self._v_cache = torch.empty_like(self._k_cache)
        self._block_refcounts: dict[int, int] = {}

    def _alloc_block_index(self, layer_idx: int) -> int:
        free_list = self._free_block_indices[layer_idx]
        if free_list:
            return free_list.pop()
        idx = self._next_block_index[layer_idx]
        if idx >= self.max_blocks_per_layer:
            raise RuntimeError(f"no more blocks available for layer {layer_idx}")
        self._next_block_index[layer_idx] += 1
        return idx

    def _to_global_block_id(
        self,
        layer_idx: int,
        block_index: int,
    ) -> int:
        return layer_idx * self.max_blocks_per_layer + block_index

    def register_prefill_layer(
        self,
        layer_idx: int,
        key: torch.Tensor,  # [1, H, T, D]
        value: torch.Tensor,
    ) -> list[int]:
        assert 0 <= layer_idx < self.num_layers
        seq_len = key.size(2)
        block_size = self.block_size
        num_blocks = (seq_len + block_size - 1) // block_size
        block_ids: list[int] = []
        for i in range(num_blocks):
            start = i * block_size
            end = min(start + block_size, seq_len)
            length = end - start
            k_slice = key[:, :, start:end, :]
            v_slice = value[:, :, start:end, :]
            block_idx = self._alloc_block_index(layer_idx)
            global_id = self._to_global_block_id(
                layer_idx,
                block_idx,
            )
            info = KVBlockInfo(
                layer=layer_idx,
                block_index=block_idx,
                start=start,
                length=length,
            )
            self._block_infos[global_id] = info
            self._k_cache[layer_idx, block_idx, :, :length, :].copy_(k_slice[0])
            self._v_cache[layer_idx, block_idx, :, :length, :].copy_(v_slice[0])
            self._block_refcounts[global_id] = 1
            block_ids.append(global_id)
        return block_ids

    def incref_blocks(
        self,
        block_ids: list[int],
    ) -> None:
        for global_id in block_ids:
            self._block_refcounts[global_id] = (
                self._block_refcounts.get(
                    global_id,
                    0,
                )
                + 1
            )

    def free_blocks(
        self,
        layer_idx: int,
        block_ids: list[int],
    ) -> None:
        for global_id in block_ids:
            ref = self._block_refcounts.get(global_id)
            if ref is None:
                continue
            ref -= 1
            if ref > 0:
                self._block_refcounts[global_id] = ref
                continue
            self._block_refcounts.pop(global_id, None)
            info = self._block_infos.pop(global_id, None)
            if info is None:
                continue
            assert info.layer == layer_idx
            self._free_block_indices[layer_idx].append(
                info.block_index,
            )

    def append_token(
        self,
        layer_idx: int,
        block_ids: list[int],
        key_new: torch.Tensor,  # [H, D]
        value_new: torch.Tensor,  # [H, D]
    ) -> None:
        assert 0 <= layer_idx < self.num_layers
        if not block_ids:
            block_idx = self._alloc_block_index(layer_idx)
            global_id = self._to_global_block_id(
                layer_idx,
                block_idx,
            )
            info = KVBlockInfo(
                layer=layer_idx,
                block_index=block_idx,
                start=0,
                length=0,
            )
            self._block_infos[global_id] = info
            self._block_refcounts[global_id] = 1
            block_ids.append(global_id)
        last_id = block_ids[-1]
        info = self._block_infos[last_id]
        ref = self._block_refcounts.get(last_id, 1)
        if ref > 1 and info.length < self.block_size:
            self._block_refcounts[last_id] = ref - 1
            block_idx = self._alloc_block_index(layer_idx)
            new_global_id = self._to_global_block_id(
                layer_idx,
                block_idx,
            )
            new_info = KVBlockInfo(
                layer=layer_idx,
                block_index=block_idx,
                start=info.start,
                length=info.length,
            )
            self._block_infos[new_global_id] = new_info
            self._k_cache[
                layer_idx,
                block_idx,
            ].copy_(self._k_cache[layer_idx, info.block_index])
            self._v_cache[
                layer_idx,
                block_idx,
            ].copy_(self._v_cache[layer_idx, info.block_index])
            self._block_refcounts[new_global_id] = 1
            block_ids[-1] = new_global_id
            last_id = new_global_id
            info = new_info
        if info.length >= self.block_size:
            block_idx = self._alloc_block_index(layer_idx)
            global_id = self._to_global_block_id(layer_idx, block_idx)
            info = KVBlockInfo(
                layer=info.layer,
                block_index=block_idx,
                start=info.start + info.length,
                length=0,
            )
            self._block_infos[global_id] = info
            self._block_refcounts[global_id] = 1
            block_ids.append(global_id)
            last_id = global_id
        info = self._block_infos[last_id]
        k_block = self._k_cache[layer_idx, info.block_index]
        v_block = self._v_cache[layer_idx, info.block_index]
        pos = info.length
        k_block[:, pos, :].copy_(key_new)
        v_block[:, pos, :].copy_(value_new)
        new_info = KVBlockInfo(
            layer=info.layer,
            block_index=info.block_index,
            start=info.start,
            length=info.length + 1,
        )
        self._block_infos[last_id] = new_info

    def append_token_batch(
        self,
        layer_idx: int,
        block_ids_list: list[list[int]],
        key_new: torch.Tensor,  # [B, H, D]
        value_new: torch.Tensor,  # [B, H, D]
    ) -> None:
        assert 0 <= layer_idx < self.num_layers
        assert key_new.dim() == 3 and value_new.dim() == 3
        assert key_new.shape == value_new.shape
        assert key_new.size(1) == self.num_heads and key_new.size(2) == self.head_dim

        batch_size = int(key_new.size(0))
        if batch_size == 0:
            return
        if len(block_ids_list) != batch_size:
            raise ValueError(
                f"block_ids_list size mismatch ({len(block_ids_list)} != {batch_size})"
            )

        fast_batch_idx: list[int] = []
        fast_block_idx: list[int] = []
        fast_pos: list[int] = []
        fast_last_gid: list[int] = []
        slow_batch_idx: list[int] = []

        for b, block_ids in enumerate(block_ids_list):
            if not block_ids:
                slow_batch_idx.append(b)
                continue
            last_gid = block_ids[-1]
            info = self._block_infos[last_gid]
            ref = self._block_refcounts.get(last_gid, 1)
            if ref != 1 or info.length >= self.block_size:
                slow_batch_idx.append(b)
                continue
            fast_batch_idx.append(b)
            fast_block_idx.append(info.block_index)
            fast_pos.append(info.length)
            fast_last_gid.append(last_gid)

        if fast_batch_idx:
            device = self.device
            idx_t = torch.tensor(
                fast_batch_idx,
                device=device,
                dtype=torch.long,
            )
            blk_t = torch.tensor(
                fast_block_idx,
                device=device,
                dtype=torch.long,
            )
            pos_t = torch.tensor(
                fast_pos,
                device=device,
                dtype=torch.long,
            )

            k_src = key_new.index_select(0, idx_t)
            v_src = value_new.index_select(0, idx_t)
            k_layer = self._k_cache[layer_idx]
            v_layer = self._v_cache[layer_idx]
            k_layer[blk_t, :, pos_t, :] = k_src
            v_layer[blk_t, :, pos_t, :] = v_src

            for gid in fast_last_gid:
                info = self._block_infos[gid]
                self._block_infos[gid] = KVBlockInfo(
                    layer=info.layer,
                    block_index=info.block_index,
                    start=info.start,
                    length=info.length + 1,
                )

        for b in slow_batch_idx:
            self.append_token(
                layer_idx,
                block_ids_list[b],
                key_new[b],
                value_new[b],
            )

    def gather_sequence_into(
        self,
        layer_idx: int,
        block_ids: list[int],
        total_len: int,
        out_k: torch.Tensor,  # [H, >=total_len, D]
        out_v: torch.Tensor,  # [H, >=total_len, D]
    ) -> None:
        assert 0 <= layer_idx < self.num_layers
        cur = 0
        for global_id in block_ids:
            info = self._block_infos[global_id]
            if info is None or info.layer != layer_idx:
                continue
            length = info.length
            if length <= 0:
                continue
            end = min(cur + length, total_len)
            take = end - cur
            if take <= 0:
                break
            k_block = self._k_cache[layer_idx, info.block_index]
            v_block = self._v_cache[layer_idx, info.block_index]
            out_k[:, cur:end, :].copy_(k_block[:, :take, :])
            out_v[:, cur:end, :].copy_(v_block[:, :take, :])
            cur = end
            if cur >= total_len:
                break

    def gather_sequence(
        self,
        layer_idx: int,
        block_ids: list[int],
        total_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert 0 <= layer_idx < self.num_layers
        k_seq = torch.zeros(
            (
                self.num_heads,
                total_len,
                self.head_dim,
            ),
            dtype=self.dtype,
            device=self.device,
        )
        v_seq = torch.zeros_like(k_seq)
        self.gather_sequence_into(
            layer_idx,
            block_ids,
            total_len,
            k_seq,
            v_seq,
        )
        k_seq.unsqueeze_(0)
        v_seq.unsqueeze_(0)
        return k_seq, v_seq
