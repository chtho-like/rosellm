import importlib.util
import os
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch
from torch.profiler import record_function

from rosellm.roseinfer.detokenizer import BaseDetokenizer, build_detokenizer
from rosellm.rosetrainer.config import GPTConfig
from rosellm.rosetrainer.dataset import build_tokenizer
from rosellm.rosetrainer.model import GPTModel

try:
    import flashinfer  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    flashinfer = None  # type: ignore[assignment]

PrefixCacheKey = str | tuple[int, ...]


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


@dataclass
class _PagedDecodeCudaGraph:
    batch_size: int
    global_block_tables_ptr: int
    graph: torch.cuda.CUDAGraph
    input_ids: torch.Tensor  # [B, 1] int64 cuda
    position_ids: torch.Tensor  # [B, 1] int64 cuda
    slot_mapping: torch.Tensor  # [B] int32 cuda
    context_lens: torch.Tensor  # [B] int32 cuda
    input_ids_host: torch.Tensor  # [B] int64 cpu (pinned)
    position_ids_host: torch.Tensor  # [B] int64 cpu (pinned)
    slot_mapping_host: torch.Tensor  # [B] int32 cpu (pinned)
    context_lens_host: torch.Tensor  # [B] int32 cpu (pinned)
    input_ids_host_np: np.ndarray
    position_ids_host_np: np.ndarray
    slot_mapping_host_np: np.ndarray
    context_lens_host_np: np.ndarray
    logits: torch.Tensor  # [B, 1, vocab]
    presents: list[tuple[torch.Tensor, torch.Tensor]]


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
        use_paged_attention: bool = True,
        use_cuda_graph: bool = True,
        prefill_attn_backend: str = "auto",
        decode_attn_backend: str = "auto",
        use_fused_ops: bool = True,
        use_fused_mlp: bool = True,
        use_fused_sampler: bool = True,
        use_fused_kv_append: bool = True,
        model: GPTModel | None = None,
        config: GPTConfig | None = None,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.use_paged_attention = bool(use_paged_attention)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if self.use_paged_attention and (
            self.device.type != "cuda" or not torch.cuda.is_available()
        ):
            self.use_paged_attention = False

        self.use_cuda_graph = bool(use_cuda_graph) and self.use_paged_attention
        self.use_cuda_graph = (
            self.use_cuda_graph
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        self.use_amp = use_amp and self.device.type == "cuda"
        if self.use_amp and self.device.type == "cuda":
            if bf16:
                self.amp_dtype: torch.dtype | None = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
        else:
            self.amp_dtype = None

        def _module_available(name: str) -> bool:
            try:
                return importlib.util.find_spec(name) is not None
            except Exception:
                return False

        def _resolve_prefill_backend(name: str) -> str:
            name = str(name or "auto").lower()
            if name not in ("auto", "auto2"):
                return name
            if self.device.type != "cuda" or not torch.cuda.is_available():
                return "naive"
            if not self.use_amp or self.amp_dtype not in (
                torch.float16,
                torch.bfloat16,
            ):
                return "naive"
            if name == "auto2":
                if _module_available("flash_attn"):
                    return "flashattn"
                if _module_available("flashinfer"):
                    return "flashinfer"
            else:
                if _module_available("flashinfer"):
                    return "flashinfer"
                if _module_available("flash_attn"):
                    return "flashattn"
            return "naive"

        def _resolve_decode_backend(name: str) -> str:
            name = str(name or "auto").lower()
            if name == "auto":
                return "naive"
            # Dense past_kv decode only supports the eager kernel in our GPT2 stack.
            if not self.use_paged_attention and name in ("flashinfer", "flashattn"):
                return "naive"
            return name

        self.prefill_attn_backend = _resolve_prefill_backend(prefill_attn_backend)
        self.decode_attn_backend = _resolve_decode_backend(decode_attn_backend)
        self.use_fused_ops = bool(use_fused_ops)
        self.use_fused_mlp = bool(use_fused_mlp)
        self.use_fused_sampler = bool(use_fused_sampler)
        self.use_fused_kv_append = bool(use_fused_kv_append)
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
        if hasattr(self.model, "use_fused_ops"):
            self.model.use_fused_ops = bool(self.use_fused_ops)
        if hasattr(self.model, "use_fused_mlp"):
            self.model.use_fused_mlp = bool(self.use_fused_mlp)
        if tokenizer is None:
            self.tokenizer = build_tokenizer(tokenizer_name)
        else:
            self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self._make_detok = lambda: build_detokenizer(
            self.tokenizer, tokenizer_name=tokenizer_name
        )
        block_size = 64
        max_context = max_position_embeddings or self.config.max_position_embeddings
        max_concurrency = max(1, kv_cache_max_concurrency)
        self.kv_cache_max_concurrency = max_concurrency
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
        self._paged_decode_cuda_graphs: dict[int, _PagedDecodeCudaGraph] = {}
        self._cuda_graph_pool = (
            torch.cuda.graphs.graph_pool_handle() if self.use_cuda_graph else None
        )

        self._sampling_generator: torch.Generator | None = None
        if (
            self.device.type == "cuda"
            and self.use_fused_sampler
            and flashinfer is not None
        ):
            self._sampling_generator = torch.Generator(device=self.device)
            seed_env = os.environ.get("ROSEINFER_SAMPLING_SEED")
            if seed_env is not None and seed_env.strip():
                self._sampling_generator.manual_seed(int(seed_env))
            else:
                self._sampling_generator.manual_seed(int(torch.initial_seed()))

        if self.config.vocab_size < self.tokenizer.vocab_size:
            raise ValueError("the model vocab_size is less than tokenizer vocab_size")

    def warmup_paged_attention_decode(self) -> None:
        if not self.use_paged_attention:
            return
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return
        token_id = int(self.eos_token_id or 0)
        sess = InferenceSession(self)
        sess.prompt_length = 0
        sess.generated_ids = [token_id]
        sess.step_count = 1
        sess.committed_step_count = 1
        try:
            self.decode_step_sessions([sess])
            torch.cuda.synchronize()
        finally:
            sess.release_kv_blocks()

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
        cache_key: PrefixCacheKey = prompt
        if prompt_token_ids is not None:
            ids = list(prompt_token_ids)
            if not ids:
                ids = [self.eos_token_id]
            max_pos = int(self.config.max_position_embeddings)
            if len(ids) > max_pos:
                ids = ids[-max_pos:]
            cache_key = tuple(ids)
        cached_logits = None
        if use_prefix_cache:
            cached_logits = self.prefix_cache.attach(cache_key, session)
        if cached_logits is None:
            logits = session.prefill(input_ids)
            last_logits = logits[:, -1, :]
            session.kv_cache = None
            if use_prefix_cache:
                self.prefix_cache.put(cache_key, session, last_logits)
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
        session.committed_step_count = 1
        if stop_on_eos:
            eos_id = self.eos_token_id
            if eos_id is not None and token_id == eos_id:
                session.finished = True
        if max_new_tokens > 0 and session.committed_step_count >= max_new_tokens:
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

        seq_len = int(input_ids.size(1))
        no_padding = all(int(l) == seq_len for l in lengths)
        if no_padding:
            position_ids = torch.arange(
                seq_len,
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
            if position_ids.size(0) != input_ids.size(0):
                position_ids = position_ids.expand(input_ids.size(0), -1).contiguous()
            attn_mask_for_model = None
        else:
            position_ids = attention_mask.to(dtype=torch.long).cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            attn_mask_for_model = attention_mask

        with record_function("roseinfer.prefill_batch.model_forward"):
            if self.use_amp:
                with autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                ):
                    logits, _, presents = self.model(
                        input_ids=input_ids,
                        attention_mask=attn_mask_for_model,
                        labels=None,
                        past_key_values=None,
                        use_cache=True,
                        position_ids=position_ids,
                        attn_backend=self.prefill_attn_backend,
                    )
            else:
                logits, _, presents = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask_for_model,
                    labels=None,
                    past_key_values=None,
                    use_cache=True,
                    position_ids=position_ids,
                    attn_backend=self.prefill_attn_backend,
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

    def _sync_paged_global_block_tables_for_sessions(
        self,
        sessions: list["InferenceSession"],
    ) -> None:
        if not sessions:
            return
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return
        global_block_tables = self._get_paged_global_block_tables()
        max_blocks_per_layer = int(self.kv_manager.max_blocks_per_layer)
        num_layers = int(self.kv_manager.num_layers)

        dirty_idx: list[int] = []
        slot_ids: list[int] = []
        for idx, sess in enumerate(sessions):
            if sess.paged_slot_id is None:
                raise RuntimeError("paged_slot_id must be allocated before syncing")
            slot_ids.append(int(sess.paged_slot_id))
            _, dirty = sess.get_paged_block_table_row_cpu_and_dirty(
                layer_idx=0,
                offset=0,
            )
            if dirty:
                dirty_idx.append(idx)

        if not dirty_idx:
            return

        dirty_slot_ids = [slot_ids[idx] for idx in dirty_idx]
        dirty_slot_ids_t = torch.tensor(
            dirty_slot_ids,
            device=self.device,
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

    @torch.no_grad()
    def prefill_chunk_sessions(
        self,
        *,
        sessions: list["InferenceSession"],
        chunk_token_ids: list[list[int]],
    ) -> torch.Tensor:
        if not sessions:
            raise ValueError("sessions must be non-empty")
        if len(sessions) != len(chunk_token_ids):
            raise ValueError("sessions/chunk_token_ids size mismatch")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("chunked prefill requires CUDA")
        if not self.use_paged_attention:
            raise RuntimeError("chunked prefill requires --paged-attn (paged KV cache)")
        if not self.use_amp or self.amp_dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError("chunked prefill requires fp16/bf16 AMP on CUDA")

        from torch.amp import autocast

        from rosellm.rosetrainer.paged_attention import PagedKVCache

        eng = self
        kvm = self.kv_manager
        num_layers = int(kvm.num_layers)
        block_size = int(kvm.block_size)
        max_ctx = int(self.config.max_position_embeddings)

        start_lens: list[int] = []
        new_lens: list[int] = []
        chunk_lens: list[int] = []

        for sess, chunk in zip(sessions, chunk_token_ids):
            n = int(len(chunk))
            if n <= 0:
                raise ValueError("chunk_token_ids entries must be non-empty")
            cur = int(sess.prompt_length)
            nxt = cur + n
            if nxt > max_ctx:
                raise ValueError(
                    f"chunked prefill exceeds max_position_embeddings ({nxt} > {max_ctx})"
                )
            start_lens.append(cur)
            new_lens.append(nxt)
            chunk_lens.append(n)

        for sess in sessions:
            if sess.paged_slot_id is None:
                sess.paged_slot_id = self._alloc_paged_slot()
                sess.clear_paged_block_table_cache()

        # Reserve KV metadata for the appended tokens (and clone on write when needed).
        for sess, n_append in zip(sessions, chunk_lens):
            for layer_idx in range(num_layers):
                kvm.reserve_append_tokens(
                    layer_idx=layer_idx,
                    block_ids=sess.block_ids_per_layer[layer_idx],
                    n_append=n_append,
                )

        # Update prompt lengths after KV reservation.
        for sess, nxt in zip(sessions, new_lens):
            sess.prompt_length = int(nxt)

        # Sync paged block tables so flashinfer_paged can gather kv_indices correctly.
        self._sync_paged_global_block_tables_for_sessions(sessions)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.eos_token_id

        max_chunk = int(max(chunk_lens))
        batch_ids: list[list[int]] = []
        masks: list[list[int]] = []
        pos_ids: list[list[int]] = []
        for start, chunk in zip(start_lens, chunk_token_ids):
            n = len(chunk)
            pad_len = max_chunk - n
            batch_ids.append(list(chunk) + [int(pad_id)] * pad_len)
            masks.append([1] * n + [0] * pad_len)
            pos_ids.append(list(range(int(start), int(start) + n)) + [0] * pad_len)

        input_ids = torch.tensor(
            batch_ids,
            device=self.device,
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            masks,
            device=self.device,
            dtype=torch.long,
        )
        position_ids = torch.tensor(
            pos_ids,
            device=self.device,
            dtype=torch.long,
        )

        slot_mapping = torch.tensor(
            [int(sess.paged_slot_id) for sess in sessions],
            device=self.device,
            dtype=torch.int32,
        )
        context_lens = torch.tensor(
            new_lens,
            device=self.device,
            dtype=torch.int32,
        )
        global_block_tables = self._get_paged_global_block_tables()
        block_tables = [
            global_block_tables[layer_idx] for layer_idx in range(num_layers)
        ]
        paged = PagedKVCache(
            k_cache=kvm._k_cache,
            v_cache=kvm._v_cache,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_size=block_size,
        )

        with record_function("roseinfer.prefill_chunk.model_forward"):
            if eng.use_amp:
                with autocast(
                    device_type=eng.device.type,
                    dtype=eng.amp_dtype,
                ):
                    logits, _ = eng.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None,
                        past_key_values=None,
                        use_cache=False,
                        position_ids=position_ids,
                        paged_kv_cache=paged,
                        attn_backend="flashinfer_paged",
                    )
            else:
                logits, _ = eng.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                    past_key_values=None,
                    use_cache=False,
                    position_ids=position_ids,
                    paged_kv_cache=paged,
                    attn_backend="flashinfer_paged",
                )

        # Return last logits for each sequence's last valid token in this chunk.
        bsz = int(input_ids.size(0))
        last_pos = torch.tensor(
            [int(n) - 1 for n in chunk_lens],
            device=self.device,
            dtype=torch.long,
        )
        row = torch.arange(bsz, device=self.device, dtype=torch.long)
        return logits[row, last_pos, :]

    def _top_k_logits(
        self,
        logits: torch.Tensor,  # [..., vocab]
        top_k: int,
    ) -> torch.Tensor:
        vocab = int(logits.size(-1))
        top_k = int(top_k)
        if top_k <= 0 or top_k >= vocab:
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
        logits_2d = logits.view(-1, int(logits.size(-1)))
        token_ids = self._sample_next_token_batch(
            logits=logits_2d,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )
        return int(token_ids[0].item())

    def _sample_next_token_batch(
        self,
        logits: torch.Tensor,  # [batch, vocab]
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
    ) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(
                f"logits must be 2D [B, V], got shape={tuple(logits.shape)}"
            )
        vocab = int(logits.size(-1))
        if (not do_sample) or temperature <= 0.0:
            return torch.argmax(logits, dim=-1)

        if (
            self.use_fused_sampler
            and flashinfer is not None
            and logits.is_cuda
            and self._sampling_generator is not None
        ):
            top_k = int(top_k)
            top_p = float(top_p)
            if top_p <= 0.0 or top_p >= 1.0:
                top_p = 1.0
            scaled = (
                logits if float(temperature) == 1.0 else logits / float(temperature)
            )
            sampled = flashinfer.sampling.top_k_top_p_sampling_from_logits(
                scaled,
                top_k=top_k,
                top_p=top_p,
                filter_apply_order="joint",
                deterministic=True,
                generator=self._sampling_generator,
            )
            # Guard against rare out-of-range ids (would crash embedding lookup).
            return sampled.clamp(min=0, max=max(0, vocab - 1))
        scaled = logits / float(temperature)
        top_k = int(top_k)

        if top_p <= 0.0 or top_p >= 1.0:
            if top_k <= 0 or top_k >= vocab:
                probs = torch.softmax(scaled, dim=-1).clamp_min(1e-9)
                return torch.multinomial(probs, num_samples=1).squeeze(-1)
            k = min(top_k, vocab)
            topk_logits, topk_idx = torch.topk(scaled, k, dim=-1)  # [B, K], [B, K]
            probs = torch.softmax(topk_logits, dim=-1).clamp_min(1e-9)
            choice = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
            return topk_idx.gather(-1, choice.unsqueeze(-1)).squeeze(-1)

        # top_p in (0, 1): sample from sorted logits (top-k or full) without
        # scattering back to vocab space.
        k = vocab if top_k <= 0 else min(top_k, vocab)
        sorted_logits, sorted_idx = torch.topk(scaled, k, dim=-1)  # [B, K], [B, K]
        probs = torch.softmax(sorted_logits, dim=-1)  # [B, K]
        cum_probs = torch.cumsum(probs, dim=-1)  # [B, K]
        mask = cum_probs > float(top_p)
        mask[..., 0] = False  # keep at least one token
        probs = probs.masked_fill(mask, 0.0).clamp_min(1e-9)
        choice = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
        return sorted_idx.gather(-1, choice.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _resolve_future_token_ids(
        input_ids: torch.Tensor,
        future_token_ids_map: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.dtype != torch.long:
            input_ids = input_ids.to(torch.long)
        if input_ids.numel() == 0:
            return input_ids
        max_idx = int(future_token_ids_map.numel() - 1)
        idx = torch.clamp(-input_ids, min=0, max=max_idx)
        return torch.where(input_ids < 0, future_token_ids_map[idx], input_ids)

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
            next_list = next_ids.tolist()
            for b in range(batch_size):
                token_id = int(next_list[b])
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
                next_list = next_ids.tolist()
                for b in range(batch_size):
                    token_id = int(next_list[b])
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

    def _get_or_create_paged_decode_cuda_graph(
        self,
        *,
        batch_size: int,
        global_block_tables: torch.Tensor,
    ) -> _PagedDecodeCudaGraph:
        if not self.use_cuda_graph:
            raise RuntimeError("use_cuda_graph is disabled")
        if self.device.type != "cuda":
            raise RuntimeError("CUDA graph requires CUDA device")
        if not self.use_paged_attention:
            raise RuntimeError("CUDA graph path only supports paged attention decode")

        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        global_ptr = int(global_block_tables.data_ptr())
        if self._paged_decode_cuda_graphs:
            any_graph = next(iter(self._paged_decode_cuda_graphs.values()))
            if any_graph.global_block_tables_ptr != global_ptr:
                self._paged_decode_cuda_graphs.clear()

        cached = self._paged_decode_cuda_graphs.get(batch_size)
        if cached is not None:
            return cached

        from torch.amp import autocast

        from rosellm.rosetrainer.paged_attention import PagedKVCache

        num_layers = int(self.kv_manager.num_layers)
        block_tables = [
            global_block_tables[layer_idx] for layer_idx in range(num_layers)
        ]

        input_ids = torch.zeros(
            (batch_size, 1),
            device=self.device,
            dtype=torch.long,
        )
        position_ids = torch.zeros_like(input_ids)
        slot_mapping = torch.zeros(
            (batch_size,),
            device=self.device,
            dtype=torch.int32,
        )
        context_lens = torch.zeros_like(slot_mapping)
        input_ids_host = torch.empty(
            (batch_size,),
            device="cpu",
            dtype=torch.long,
            pin_memory=True,
        )
        position_ids_host = torch.empty(
            (batch_size,),
            device="cpu",
            dtype=torch.long,
            pin_memory=True,
        )
        slot_mapping_host = torch.empty(
            (batch_size,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=True,
        )
        context_lens_host = torch.empty(
            (batch_size,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=True,
        )

        paged = PagedKVCache(
            k_cache=self.kv_manager._k_cache,
            v_cache=self.kv_manager._v_cache,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_size=int(self.kv_manager.block_size),
            write_kv=bool(self.use_fused_kv_append),
        )

        def run_model():
            return self.model(
                input_ids=input_ids,
                attention_mask=None,
                labels=None,
                past_key_values=None,
                use_cache=True,
                position_ids=position_ids,
                paged_kv_cache=paged,
                attn_backend=self.decode_attn_backend,
            )

        # Warm up: compile kernels / populate allocator cache before capture.
        for _ in range(3):
            if self.use_amp:
                with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    run_model()
            else:
                run_model()
        torch.cuda.synchronize(device=self.device)

        graph = torch.cuda.CUDAGraph()
        pool = self._cuda_graph_pool
        torch.cuda.synchronize(device=self.device)
        with torch.cuda.graph(graph, pool=pool):
            if self.use_amp:
                with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    logits, _, presents = run_model()
            else:
                logits, _, presents = run_model()

        if not isinstance(presents, list):
            raise TypeError("expected presents to be a list of (k, v) tuples")

        captured = _PagedDecodeCudaGraph(
            batch_size=batch_size,
            global_block_tables_ptr=global_ptr,
            graph=graph,
            input_ids=input_ids,
            position_ids=position_ids,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            input_ids_host=input_ids_host,
            position_ids_host=position_ids_host,
            slot_mapping_host=slot_mapping_host,
            context_lens_host=context_lens_host,
            input_ids_host_np=input_ids_host.numpy(),
            position_ids_host_np=position_ids_host.numpy(),
            slot_mapping_host_np=slot_mapping_host.numpy(),
            context_lens_host_np=context_lens_host.numpy(),
            logits=logits,
            presents=presents,
        )
        self._paged_decode_cuda_graphs[batch_size] = captured
        return captured

    @torch.no_grad()
    def decode_step_sessions(
        self,
        sessions: list["InferenceSession"],
        *,
        input_token_ids: Optional[list[int]] = None,
        position_ids: Optional[list[int]] = None,
        future_token_ids_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with record_function("roseinfer.decode_step_sessions.total"):
            assert sessions
            from torch.amp import autocast

            device = self.device
            batch_size = len(sessions)
            kvm = self.kv_manager

            if input_token_ids is None:
                last_ids: list[int] = []
                seq_lens: list[int] = []
                for sess in sessions:
                    if sess.finished:
                        continue
                    assert sess.generated_ids
                    last_ids.append(int(sess.generated_ids[-1]))
                    seq_len = int(sess.prompt_length + sess.step_count - 1)
                    seq_lens.append(seq_len)
                assert len(last_ids) == batch_size
            else:
                if position_ids is None:
                    raise ValueError(
                        "position_ids must be set when input_token_ids is set"
                    )
                if (
                    len(input_token_ids) != batch_size
                    or len(position_ids) != batch_size
                ):
                    raise ValueError("input_token_ids/position_ids batch size mismatch")
                last_ids = [int(x) for x in input_token_ids]
                seq_lens = [int(x) for x in position_ids]

            need_resolve = future_token_ids_map is not None and any(
                int(t) < 0 for t in last_ids
            )
            num_layers = kvm.num_layers
            num_heads = kvm.num_heads
            head_dim = kvm.head_dim
            if self.use_paged_attention:
                nvtx = device.type == "cuda" and os.environ.get("ROSEINFER_NVTX") == "1"
                block_size = kvm.block_size
                max_blocks_per_layer = kvm.max_blocks_per_layer
                fused_kv_append = bool(self.use_fused_kv_append)
                slot_ids: list[int] = []
                for sess in sessions:
                    if sess.paged_slot_id is None:
                        sess.paged_slot_id = self._alloc_paged_slot()
                        sess.clear_paged_block_table_cache()
                    assert sess.paged_slot_id is not None
                    slot_ids.append(sess.paged_slot_id)

                if fused_kv_append:
                    with _maybe_nvtx_range(
                        "roseinfer.kv.reserve_append_token", nvtx
                    ), record_function("roseinfer.kv.reserve_append_token"):
                        for layer_idx in range(num_layers):
                            for sess in sessions:
                                kvm.reserve_append_tokens(
                                    layer_idx=layer_idx,
                                    block_ids=sess.block_ids_per_layer[layer_idx],
                                    n_append=1,
                                )
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

                if self.use_cuda_graph:
                    graph = self._get_or_create_paged_decode_cuda_graph(
                        batch_size=batch_size,
                        global_block_tables=global_block_tables,
                    )
                    graph.input_ids_host_np[:] = last_ids
                    graph.position_ids_host_np[:] = seq_lens
                    graph.slot_mapping_host_np[:] = slot_ids
                    graph.context_lens_host_np[:] = seq_lens
                    graph.input_ids[:, 0].copy_(graph.input_ids_host, non_blocking=True)
                    graph.position_ids[:, 0].copy_(
                        graph.position_ids_host, non_blocking=True
                    )
                    graph.slot_mapping.copy_(graph.slot_mapping_host, non_blocking=True)
                    graph.context_lens.copy_(graph.context_lens_host, non_blocking=True)
                    if need_resolve:
                        assert future_token_ids_map is not None
                        ids = graph.input_ids
                        ids[:] = self._resolve_future_token_ids(
                            ids, future_token_ids_map
                        )
                    with _maybe_nvtx_range(
                        "roseinfer.model.forward.cuda_graph_replay", nvtx
                    ), record_function("roseinfer.model.forward.cuda_graph_replay"):
                        graph.graph.replay()
                    logits = graph.logits
                    presents = graph.presents
                else:
                    from rosellm.rosetrainer.paged_attention import PagedKVCache

                    lens = torch.tensor(seq_lens, device=device, dtype=torch.long)
                    input_ids = torch.tensor(  # [B, 1]
                        last_ids,
                        dtype=torch.long,
                        device=device,
                    ).view(batch_size, 1)
                    if need_resolve:
                        assert future_token_ids_map is not None
                        input_ids[:] = self._resolve_future_token_ids(
                            input_ids,
                            future_token_ids_map,
                        )
                    position_ids = lens.view(batch_size, 1)
                    slot_mapping = torch.tensor(
                        slot_ids,
                        device=device,
                        dtype=torch.int32,
                    )
                    block_tables = [
                        global_block_tables[layer_idx]
                        for layer_idx in range(num_layers)
                    ]
                    paged = PagedKVCache(
                        k_cache=kvm._k_cache,
                        v_cache=kvm._v_cache,
                        block_tables=block_tables,
                        slot_mapping=slot_mapping,
                        context_lens=lens.to(torch.int32),
                        block_size=block_size,
                        write_kv=fused_kv_append,
                    )
                    with _maybe_nvtx_range(
                        "roseinfer.model.forward", nvtx
                    ), record_function(
                        "roseinfer.model.forward",
                    ):
                        if self.use_amp:
                            with autocast(
                                device_type=device.type,
                                dtype=self.amp_dtype,
                            ):
                                logits, _, presents = self.model(
                                    input_ids=input_ids,
                                    attention_mask=None,
                                    labels=None,
                                    past_key_values=None,
                                    use_cache=True,
                                    position_ids=position_ids,
                                    paged_kv_cache=paged,
                                    attn_backend=self.decode_attn_backend,
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
                                attn_backend=self.decode_attn_backend,
                            )
                last_logits = logits[:, -1, :]  # [B, V]
                if not fused_kv_append:
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

            lens = torch.tensor(seq_lens, device=device, dtype=torch.long)
            input_ids = torch.tensor(  # [B, 1]
                last_ids,
                dtype=torch.long,
                device=device,
            ).view(batch_size, 1)
            if need_resolve:
                assert future_token_ids_map is not None
                input_ids[:] = self._resolve_future_token_ids(
                    input_ids, future_token_ids_map
                )
            position_ids = lens.view(batch_size, 1)
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
                            attn_backend=self.decode_attn_backend,
                        )
                else:
                    logits, _, presents = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None,
                        past_key_values=tuple(batched_past),
                        use_cache=True,
                        position_ids=position_ids,
                        attn_backend=self.decode_attn_backend,
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
        self.committed_step_count: int = 0
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
        return base_ids + [int(t) for t in self.generated_ids if int(t) >= 0]

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
                    attn_backend=eng.prefill_attn_backend,
                )
        else:
            logits, _, presents = eng.model(
                input_ids=input_ids,
                attention_mask=None,
                labels=None,
                past_key_values=None,
                use_cache=True,
                attn_backend=eng.prefill_attn_backend,
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
                    attn_backend=eng.prefill_attn_backend,
                )
        else:
            logits, _, presents = eng.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                past_key_values=None,
                use_cache=True,
                attn_backend=eng.prefill_attn_backend,
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
                    attn_backend=eng.decode_attn_backend,
                )
        else:
            logits, _, presents = eng.model(
                input_ids=input_ids,
                attention_mask=None,
                labels=None,
                past_key_values=self.kv_cache,
                use_cache=True,
                attn_backend=eng.decode_attn_backend,
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
        return self.apply_token_id(int(next_token))

    def apply_token_id(self, token_id: int) -> int | None:
        if self.finished:
            return None
        eng = self.engine
        token_id = int(token_id)
        self.generated_ids.append(token_id)
        self.step_count += 1
        self.committed_step_count += 1
        if self.stop_on_eos:
            eos_id = eng.eos_token_id
            if eos_id is not None and token_id == eos_id:
                self.finished = True
        if self.max_new_tokens > 0 and self.committed_step_count >= self.max_new_tokens:
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
        return self.apply_token_id(int(next_token))

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
                    attn_backend=eng.decode_attn_backend,
                )
        else:
            logits, _, presents = eng.model(
                input_ids=input_ids,
                attention_mask=None,
                labels=None,
                past_key_values=self.kv_cache,
                use_cache=True,
                attn_backend=eng.decode_attn_backend,
            )
        self.kv_cache = presents
        next_logits = logits[:, -1, :]  # [B, V]
        return next_logits  # [B, V]


class PrefixCacheEntry:
    def __init__(
        self,
        key: PrefixCacheKey,
        prompt_length: int,
        blocks_ids_per_layer: list[list[int]],
        last_logits: torch.Tensor,
    ) -> None:
        self.key = key
        self.prompt_length = int(prompt_length)
        self.blocks_ids_per_layer = [list(ids) for ids in blocks_ids_per_layer]
        self.last_logits = last_logits.detach().clone()


class _TokenTrieNode:
    __slots__ = ("children", "entry", "count")

    def __init__(self) -> None:
        self.children: dict[int, "_TokenTrieNode"] = {}
        self.entry: PrefixCacheEntry | None = None
        self.count: int = 0


class _TokenTrie:
    def __init__(self) -> None:
        self.root = _TokenTrieNode()

    def insert(
        self,
        key: tuple[int, ...],
        entry: PrefixCacheEntry,
    ) -> None:
        node = self.root
        node.count += 1
        for tok in key:
            nxt = node.children.get(tok)
            if nxt is None:
                nxt = _TokenTrieNode()
                node.children[tok] = nxt
            node = nxt
            node.count += 1
        node.entry = entry

    def remove(
        self,
        key: tuple[int, ...],
        entry: PrefixCacheEntry | None = None,
    ) -> None:
        node = self.root
        stack: list[tuple[_TokenTrieNode, int, _TokenTrieNode]] = []
        for tok in key:
            nxt = node.children.get(tok)
            if nxt is None:
                return
            stack.append((node, tok, nxt))
            node = nxt

        if node.entry is None:
            return
        if entry is not None and node.entry is not entry:
            return

        node.entry = None
        self.root.count -= 1
        for _, _, child in stack:
            child.count -= 1

        for parent, tok, child in reversed(stack):
            if child.count > 0:
                break
            parent.children.pop(tok, None)

    def longest_prefix(
        self,
        key: tuple[int, ...],
    ) -> PrefixCacheEntry | None:
        node = self.root
        best: PrefixCacheEntry | None = None
        for tok in key:
            nxt = node.children.get(tok)
            if nxt is None:
                break
            node = nxt
            if node.entry is not None:
                best = node.entry
        return best


class PrefixCache:
    def __init__(
        self,
        kv_manager: "KVBlockManager",
        max_entries: int = 256,
    ) -> None:
        self.kv_manager = kv_manager
        self.max_entries = max(0, int(max_entries))
        self._entries: OrderedDict[PrefixCacheKey, PrefixCacheEntry] = OrderedDict()
        self._token_trie = _TokenTrie()

    def _release_entry(self, entry: PrefixCacheEntry) -> None:
        for layer_idx, block_ids in enumerate(entry.blocks_ids_per_layer):
            if block_ids:
                self.kv_manager.free_blocks(layer_idx, block_ids)

    def _evict_one(self) -> None:
        if not self._entries:
            return
        _, entry = self._entries.popitem(last=False)
        if isinstance(entry.key, tuple):
            self._token_trie.remove(entry.key, entry)
        self._release_entry(entry)

    def clear(self) -> None:
        while self._entries:
            _, entry = self._entries.popitem(last=False)
            if isinstance(entry.key, tuple):
                self._token_trie.remove(entry.key, entry)
            self._release_entry(entry)

    def get(self, key: PrefixCacheKey) -> PrefixCacheEntry | None:
        return self._entries.get(key)

    def find_longest_token_prefix(
        self,
        key: tuple[int, ...],
    ) -> PrefixCacheEntry | None:
        key_len = len(key)
        if key_len <= 0:
            return None
        best_entry = self._token_trie.longest_prefix(key)
        if best_entry is None or int(best_entry.prompt_length) >= key_len:
            return None
        self._entries.move_to_end(best_entry.key)
        return best_entry

    def put(
        self,
        key: PrefixCacheKey,
        session: "InferenceSession",
        last_logits: torch.Tensor,
    ) -> None:
        if key in self._entries:
            self._entries.move_to_end(key)
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
            key=key,
            prompt_length=prompt_length,
            blocks_ids_per_layer=block_ids_per_layer,
            last_logits=last_logits,
        )
        while self.max_entries > 0 and len(self._entries) >= self.max_entries:
            self._evict_one()
        self._entries[key] = entry
        if isinstance(key, tuple):
            self._token_trie.insert(key, entry)
        self._entries.move_to_end(key)

    def attach(
        self,
        key: PrefixCacheKey,
        session: "InferenceSession",
    ) -> torch.Tensor | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
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

        groups: dict[tuple[float, int, float, bool], list[int]] = {}
        for i, (_, sess) in enumerate(active_pairs):
            key = (
                float(sess.temperature),
                int(sess.top_k),
                float(sess.top_p),
                bool(sess.do_sample),
            )
            groups.setdefault(key, []).append(i)

        next_token_ids: list[int] = [0 for _ in range(len(active_pairs))]
        for (temp, top_k, top_p, do_sample), idxs in groups.items():
            if len(idxs) == len(active_pairs) and idxs == list(
                range(len(active_pairs))
            ):
                next_ids = self.engine._sample_next_token_batch(
                    logits=last_logits,
                    temperature=temp,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                )
                next_token_ids = [int(x) for x in next_ids.tolist()]
                break
            idx_t = torch.tensor(
                idxs,
                device=self.engine.device,
                dtype=torch.long,
            )
            logits_g = last_logits.index_select(0, idx_t)
            next_ids = self.engine._sample_next_token_batch(
                logits=logits_g,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            next_list = next_ids.tolist()
            for pos, i in enumerate(idxs):
                next_token_ids[i] = int(next_list[pos])

        for i, (rid, sess) in enumerate(active_pairs):
            token_id = sess.apply_token_id(next_token_ids[i])
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


@dataclass(slots=True)
class _OverlapDecodeBatch:
    rids: list[int]
    sessions: list[InferenceSession]
    placeholder_positions: list[int]
    next_token_ids_cpu: torch.Tensor
    copy_done: "torch.cuda.Event"


@dataclass(slots=True)
class _OverlapPrefillBatch:
    all_rids: list[int]
    all_sessions: list[InferenceSession]
    ready_rids: list[int]
    ready_sessions: list[InferenceSession]
    next_token_ids_cpu: torch.Tensor
    copy_done: "torch.cuda.Event"


class OnlineScheduler:
    def __init__(
        self,
        engine: "InferenceEngine",
        max_batch_size: int = 8,
        use_prefix_cache: bool = True,
        overlap_schedule: bool = True,
    ) -> None:
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.use_prefix_cache = use_prefix_cache
        self._sessions: dict[int, InferenceSession] = {}
        self._next_request_id: int = 0
        self._active_rids: deque[int] = deque()
        self._finished_ids: list[int] = []

        self._overlap_schedule = bool(overlap_schedule)
        self._overlap_schedule = (
            self._overlap_schedule
            and engine.device.type == "cuda"
            and torch.cuda.is_available()
        )
        self._overlap_future_token_ids_ct: int = 0
        self._overlap_future_token_ids_limit: int = 0
        self._overlap_future_token_ids_map: torch.Tensor | None = None
        self._overlap_pending: deque[_OverlapDecodeBatch] = deque()
        self._overlap_inflight: dict[int, int] = {}
        self._overlap_canceled: set[int] = set()
        if self._overlap_schedule:
            max_running = int(getattr(engine, "kv_cache_max_concurrency", 0) or 0)
            max_running = max(1, max_running)
            self._overlap_future_token_ids_limit = max_running * 3
            # Leave headroom so (ct + batch_size) slices never exceed the map.
            map_size = max_running * 5 + int(self.max_batch_size) + 8
            # Use zeros to avoid undefined reads if a placeholder index is ever
            # referenced before being written (should be rare, but crashing is worse).
            self._overlap_future_token_ids_map = torch.zeros(
                (map_size,),
                dtype=torch.long,
                device=engine.device,
            )

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
        cache_keys: list[PrefixCacheKey] = []
        last_logits_per_req: list[torch.Tensor | None] = [None for _ in requests]
        prefix_suffix_ids: dict[int, list[int]] = {}
        miss_idx: list[int] = []
        dup_of: dict[int, int] = {}
        first_idx_for_key: dict[PrefixCacheKey, int] = {}

        for i, req in enumerate(requests):
            rid = alloc_rid(req.request_id)
            rids.append(rid)

            if req.prompt_token_ids is None:
                ids = eng.tokenizer.encode(
                    req.prompt,
                    add_special_tokens=False,
                )
            else:
                ids = (
                    req.prompt_token_ids
                    if isinstance(req.prompt_token_ids, list)
                    else list(req.prompt_token_ids)
                )
            if not ids:
                ids = [eng.eos_token_id]
            max_pos = int(eng.config.max_position_embeddings)
            if len(ids) > max_pos:
                ids = ids[-max_pos:]
            token_ids_list.append(ids)
            cache_key: PrefixCacheKey = (
                req.prompt if req.prompt_token_ids is None else tuple(ids)
            )
            cache_keys.append(cache_key)

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
                src = first_idx_for_key.get(cache_key)
                if src is not None:
                    dup_of[i] = src
                    continue
                first_idx_for_key[cache_key] = i

                cached_logits = eng.prefix_cache.attach(cache_key, sess)
                if cached_logits is not None:
                    last_logits_per_req[i] = cached_logits
                    continue
                if eng.use_paged_attention and isinstance(cache_key, tuple):
                    prefix = eng.prefix_cache.find_longest_token_prefix(cache_key)
                    if prefix is not None and prefix.prompt_length < len(ids):
                        eng.prefix_cache.attach(prefix.key, sess)
                        prefix_suffix_ids[i] = ids[prefix.prompt_length :]
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
                        cache_keys[idx],
                        sessions[idx],
                        logits,
                    )

        if prefix_suffix_ids:
            suffix_pos: dict[int, int] = {idx: 0 for idx in prefix_suffix_ids}
            while True:
                active = [
                    idx
                    for idx, pos in suffix_pos.items()
                    if pos < len(prefix_suffix_ids[idx])
                ]
                if not active:
                    break
                active_sessions = [sessions[idx] for idx in active]
                for idx, sess in zip(active, active_sessions):
                    tok = int(prefix_suffix_ids[idx][suffix_pos[idx]])
                    sess.generated_ids = [tok]
                    sess.step_count = 1
                    sess.committed_step_count = 1
                step_logits = eng.decode_step_sessions(active_sessions)
                for b, idx in enumerate(active):
                    logits = step_logits[b : b + 1]
                    last_logits_per_req[idx] = logits
                    sessions[idx].prompt_length += 1
                    suffix_pos[idx] += 1

            for idx in prefix_suffix_ids:
                sess = sessions[idx]
                sess.generated_ids = []
                sess.step_count = 0
                sess.committed_step_count = 0
                logits = last_logits_per_req[idx]
                if logits is None:
                    raise RuntimeError(
                        f"missing prefix reuse logits for request {rids[idx]}"
                    )
                if self.use_prefix_cache:
                    eng.prefix_cache.put(
                        cache_keys[idx],
                        sess,
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
            sess.committed_step_count = 1
            if sess.stop_on_eos:
                eos_id = eng.eos_token_id
                if eos_id is not None and int(token_id) == eos_id:
                    sess.finished = True
            if (
                sess.max_new_tokens > 0
                and sess.committed_step_count >= sess.max_new_tokens
            ):
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
        if self._overlap_schedule and self._overlap_pending:
            return True
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
        return [int(t) for t in session.generated_ids if int(t) >= 0]

    def get_step_count(self, request_id: int) -> int:
        session = self._sessions.get(request_id)
        if session is None:
            return 0
        return int(session.committed_step_count)

    @torch.no_grad()
    def step(self) -> dict[int, int]:
        if self._overlap_schedule:
            with record_function("roseinfer.scheduler.step.overlap"):
                return self._step_overlap()
        with record_function("roseinfer.scheduler.step.sync"):
            return self._step_sync()

    @torch.no_grad()
    def _step_sync(self) -> dict[int, int]:
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

        groups: dict[tuple[float, int, float, bool], list[int]] = {}
        for i, (_, sess) in enumerate(selected_pairs):
            key = (
                float(sess.temperature),
                int(sess.top_k),
                float(sess.top_p),
                bool(sess.do_sample),
            )
            groups.setdefault(key, []).append(i)

        next_token_ids: list[int] = [0 for _ in range(len(selected_pairs))]
        for (temp, top_k, top_p, do_sample), idxs in groups.items():
            if len(idxs) == len(selected_pairs) and idxs == list(
                range(len(selected_pairs))
            ):
                next_ids = self.engine._sample_next_token_batch(
                    logits=last_logits,
                    temperature=temp,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                )
                next_token_ids = [int(x) for x in next_ids.tolist()]
                break
            idx_t = torch.tensor(
                idxs,
                device=self.engine.device,
                dtype=torch.long,
            )
            logits_g = last_logits.index_select(0, idx_t)
            next_ids = self.engine._sample_next_token_batch(
                logits=logits_g,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            next_list = next_ids.tolist()
            for pos, i in enumerate(idxs):
                next_token_ids[i] = int(next_list[pos])

        for i, (rid, sess) in enumerate(selected_pairs):
            token_id = sess.apply_token_id(next_token_ids[i])
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

    @torch.no_grad()
    def _step_overlap(self) -> dict[int, int]:
        eng = self.engine
        nvtx = eng.device.type == "cuda" and os.environ.get("ROSEINFER_NVTX") == "1"
        step_tokens: dict[int, int] = {}
        just_finished: list[int] = []

        # Schedule a new decode batch first.
        selected_pairs: list[tuple[int, InferenceSession]] = []
        with _maybe_nvtx_range(
            "roseinfer.scheduler.overlap.schedule", nvtx
        ), record_function("roseinfer.scheduler.overlap.schedule"):
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
                if not sess.generated_ids:
                    continue
                if sess.max_new_tokens > 0 and int(sess.step_count) >= int(
                    sess.max_new_tokens
                ):
                    continue
                selected_pairs.append((rid, sess))

            if selected_pairs:
                # Keep round-robin ordering even when overlap is enabled.
                for rid, _ in selected_pairs:
                    self._active_rids.append(rid)

                rids = [rid for rid, _ in selected_pairs]
                sessions = [sess for _, sess in selected_pairs]
                input_ids = [int(sess.generated_ids[-1]) for sess in sessions]
                pos_ids = [
                    int(sess.prompt_length + sess.step_count - 1) for sess in sessions
                ]
                future_map = self._overlap_future_token_ids_map
                if future_map is None:
                    raise RuntimeError("overlap future token map is not initialized")

                last_logits = eng.decode_step_sessions(
                    sessions,
                    input_token_ids=input_ids,
                    position_ids=pos_ids,
                    future_token_ids_map=future_map,
                )

                # Sample next token IDs on device without syncing to CPU.
                groups: dict[tuple[float, int, float, bool], list[int]] = {}
                for i, sess in enumerate(sessions):
                    key = (
                        float(sess.temperature),
                        int(sess.top_k),
                        float(sess.top_p),
                        bool(sess.do_sample),
                    )
                    groups.setdefault(key, []).append(i)

                next_ids = torch.empty(
                    (len(sessions),),
                    device=eng.device,
                    dtype=torch.long,
                )
                for (temp, top_k, top_p, do_sample), idxs in groups.items():
                    if len(idxs) == len(sessions) and idxs == list(
                        range(len(sessions))
                    ):
                        next_ids = eng._sample_next_token_batch(
                            logits=last_logits,
                            temperature=temp,
                            top_k=top_k,
                            top_p=top_p,
                            do_sample=do_sample,
                        )
                        break
                    idx_t = torch.tensor(
                        idxs,
                        device=eng.device,
                        dtype=torch.long,
                    )
                    logits_g = last_logits.index_select(0, idx_t)
                    sampled = eng._sample_next_token_batch(
                        logits=logits_g,
                        temperature=temp,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=do_sample,
                    )
                    next_ids.index_copy_(0, idx_t, sampled)

                # Allocate placeholders for the next step and update the future map.
                bs = int(len(sessions))
                ct = int(self._overlap_future_token_ids_ct)
                limit = int(self._overlap_future_token_ids_limit)
                placeholder_positions: list[int] = []
                for i, sess in enumerate(sessions):
                    placeholder_positions.append(len(sess.generated_ids))
                    sess.generated_ids.append(-(ct + i + 1))
                    sess.step_count += 1

                future_map[ct + 1 : ct + 1 + bs] = next_ids
                self._overlap_future_token_ids_ct = (ct + bs) % max(1, limit)

                # Async copy results to CPU and record an event for later synchronization.
                next_ids_cpu = next_ids.to("cpu", non_blocking=True)
                copy_done = torch.cuda.Event()
                copy_done.record()
                self._overlap_pending.append(
                    _OverlapDecodeBatch(
                        rids=rids,
                        sessions=sessions,
                        placeholder_positions=placeholder_positions,
                        next_token_ids_cpu=next_ids_cpu,
                        copy_done=copy_done,
                    )
                )
                for rid in rids:
                    self._overlap_inflight[rid] = self._overlap_inflight.get(rid, 0) + 1

        # Process the oldest pending decode result (one-step delayed output).
        with _maybe_nvtx_range(
            "roseinfer.scheduler.overlap.process_pending", nvtx
        ), record_function("roseinfer.scheduler.overlap.process_pending"):
            if self._overlap_pending and (
                len(self._overlap_pending) > 1 or not selected_pairs
            ):
                item = self._overlap_pending.popleft()
                item.copy_done.synchronize()
                token_ids = [int(x) for x in item.next_token_ids_cpu.tolist()]
                for rid, sess, pos, token_id in zip(
                    item.rids,
                    item.sessions,
                    item.placeholder_positions,
                    token_ids,
                ):
                    inflight = self._overlap_inflight.get(rid, 0) - 1
                    if inflight <= 0:
                        self._overlap_inflight.pop(rid, None)
                    else:
                        self._overlap_inflight[rid] = inflight

                    if rid in self._overlap_canceled:
                        if pos < 0 or pos >= len(sess.generated_ids):
                            raise RuntimeError(
                                "overlap placeholder position out of range"
                            )
                        if int(sess.generated_ids[pos]) < 0:
                            sess.generated_ids[pos] = int(token_id)
                        if inflight <= 0:
                            self._overlap_canceled.discard(rid)
                            sess.release_kv_blocks()
                            self._sessions.pop(rid, None)
                        continue

                    if sess.finished:
                        if 0 <= pos < len(sess.generated_ids):
                            sess.generated_ids.pop(pos)
                            sess.step_count -= 1
                        if inflight <= 0:
                            just_finished.append(rid)
                            sess.release_kv_blocks()
                        continue

                    if pos < 0 or pos >= len(sess.generated_ids):
                        raise RuntimeError("overlap placeholder position out of range")
                    if int(sess.generated_ids[pos]) < 0:
                        sess.generated_ids[pos] = int(token_id)
                    else:
                        raise RuntimeError("overlap placeholder already resolved")
                    sess.committed_step_count += 1

                    if sess.stop_on_eos and eng.eos_token_id is not None:
                        if int(token_id) == int(eng.eos_token_id):
                            sess.finished = True
                    if (
                        sess.max_new_tokens > 0
                        and sess.committed_step_count >= sess.max_new_tokens
                    ):
                        sess.finished = True

                    step_tokens[rid] = int(token_id)

                    if sess.finished and inflight <= 0:
                        just_finished.append(rid)
                        sess.release_kv_blocks()

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
        rid = int(request_id)
        session = self._sessions.get(rid)
        if session is None:
            return
        if self._overlap_schedule and self._overlap_inflight.get(rid, 0) > 0:
            session.finished = True
            self._overlap_canceled.add(rid)
            return
        session = self._sessions.pop(rid, None)
        if session is not None:
            session.release_kv_blocks()


class ChunkedOnlineScheduler:
    def __init__(
        self,
        engine: "InferenceEngine",
        *,
        max_batch_size: int = 8,
        prefill_chunk_size: int = 256,
        prefill_max_batch_size: Optional[int] = None,
        use_prefix_cache: bool = True,
        overlap_schedule: bool = True,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        prefill_chunk_size = int(prefill_chunk_size)
        if prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        if prefill_max_batch_size is None:
            prefill_max_batch_size = max_batch_size
        if int(prefill_max_batch_size) <= 0:
            raise ValueError("prefill_max_batch_size must be positive")
        if not engine.use_paged_attention:
            raise ValueError(
                "ChunkedOnlineScheduler requires engine.use_paged_attention"
            )
        self.engine = engine
        self.max_batch_size = int(max_batch_size)
        self.prefill_chunk_size = prefill_chunk_size
        self.prefill_max_batch_size = int(prefill_max_batch_size)
        self.use_prefix_cache = bool(use_prefix_cache)
        self._sessions: dict[int, InferenceSession] = {}
        self._prompt_token_ids: dict[int, list[int]] = {}
        self._cache_keys: dict[int, PrefixCacheKey] = {}
        self._prefill_rids: deque[int] = deque()
        self._decode_rids: deque[int] = deque()
        self._finished_ids: list[int] = []
        self._next_request_id: int = 0

        self._overlap_schedule = bool(overlap_schedule)
        self._overlap_schedule = (
            self._overlap_schedule
            and engine.device.type == "cuda"
            and torch.cuda.is_available()
        )
        self._overlap_future_token_ids_ct: int = 0
        self._overlap_future_token_ids_limit: int = 0
        self._overlap_future_token_ids_map: torch.Tensor | None = None
        self._overlap_decode_pending: deque[_OverlapDecodeBatch] = deque()
        self._overlap_prefill_pending: deque[_OverlapPrefillBatch] = deque()
        self._overlap_inflight: dict[int, int] = {}
        self._overlap_canceled: set[int] = set()
        if self._overlap_schedule:
            max_running = int(getattr(engine, "kv_cache_max_concurrency", 0) or 0)
            max_running = max(1, max_running)
            self._overlap_future_token_ids_limit = max_running * 3
            map_size = max_running * 5 + int(self.max_batch_size) + 8
            # Use zeros to avoid undefined reads if a placeholder index is ever
            # referenced before being written (should be rare, but crashing is worse).
            self._overlap_future_token_ids_map = torch.zeros(
                (map_size,),
                dtype=torch.long,
                device=engine.device,
            )

    def pop_finished_ids(self) -> list[int]:
        ids, self._finished_ids = self._finished_ids, []
        return ids

    def has_unfinished(self) -> bool:
        if self._overlap_schedule and (
            self._overlap_decode_pending or self._overlap_prefill_pending
        ):
            return True
        while self._decode_rids:
            rid = self._decode_rids[0]
            sess = self._sessions.get(rid)
            if sess is None or sess.finished:
                self._decode_rids.popleft()
                continue
            return True
        while self._prefill_rids:
            rid = self._prefill_rids[0]
            sess = self._sessions.get(rid)
            if sess is None or sess.finished:
                self._prefill_rids.popleft()
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
        return [int(t) for t in session.generated_ids if int(t) >= 0]

    def get_step_count(self, request_id: int) -> int:
        session = self._sessions.get(request_id)
        if session is None:
            return 0
        return int(session.committed_step_count)

    def get_response(self, request_id: int) -> str:
        session = self._sessions[request_id]
        return session.decode_text()

    def pop_response(self, request_id: int) -> str:
        session = self._sessions.pop(request_id)
        session.release_kv_blocks()
        self._prompt_token_ids.pop(request_id, None)
        self._cache_keys.pop(request_id, None)
        return session.decode_text()

    def discard_request(self, request_id: int) -> None:
        rid = int(request_id)
        session = self._sessions.get(rid)
        if session is None:
            self._prompt_token_ids.pop(rid, None)
            self._cache_keys.pop(rid, None)
            return
        if self._overlap_schedule and self._overlap_inflight.get(rid, 0) > 0:
            session.finished = True
            self._overlap_canceled.add(rid)
            self._prompt_token_ids.pop(rid, None)
            self._cache_keys.pop(rid, None)
            return
        session = self._sessions.pop(rid, None)
        if session is not None:
            session.release_kv_blocks()
        self._prompt_token_ids.pop(rid, None)
        self._cache_keys.pop(rid, None)

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
        req = OnlineRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_on_eos=stop_on_eos,
            do_sample=do_sample,
            prompt_token_ids=prompt_token_ids,
            request_id=request_id,
        )
        return self.add_requests([req])[0]

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

        for req in requests:
            rid = alloc_rid(req.request_id)
            rids.append(rid)

            if req.prompt_token_ids is None:
                ids = eng.tokenizer.encode(req.prompt, add_special_tokens=False)
                cache_key: PrefixCacheKey = req.prompt
            else:
                ids = (
                    req.prompt_token_ids
                    if isinstance(req.prompt_token_ids, list)
                    else list(req.prompt_token_ids)
                )
                cache_key = tuple(ids)
            if not ids:
                ids = [eng.eos_token_id]
                cache_key = tuple(ids)
            max_pos = int(eng.config.max_position_embeddings)
            if len(ids) > max_pos:
                ids = ids[-max_pos:]
                cache_key = tuple(ids)

            sess = InferenceSession(eng)
            sess.input_ids = eng._encode_prompt_token_ids(ids)

            max_new = int(req.max_new_tokens)
            if max_new > 0:
                available = max_pos - len(ids)
                if available <= 0:
                    sess.finished = True
                elif max_new > available:
                    max_new = available
            sess.set_generation_config(
                max_new_tokens=max_new,
                temperature=float(req.temperature),
                top_k=int(req.top_k),
                top_p=float(req.top_p),
                do_sample=bool(req.do_sample),
                stop_on_eos=bool(req.stop_on_eos),
            )

            self._sessions[rid] = sess
            self._prompt_token_ids[rid] = ids
            self._cache_keys[rid] = cache_key

            cached_logits: torch.Tensor | None = None
            if self.use_prefix_cache:
                cached_logits = eng.prefix_cache.attach(cache_key, sess)

            if cached_logits is not None:
                token_id = eng._sample_next_token(
                    logits=cached_logits,
                    temperature=sess.temperature,
                    top_k=sess.top_k,
                    top_p=sess.top_p,
                    do_sample=sess.do_sample,
                )
                sess.generated_ids.append(int(token_id))
                sess.step_count = 1
                sess.committed_step_count = 1
                if (
                    sess.stop_on_eos
                    and eng.eos_token_id is not None
                    and int(token_id) == int(eng.eos_token_id)
                ):
                    sess.finished = True
                if (
                    sess.max_new_tokens > 0
                    and sess.committed_step_count >= sess.max_new_tokens
                ):
                    sess.finished = True
                if sess.finished:
                    sess.release_kv_blocks()
                    self._finished_ids.append(rid)
                else:
                    self._decode_rids.append(rid)
                continue

            if sess.finished:
                sess.release_kv_blocks()
                self._finished_ids.append(rid)
                continue

            if self.use_prefix_cache and isinstance(cache_key, tuple):
                entry = eng.prefix_cache.find_longest_token_prefix(cache_key)
                if entry is not None:
                    eng.prefix_cache.attach(entry.key, sess)

            if int(sess.prompt_length) >= len(ids):
                raise RuntimeError("prefix cache attach produced invalid prompt_length")
            self._prefill_rids.append(rid)

        self._next_request_id = next_rid
        return rids

    def _select_decode_batch(self) -> list[tuple[int, InferenceSession]]:
        selected: list[tuple[int, InferenceSession]] = []
        max_examine = len(self._decode_rids)
        while (
            len(selected) < self.max_batch_size
            and self._decode_rids
            and max_examine > 0
        ):
            max_examine -= 1
            rid = self._decode_rids.popleft()
            sess = self._sessions.get(rid)
            if sess is None or sess.finished:
                continue
            if not sess.generated_ids:
                continue
            selected.append((rid, sess))
        return selected

    def _select_prefill_batch(self) -> list[tuple[int, InferenceSession, list[int]]]:
        selected: list[tuple[int, InferenceSession, list[int]]] = []
        max_examine = len(self._prefill_rids)
        while (
            len(selected) < self.prefill_max_batch_size
            and self._prefill_rids
            and max_examine > 0
        ):
            max_examine -= 1
            rid = self._prefill_rids.popleft()
            sess = self._sessions.get(rid)
            if sess is None or sess.finished:
                continue
            prompt_ids = self._prompt_token_ids.get(rid)
            if not prompt_ids:
                continue
            cur = int(sess.prompt_length)
            if cur >= len(prompt_ids):
                continue
            remaining = len(prompt_ids) - cur
            take = min(int(self.prefill_chunk_size), int(remaining))
            chunk = prompt_ids[cur : cur + take]
            if not chunk:
                continue
            selected.append((rid, sess, chunk))
        return selected

    @torch.no_grad()
    def step(self) -> dict[int, int]:
        if self._overlap_schedule:
            return self._step_overlap()
        eng = self.engine
        step_tokens: dict[int, int] = {}
        just_finished: list[int] = []

        decode_pairs = self._select_decode_batch()
        if decode_pairs:
            sessions = [sess for _, sess in decode_pairs]
            last_logits = eng.decode_step_sessions(sessions)

            groups: dict[tuple[float, int, float, bool], list[int]] = {}
            for i, (_, sess) in enumerate(decode_pairs):
                key = (
                    float(sess.temperature),
                    int(sess.top_k),
                    float(sess.top_p),
                    bool(sess.do_sample),
                )
                groups.setdefault(key, []).append(i)

            next_token_ids: list[int] = [0 for _ in range(len(decode_pairs))]
            for (temp, top_k, top_p, do_sample), idxs in groups.items():
                if len(idxs) == len(decode_pairs) and idxs == list(
                    range(len(decode_pairs))
                ):
                    next_ids = eng._sample_next_token_batch(
                        logits=last_logits,
                        temperature=temp,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=do_sample,
                    )
                    next_token_ids = [int(x) for x in next_ids.tolist()]
                    break
                idx_t = torch.tensor(
                    idxs,
                    device=eng.device,
                    dtype=torch.long,
                )
                logits_g = last_logits.index_select(0, idx_t)
                next_ids = eng._sample_next_token_batch(
                    logits=logits_g,
                    temperature=temp,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                )
                next_list = next_ids.tolist()
                for pos, i in enumerate(idxs):
                    next_token_ids[i] = int(next_list[pos])

            for i, (rid, sess) in enumerate(decode_pairs):
                token_id = sess.apply_token_id(next_token_ids[i])
                if token_id is not None:
                    step_tokens[rid] = int(token_id)
                if sess.finished:
                    just_finished.append(rid)
                    sess.release_kv_blocks()
                else:
                    self._decode_rids.append(rid)

        prefill_batch = self._select_prefill_batch()
        if prefill_batch:
            sessions = [sess for _, sess, _ in prefill_batch]
            chunks = [chunk for _, _, chunk in prefill_batch]
            last_logits = eng.prefill_chunk_sessions(
                sessions=sessions,
                chunk_token_ids=chunks,
            )

            ready_pairs: list[tuple[int, int, InferenceSession]] = []
            for b, (rid, sess, _) in enumerate(prefill_batch):
                prompt_ids = self._prompt_token_ids.get(rid) or []
                if int(sess.prompt_length) < len(prompt_ids):
                    self._prefill_rids.append(rid)
                    continue
                ready_pairs.append((b, rid, sess))

            if ready_pairs:
                groups: dict[tuple[float, int, float, bool], list[int]] = {}
                for i, (_, _, sess) in enumerate(ready_pairs):
                    key = (
                        float(sess.temperature),
                        int(sess.top_k),
                        float(sess.top_p),
                        bool(sess.do_sample),
                    )
                    groups.setdefault(key, []).append(i)

                next_token_ids: list[int] = [0 for _ in range(len(ready_pairs))]
                logits_ready = last_logits.index_select(
                    0,
                    torch.tensor(
                        [b for b, _, _ in ready_pairs],
                        device=eng.device,
                        dtype=torch.long,
                    ),
                )
                for (temp, top_k, top_p, do_sample), idxs in groups.items():
                    if len(idxs) == len(ready_pairs) and idxs == list(
                        range(len(ready_pairs))
                    ):
                        next_ids = eng._sample_next_token_batch(
                            logits=logits_ready,
                            temperature=temp,
                            top_k=top_k,
                            top_p=top_p,
                            do_sample=do_sample,
                        )
                        next_token_ids = [int(x) for x in next_ids.tolist()]
                        break
                    idx_t = torch.tensor(
                        idxs,
                        device=eng.device,
                        dtype=torch.long,
                    )
                    logits_g = logits_ready.index_select(0, idx_t)
                    next_ids = eng._sample_next_token_batch(
                        logits=logits_g,
                        temperature=temp,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=do_sample,
                    )
                    next_list = next_ids.tolist()
                    for pos, i in enumerate(idxs):
                        next_token_ids[i] = int(next_list[pos])

                for i, (b, rid, sess) in enumerate(ready_pairs):
                    logits_b = last_logits[b : b + 1]
                    if self.use_prefix_cache:
                        cache_key = self._cache_keys.get(rid)
                        if cache_key is not None:
                            eng.prefix_cache.put(cache_key, sess, logits_b)

                    token_id = int(next_token_ids[i])
                    sess.generated_ids.append(token_id)
                    sess.step_count = 1
                    sess.committed_step_count = 1
                    if (
                        sess.stop_on_eos
                        and eng.eos_token_id is not None
                        and token_id == int(eng.eos_token_id)
                    ):
                        sess.finished = True
                    if (
                        sess.max_new_tokens > 0
                        and sess.committed_step_count >= sess.max_new_tokens
                    ):
                        sess.finished = True

                    if sess.finished:
                        sess.release_kv_blocks()
                        just_finished.append(rid)
                    else:
                        self._decode_rids.append(rid)
                    step_tokens.setdefault(rid, token_id)

        if just_finished:
            self._finished_ids.extend(just_finished)
        return step_tokens

    @torch.no_grad()
    def _step_overlap(self) -> dict[int, int]:
        eng = self.engine
        step_tokens: dict[int, int] = {}
        just_finished: list[int] = []

        future_map = self._overlap_future_token_ids_map
        if future_map is None:
            raise RuntimeError("overlap future token map is not initialized")

        # 1) Schedule decode batch (enqueue GPU work first).
        decode_pairs: list[tuple[int, InferenceSession]] = []
        max_examine = len(self._decode_rids)
        while (
            len(decode_pairs) < self.max_batch_size
            and self._decode_rids
            and max_examine > 0
        ):
            max_examine -= 1
            rid = self._decode_rids.popleft()
            sess = self._sessions.get(rid)
            if sess is None or sess.finished:
                continue
            if not sess.generated_ids:
                continue
            if sess.max_new_tokens > 0 and int(sess.step_count) >= int(
                sess.max_new_tokens
            ):
                continue
            decode_pairs.append((rid, sess))

        if decode_pairs:
            for rid, _ in decode_pairs:
                self._decode_rids.append(rid)

            decode_rids = [rid for rid, _ in decode_pairs]
            decode_sessions = [sess for _, sess in decode_pairs]
            input_ids = [int(sess.generated_ids[-1]) for sess in decode_sessions]
            pos_ids = [
                int(sess.prompt_length + sess.step_count - 1)
                for sess in decode_sessions
            ]
            last_logits = eng.decode_step_sessions(
                decode_sessions,
                input_token_ids=input_ids,
                position_ids=pos_ids,
                future_token_ids_map=future_map,
            )

            groups: dict[tuple[float, int, float, bool], list[int]] = {}
            for i, sess in enumerate(decode_sessions):
                key = (
                    float(sess.temperature),
                    int(sess.top_k),
                    float(sess.top_p),
                    bool(sess.do_sample),
                )
                groups.setdefault(key, []).append(i)

            next_ids = torch.empty(
                (len(decode_sessions),),
                device=eng.device,
                dtype=torch.long,
            )
            for (temp, top_k, top_p, do_sample), idxs in groups.items():
                if len(idxs) == len(decode_sessions) and idxs == list(
                    range(len(decode_sessions))
                ):
                    next_ids = eng._sample_next_token_batch(
                        logits=last_logits,
                        temperature=temp,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=do_sample,
                    )
                    break
                idx_t = torch.tensor(idxs, device=eng.device, dtype=torch.long)
                logits_g = last_logits.index_select(0, idx_t)
                sampled = eng._sample_next_token_batch(
                    logits=logits_g,
                    temperature=temp,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                )
                next_ids.index_copy_(0, idx_t, sampled)

            bs = int(len(decode_sessions))
            ct = int(self._overlap_future_token_ids_ct)
            limit = int(self._overlap_future_token_ids_limit)
            placeholder_positions: list[int] = []
            for i, sess in enumerate(decode_sessions):
                placeholder_positions.append(len(sess.generated_ids))
                sess.generated_ids.append(-(ct + i + 1))
                sess.step_count += 1

            future_map[ct + 1 : ct + 1 + bs] = next_ids
            self._overlap_future_token_ids_ct = (ct + bs) % max(1, limit)

            next_ids_cpu = next_ids.to("cpu", non_blocking=True)
            copy_done = torch.cuda.Event()
            copy_done.record()
            self._overlap_decode_pending.append(
                _OverlapDecodeBatch(
                    rids=decode_rids,
                    sessions=decode_sessions,
                    placeholder_positions=placeholder_positions,
                    next_token_ids_cpu=next_ids_cpu,
                    copy_done=copy_done,
                )
            )
            for rid in decode_rids:
                self._overlap_inflight[rid] = self._overlap_inflight.get(rid, 0) + 1

        # 2) Schedule one prefill chunk batch.
        prefill_batch = self._select_prefill_batch()
        if prefill_batch:
            prefill_rids = [rid for rid, _, _ in prefill_batch]
            prefill_sessions = [sess for _, sess, _ in prefill_batch]
            chunks = [chunk for _, _, chunk in prefill_batch]

            last_logits = eng.prefill_chunk_sessions(
                sessions=prefill_sessions,
                chunk_token_ids=chunks,
            )

            ready_bs: list[int] = []
            ready_rids: list[int] = []
            ready_sessions: list[InferenceSession] = []
            for b, (rid, sess, _) in enumerate(prefill_batch):
                prompt_ids = self._prompt_token_ids.get(rid) or []
                if int(sess.prompt_length) < len(prompt_ids):
                    self._prefill_rids.append(rid)
                    continue
                ready_bs.append(b)
                ready_rids.append(rid)
                ready_sessions.append(sess)

                if self.use_prefix_cache:
                    cache_key = self._cache_keys.get(rid)
                    if cache_key is not None:
                        eng.prefix_cache.put(cache_key, sess, last_logits[b : b + 1])

            if ready_sessions:
                logits_ready = last_logits.index_select(
                    0,
                    torch.tensor(ready_bs, device=eng.device, dtype=torch.long),
                )
                groups: dict[tuple[float, int, float, bool], list[int]] = {}
                for i, sess in enumerate(ready_sessions):
                    key = (
                        float(sess.temperature),
                        int(sess.top_k),
                        float(sess.top_p),
                        bool(sess.do_sample),
                    )
                    groups.setdefault(key, []).append(i)

                ready_next = torch.empty(
                    (len(ready_sessions),),
                    device=eng.device,
                    dtype=torch.long,
                )
                for (temp, top_k, top_p, do_sample), idxs in groups.items():
                    if len(idxs) == len(ready_sessions) and idxs == list(
                        range(len(ready_sessions))
                    ):
                        ready_next = eng._sample_next_token_batch(
                            logits=logits_ready,
                            temperature=temp,
                            top_k=top_k,
                            top_p=top_p,
                            do_sample=do_sample,
                        )
                        break
                    idx_t = torch.tensor(idxs, device=eng.device, dtype=torch.long)
                    logits_g = logits_ready.index_select(0, idx_t)
                    sampled = eng._sample_next_token_batch(
                        logits=logits_g,
                        temperature=temp,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=do_sample,
                    )
                    ready_next.index_copy_(0, idx_t, sampled)

                next_ids_cpu = ready_next.to("cpu", non_blocking=True)
            else:
                next_ids_cpu = torch.empty((0,), dtype=torch.long, device="cpu")

            copy_done = torch.cuda.Event()
            copy_done.record()
            self._overlap_prefill_pending.append(
                _OverlapPrefillBatch(
                    all_rids=prefill_rids,
                    all_sessions=prefill_sessions,
                    ready_rids=ready_rids,
                    ready_sessions=ready_sessions,
                    next_token_ids_cpu=next_ids_cpu,
                    copy_done=copy_done,
                )
            )
            for rid in prefill_rids:
                self._overlap_inflight[rid] = self._overlap_inflight.get(rid, 0) + 1

        # 3) Process one pending decode result (delayed output).
        if self._overlap_decode_pending and (
            len(self._overlap_decode_pending) > 1 or not decode_pairs
        ):
            item = self._overlap_decode_pending.popleft()
            item.copy_done.synchronize()
            token_ids = [int(x) for x in item.next_token_ids_cpu.tolist()]
            for rid, sess, pos, token_id in zip(
                item.rids,
                item.sessions,
                item.placeholder_positions,
                token_ids,
            ):
                inflight = self._overlap_inflight.get(rid, 0) - 1
                if inflight <= 0:
                    self._overlap_inflight.pop(rid, None)
                else:
                    self._overlap_inflight[rid] = inflight

                if rid in self._overlap_canceled:
                    if pos < 0 or pos >= len(sess.generated_ids):
                        raise RuntimeError("overlap placeholder position out of range")
                    if int(sess.generated_ids[pos]) < 0:
                        sess.generated_ids[pos] = int(token_id)
                    if inflight <= 0:
                        self._overlap_canceled.discard(rid)
                        sess.release_kv_blocks()
                        self._sessions.pop(rid, None)
                        self._prompt_token_ids.pop(rid, None)
                        self._cache_keys.pop(rid, None)
                    continue

                if sess.finished:
                    if 0 <= pos < len(sess.generated_ids):
                        sess.generated_ids.pop(pos)
                        sess.step_count -= 1
                    if inflight <= 0:
                        just_finished.append(rid)
                        sess.release_kv_blocks()
                    continue

                if pos < 0 or pos >= len(sess.generated_ids):
                    raise RuntimeError("overlap placeholder position out of range")
                if int(sess.generated_ids[pos]) < 0:
                    sess.generated_ids[pos] = int(token_id)
                else:
                    raise RuntimeError("overlap placeholder already resolved")
                sess.committed_step_count += 1

                if sess.stop_on_eos and eng.eos_token_id is not None:
                    if int(token_id) == int(eng.eos_token_id):
                        sess.finished = True
                if (
                    sess.max_new_tokens > 0
                    and sess.committed_step_count >= sess.max_new_tokens
                ):
                    sess.finished = True

                step_tokens[rid] = int(token_id)

                if sess.finished and inflight <= 0:
                    just_finished.append(rid)
                    sess.release_kv_blocks()

        # 4) Process one pending prefill chunk completion (token outputs are delayed).
        if self._overlap_prefill_pending and (
            len(self._overlap_prefill_pending) > 1 or not prefill_batch
        ):
            item = self._overlap_prefill_pending.popleft()
            item.copy_done.synchronize()

            for rid, sess in zip(item.all_rids, item.all_sessions):
                inflight = self._overlap_inflight.get(rid, 0) - 1
                if inflight <= 0:
                    self._overlap_inflight.pop(rid, None)
                else:
                    self._overlap_inflight[rid] = inflight

                if rid in self._overlap_canceled and inflight <= 0:
                    self._overlap_canceled.discard(rid)
                    sess.release_kv_blocks()
                    self._sessions.pop(rid, None)
                    self._prompt_token_ids.pop(rid, None)
                    self._cache_keys.pop(rid, None)

            ready_token_ids = [int(x) for x in item.next_token_ids_cpu.tolist()]
            for rid, sess, token_id in zip(
                item.ready_rids, item.ready_sessions, ready_token_ids
            ):
                if rid in self._overlap_canceled:
                    continue
                sess.generated_ids.append(int(token_id))
                sess.step_count = 1
                sess.committed_step_count = 1
                if (
                    sess.stop_on_eos
                    and eng.eos_token_id is not None
                    and int(token_id) == int(eng.eos_token_id)
                ):
                    sess.finished = True
                if (
                    sess.max_new_tokens > 0
                    and sess.committed_step_count >= sess.max_new_tokens
                ):
                    sess.finished = True

                if sess.finished:
                    just_finished.append(rid)
                    sess.release_kv_blocks()
                else:
                    self._decode_rids.append(rid)
                step_tokens.setdefault(rid, int(token_id))

        if just_finished:
            self._finished_ids.extend(just_finished)
        return step_tokens


@dataclass(slots=True)
class KVBlockInfo:
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
        self._block_infos: list[KVBlockInfo | None] = [
            None for _ in range(num_layers * max_blocks_per_layer)
        ]
        self._k_cache = torch.empty(
            (num_layers, max_blocks_per_layer, num_heads, block_size, head_dim),
            device=device,
            dtype=dtype,
        )
        self._v_cache = torch.empty_like(self._k_cache)
        self._block_refcounts: list[int] = [
            0 for _ in range(num_layers * max_blocks_per_layer)
        ]

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
            self._block_refcounts[global_id] += 1

    def free_blocks(
        self,
        layer_idx: int,
        block_ids: list[int],
    ) -> None:
        for global_id in block_ids:
            ref = self._block_refcounts[global_id]
            if ref <= 0:
                continue
            ref -= 1
            self._block_refcounts[global_id] = ref
            if ref > 0:
                continue
            info = self._block_infos[global_id]
            self._block_infos[global_id] = None
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
        if info is None:
            raise RuntimeError(f"missing KVBlockInfo for block {last_id}")
        ref = self._block_refcounts[last_id]
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
            self._k_cache[layer_idx, block_idx].copy_(
                self._k_cache[layer_idx, info.block_index]
            )
            self._v_cache[layer_idx, block_idx].copy_(
                self._v_cache[layer_idx, info.block_index]
            )
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
        if info is None:
            raise RuntimeError(f"missing KVBlockInfo for block {last_id}")
        k_block = self._k_cache[layer_idx, info.block_index]
        v_block = self._v_cache[layer_idx, info.block_index]
        pos = info.length
        k_block[:, pos, :].copy_(key_new)
        v_block[:, pos, :].copy_(value_new)
        info.length += 1

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
        cow_old_block_idx: list[int] = []
        cow_new_block_idx: list[int] = []
        slow_batch_idx: list[int] = []

        for b, block_ids in enumerate(block_ids_list):
            if not block_ids:
                block_idx = self._alloc_block_index(layer_idx)
                last_gid = self._to_global_block_id(layer_idx, block_idx)
                info = KVBlockInfo(
                    layer=layer_idx,
                    block_index=block_idx,
                    start=0,
                    length=0,
                )
                self._block_infos[last_gid] = info
                self._block_refcounts[last_gid] = 1
                block_ids.append(last_gid)
                ref = 1
            else:
                last_gid = block_ids[-1]
                info = self._block_infos[last_gid]
                if info is None:
                    raise RuntimeError(f"missing KVBlockInfo for block {last_gid}")
                ref = self._block_refcounts[last_gid]

            if info.length >= self.block_size:
                block_idx = self._alloc_block_index(layer_idx)
                new_gid = self._to_global_block_id(layer_idx, block_idx)
                info = KVBlockInfo(
                    layer=info.layer,
                    block_index=block_idx,
                    start=info.start + info.length,
                    length=0,
                )
                self._block_infos[new_gid] = info
                self._block_refcounts[new_gid] = 1
                block_ids.append(new_gid)
                last_gid = new_gid
                ref = 1

            if ref != 1:
                old_block_idx = info.block_index
                old_length = info.length
                self._block_refcounts[last_gid] = ref - 1
                block_idx = self._alloc_block_index(layer_idx)
                new_gid = self._to_global_block_id(layer_idx, block_idx)
                info = KVBlockInfo(
                    layer=info.layer,
                    block_index=block_idx,
                    start=info.start,
                    length=old_length,
                )
                self._block_infos[new_gid] = info
                self._block_refcounts[new_gid] = 1
                block_ids[-1] = new_gid
                cow_old_block_idx.append(old_block_idx)
                cow_new_block_idx.append(block_idx)

            fast_batch_idx.append(b)
            fast_block_idx.append(info.block_index)
            fast_pos.append(info.length)
            info.length += 1

        if fast_batch_idx:
            device = self.device
            k_layer = self._k_cache[layer_idx]
            v_layer = self._v_cache[layer_idx]
            if cow_old_block_idx:
                use_triton_clone = False
                kv_clone_blocks_triton = None
                if device.type == "cuda":
                    try:
                        from rosellm.roseinfer.kv_clone_triton import (
                            TRITON_AVAILABLE as TRITON_CLONE_AVAILABLE,
                        )
                        from rosellm.roseinfer.kv_clone_triton import (
                            USE_TRITON_KV_CLONE,
                        )
                        from rosellm.roseinfer.kv_clone_triton import (
                            kv_clone_blocks_triton as _kv_clone_blocks_triton,
                        )

                        use_triton_clone = (
                            TRITON_CLONE_AVAILABLE and USE_TRITON_KV_CLONE
                        )
                        kv_clone_blocks_triton = _kv_clone_blocks_triton
                    except Exception:
                        use_triton_clone = False
                        kv_clone_blocks_triton = None

                if use_triton_clone and kv_clone_blocks_triton is not None:
                    old_blk_t = torch.tensor(
                        cow_old_block_idx,
                        device=device,
                        dtype=torch.int32,
                    )
                    new_blk_t = torch.tensor(
                        cow_new_block_idx,
                        device=device,
                        dtype=torch.int32,
                    )
                    kv_clone_blocks_triton(
                        k_cache_layer=k_layer,
                        v_cache_layer=v_layer,
                        src_block_idx=old_blk_t,
                        dst_block_idx=new_blk_t,
                    )
                else:
                    old_blk_t = torch.tensor(
                        cow_old_block_idx,
                        device=device,
                        dtype=torch.long,
                    )
                    new_blk_t = torch.tensor(
                        cow_new_block_idx,
                        device=device,
                        dtype=torch.long,
                    )
                    k_src = k_layer.index_select(0, old_blk_t)
                    v_src = v_layer.index_select(0, old_blk_t)
                    k_layer.index_copy_(0, new_blk_t, k_src)
                    v_layer.index_copy_(0, new_blk_t, v_src)
            use_triton = False
            use_triton_full_batch = False
            use_triton_identity_pos = False
            kv_append_triton = None
            kv_append_triton_full_batch = None
            kv_append_triton_identity_pos = None
            full_fast = (
                len(slow_batch_idx) == 0
                and len(fast_batch_idx) == len(block_ids_list)
                and fast_batch_idx[0] == 0
                and fast_batch_idx[-1] == len(block_ids_list) - 1
            )
            pos0 = fast_pos[0]
            const_pos = all(p == pos0 for p in fast_pos)
            if device.type == "cuda":
                try:
                    from rosellm.roseinfer.kv_append_triton import (
                        TRITON_AVAILABLE,
                        TRITON_KV_APPEND_FULL_BATCH_MIN_BATCH,
                        TRITON_KV_APPEND_MIN_BATCH,
                        USE_TRITON_KV_APPEND,
                    )
                    from rosellm.roseinfer.kv_append_triton import (
                        kv_append_triton as _kv_append_triton,
                    )
                    from rosellm.roseinfer.kv_append_triton import (
                        kv_append_triton_full_batch as _kv_append_triton_full_batch,
                    )
                    from rosellm.roseinfer.kv_append_triton import (
                        kv_append_triton_identity_pos as _kv_append_triton_identity_pos,
                    )

                    use_triton_full_batch = (
                        TRITON_AVAILABLE
                        and USE_TRITON_KV_APPEND
                        and full_fast
                        and const_pos
                        and len(fast_batch_idx) >= TRITON_KV_APPEND_FULL_BATCH_MIN_BATCH
                    )
                    use_triton = (
                        TRITON_AVAILABLE
                        and USE_TRITON_KV_APPEND
                        and len(fast_batch_idx) >= TRITON_KV_APPEND_MIN_BATCH
                    )
                    use_triton_identity_pos = (
                        TRITON_AVAILABLE
                        and USE_TRITON_KV_APPEND
                        and full_fast
                        and not const_pos
                        and len(fast_batch_idx) >= TRITON_KV_APPEND_MIN_BATCH
                    )
                    kv_append_triton = _kv_append_triton
                    kv_append_triton_full_batch = _kv_append_triton_full_batch
                    kv_append_triton_identity_pos = _kv_append_triton_identity_pos
                except Exception:
                    use_triton = False
                    use_triton_full_batch = False
                    use_triton_identity_pos = False
                    kv_append_triton = None
                    kv_append_triton_full_batch = None
                    kv_append_triton_identity_pos = None

            if use_triton_full_batch and kv_append_triton_full_batch is not None:
                blk_t = torch.tensor(
                    fast_block_idx,
                    device=device,
                    dtype=torch.int32,
                )
                kv_append_triton_full_batch(
                    k_cache_layer=k_layer,
                    v_cache_layer=v_layer,
                    key_new=key_new,
                    value_new=value_new,
                    block_idx=blk_t,
                    pos=pos0,
                )
            elif use_triton_identity_pos and kv_append_triton_identity_pos is not None:
                blk_t = torch.tensor(
                    fast_block_idx,
                    device=device,
                    dtype=torch.int32,
                )
                pos_t = torch.tensor(
                    fast_pos,
                    device=device,
                    dtype=torch.int32,
                )
                kv_append_triton_identity_pos(
                    k_cache_layer=k_layer,
                    v_cache_layer=v_layer,
                    key_new=key_new,
                    value_new=value_new,
                    block_idx=blk_t,
                    pos=pos_t,
                )
            elif use_triton and kv_append_triton is not None:
                idx_t = torch.tensor(
                    fast_batch_idx,
                    device=device,
                    dtype=torch.int32,
                )
                blk_t = torch.tensor(
                    fast_block_idx,
                    device=device,
                    dtype=torch.int32,
                )
                pos_t = torch.tensor(
                    fast_pos,
                    device=device,
                    dtype=torch.int32,
                )
                kv_append_triton(
                    k_cache_layer=k_layer,
                    v_cache_layer=v_layer,
                    key_new=key_new,
                    value_new=value_new,
                    batch_idx=idx_t,
                    block_idx=blk_t,
                    pos=pos_t,
                )
            else:
                blk_t = torch.tensor(
                    fast_block_idx,
                    device=device,
                    dtype=torch.long,
                )
                if full_fast:
                    k_src = key_new
                    v_src = value_new
                else:
                    idx_t = torch.tensor(
                        fast_batch_idx,
                        device=device,
                        dtype=torch.long,
                    )
                    k_src = key_new.index_select(0, idx_t)
                    v_src = value_new.index_select(0, idx_t)

                if const_pos:
                    k_layer[blk_t, :, pos0, :] = k_src
                    v_layer[blk_t, :, pos0, :] = v_src
                else:
                    pos_t = torch.tensor(
                        fast_pos,
                        device=device,
                        dtype=torch.long,
                    )
                    k_layer[blk_t, :, pos_t, :] = k_src
                    v_layer[blk_t, :, pos_t, :] = v_src

        for b in slow_batch_idx:
            self.append_token(
                layer_idx,
                block_ids_list[b],
                key_new[b],
                value_new[b],
            )

    def _clone_blocks(
        self,
        *,
        layer_idx: int,
        src_block_idx: list[int],
        dst_block_idx: list[int],
    ) -> None:
        if not src_block_idx:
            return
        if len(src_block_idx) != len(dst_block_idx):
            raise ValueError("src_block_idx and dst_block_idx size mismatch")
        device = self.device
        k_layer = self._k_cache[layer_idx]
        v_layer = self._v_cache[layer_idx]
        use_triton_clone = False
        kv_clone_blocks_triton = None
        if device.type == "cuda":
            try:
                from rosellm.roseinfer.kv_clone_triton import (
                    TRITON_AVAILABLE as TRITON_CLONE_AVAILABLE,
                )
                from rosellm.roseinfer.kv_clone_triton import USE_TRITON_KV_CLONE
                from rosellm.roseinfer.kv_clone_triton import (
                    kv_clone_blocks_triton as _kv_clone_blocks_triton,
                )

                use_triton_clone = TRITON_CLONE_AVAILABLE and USE_TRITON_KV_CLONE
                kv_clone_blocks_triton = _kv_clone_blocks_triton
            except Exception:
                use_triton_clone = False
                kv_clone_blocks_triton = None

        if use_triton_clone and kv_clone_blocks_triton is not None:
            src_t = torch.tensor(
                src_block_idx,
                device=device,
                dtype=torch.int32,
            )
            dst_t = torch.tensor(
                dst_block_idx,
                device=device,
                dtype=torch.int32,
            )
            kv_clone_blocks_triton(
                k_cache_layer=k_layer,
                v_cache_layer=v_layer,
                src_block_idx=src_t,
                dst_block_idx=dst_t,
            )
            return

        src_t = torch.tensor(
            src_block_idx,
            device=device,
            dtype=torch.long,
        )
        dst_t = torch.tensor(
            dst_block_idx,
            device=device,
            dtype=torch.long,
        )
        k_src = k_layer.index_select(0, src_t)
        v_src = v_layer.index_select(0, src_t)
        k_layer.index_copy_(0, dst_t, k_src)
        v_layer.index_copy_(0, dst_t, v_src)

    def reserve_append_tokens(
        self,
        *,
        layer_idx: int,
        block_ids: list[int],
        n_append: int,
    ) -> None:
        n_append = int(n_append)
        if n_append <= 0:
            return
        if not (0 <= layer_idx < self.num_layers):
            raise ValueError("layer_idx out of range")

        # Ensure at least one block exists.
        if not block_ids:
            block_idx = self._alloc_block_index(layer_idx)
            gid = self._to_global_block_id(layer_idx, block_idx)
            info = KVBlockInfo(
                layer=layer_idx,
                block_index=block_idx,
                start=0,
                length=0,
            )
            self._block_infos[gid] = info
            self._block_refcounts[gid] = 1
            block_ids.append(gid)

        block_size = int(self.block_size)

        # If the last block is full, allocate a new empty one first.
        last_gid = int(block_ids[-1])
        last_info = self._block_infos[last_gid]
        if last_info is None:
            raise RuntimeError(f"missing KVBlockInfo for block {last_gid}")
        if int(last_info.length) >= block_size:
            block_idx = self._alloc_block_index(layer_idx)
            gid = self._to_global_block_id(layer_idx, block_idx)
            info = KVBlockInfo(
                layer=layer_idx,
                block_index=block_idx,
                start=int(last_info.start + last_info.length),
                length=0,
            )
            self._block_infos[gid] = info
            self._block_refcounts[gid] = 1
            block_ids.append(gid)
            last_gid = gid
            last_info = info

        # Copy-on-write for shared last block if we will write into it.
        ref = int(self._block_refcounts[last_gid])
        if ref != 1 and int(last_info.length) < block_size:
            old_gid = last_gid
            old_info = last_info
            old_block_idx = int(old_info.block_index)
            old_len = int(old_info.length)
            self._block_refcounts[old_gid] = ref - 1
            new_block_idx = self._alloc_block_index(layer_idx)
            new_gid = self._to_global_block_id(layer_idx, new_block_idx)
            new_info = KVBlockInfo(
                layer=layer_idx,
                block_index=new_block_idx,
                start=int(old_info.start),
                length=int(old_len),
            )
            self._block_infos[new_gid] = new_info
            self._block_refcounts[new_gid] = 1
            block_ids[-1] = new_gid
            self._clone_blocks(
                layer_idx=layer_idx,
                src_block_idx=[old_block_idx],
                dst_block_idx=[new_block_idx],
            )
            last_gid = new_gid
            last_info = new_info

        # Fill remaining space in the last block.
        avail = block_size - int(last_info.length)
        take = min(avail, n_append)
        last_info.length += int(take)
        n_append -= int(take)
        if n_append <= 0:
            return

        # Allocate additional blocks for remaining tokens.
        prev_info = last_info
        while n_append > 0:
            block_idx = self._alloc_block_index(layer_idx)
            gid = self._to_global_block_id(layer_idx, block_idx)
            info = KVBlockInfo(
                layer=layer_idx,
                block_index=block_idx,
                start=int(prev_info.start + prev_info.length),
                length=0,
            )
            self._block_infos[gid] = info
            self._block_refcounts[gid] = 1
            block_ids.append(gid)
            fill_len = min(block_size, n_append)
            info.length = int(fill_len)
            n_append -= int(fill_len)
            prev_info = info

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
