from collections import OrderedDict
from typing import Iterator, NamedTuple, Optional

import torch
from roseinfer.detokenizer import (
    BaseDetokenizer,
    GPT2ByteDetokenizer,
    PrefixDiffDetokenizer,
)
from rosetrainer.config import GPTConfig
from rosetrainer.dataset import build_tokenizer
from rosetrainer.model import GPTModel

try:
    import tiktoken
except ImportError:
    tiktoken = None


class InferenceEngine:
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_name: str = "gpt2",
        device: Optional[str] = None,
        use_amp: bool = True,
        max_position_embeddings: Optional[int] = None,
        bf16: bool = False,
        kv_cache_max_concurrency: int = 256,
        prefix_cache_max_entries: int = 256,
    ) -> None:
        super().__init__()
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
        ckpt = torch.load(checkpoint_path, map_location=self.device.type)
        cfg_dict = ckpt.get("config")
        if cfg_dict is None:
            print("cannot find config from checkpoints, use GPTConfig")
            config = GPTConfig()
        else:
            config = GPTConfig(**cfg_dict)
        if max_position_embeddings is not None:
            config.max_position_embeddings = max_position_embeddings
        self.config = config
        self.model = GPTModel(config).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.tokenizer = build_tokenizer(tokenizer_name)
        self.eos_token_id = self.tokenizer.eos_token_id

        def make_detok() -> BaseDetokenizer:
            if tokenizer_name.startswith("gpt2") and tiktoken is not None:
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
        self.kv_manager = KVBlockManager(
            num_layers=self.config.n_layers,
            num_heads=self.config.n_heads,
            head_dim=self.config.d_model // self.config.n_heads,
            block_size=block_size,
            max_blocks_per_layer=max_blocks_per_layer,
            device=self.device,
            dtype=self.amp_dtype if self.use_amp else self.model.dtype,
        )
        self.prefix_cache = PrefixCache(
            self.kv_manager,
            max_entries=prefix_cache_max_entries,
        )

        if self.config.vocab_size < self.tokenizer.vocab_size:
            raise ValueError("the model vocab_size is less than tokenizer vocab_size")

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            ids = [self.eos_token_id]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        return input_ids  # [1, T0]

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
    ) -> None:
        input_ids = self._encode_prompt(prompt)
        input_ids = self._maybe_truncate(input_ids)
        session.input_ids = input_ids
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

    @torch.no_grad()
    def decode_step_sessions(
        self,
        sessions: list["InferenceSession"],
    ) -> torch.Tensor:
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
        max_len = max(seq_lens)

        input_ids = torch.tensor(  # [B, 1]
            last_ids,
            dtype=torch.long,
            device=device,
        ).view(batch_size, 1)
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
        num_layers = kvm.num_layers
        for layer_idx in range(num_layers):
            k_list = []
            v_list = []
            for idx, sess in enumerate(sessions):
                seq_len = seq_lens[idx]
                block_ids = sess.block_ids_per_layer[layer_idx]
                k_seq, v_seq = kvm.gather_sequence(
                    layer_idx,
                    block_ids,
                    seq_len,
                )  # [1, H, T_i, D]
                T_i = k_seq.size(2)
                if T_i < max_len:
                    pad_len = max_len - T_i
                    pad_shape = (
                        1,
                        k_seq.size(1),
                        pad_len,
                        k_seq.size(3),
                    )
                    k_pad = torch.zeros(
                        pad_shape,
                        dtype=k_seq.dtype,
                        device=k_seq.device,
                    )
                    v_pad = torch.zeros(
                        pad_shape,
                        dtype=v_seq.dtype,
                        device=v_seq.device,
                    )
                    k_full = torch.cat(
                        [k_seq, k_pad],
                        dim=2,
                    )
                    v_full = torch.cat(
                        [v_seq, v_pad],
                        dim=2,
                    )
                else:
                    k_full = k_seq
                    v_full = v_seq
                k_list.append(k_full)
                v_list.append(v_full)
            k_cat = torch.cat(k_list, dim=0)
            v_cat = torch.cat(v_list, dim=0)
            batched_past.append((k_cat, v_cat))
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
                )
        else:
            logits, _, presents = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                past_key_values=tuple(batched_past),
                use_cache=True,
            )
        last_logits = logits[:, -1, :]  # [B, V]
        for layer_idx in range(num_layers):
            k_b, v_b = presents[layer_idx]
            for idx, sess in enumerate(sessions):
                if sess.finished:
                    continue
                k_new = k_b[
                    idx : idx + 1,
                    :,
                    max_len : max_len + 1,
                    :,
                ]
                v_new = v_b[
                    idx : idx + 1,
                    :,
                    max_len : max_len + 1,
                    :,
                ]
                kvm.append_token(
                    layer_idx,
                    sess.block_ids_per_layer[layer_idx],
                    k_new,
                    v_new,
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
        for layer_idx, block_ids in enumerate(entry.block_ids_per_layer):
            if block_ids:
                self.kv_manager.free_blocks(layer_idx, block_ids)

    def _evict_one(self) -> None:
        if not self._entries:
            return
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
        self._round_robin_pos: int = 0

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
        )
        if session.finished:
            session.release_kv_blocks()
        request_id = self._next_request_id
        self._next_request_id += 1
        self._sessions[request_id] = session
        return request_id

    def has_unfinished(self) -> bool:
        return any(not sess.finished for sess in self._sessions.values())

    def is_finished(self, request_id: int) -> bool:
        session = self._sessions.get(request_id, None)
        return session.finished

    @torch.no_grad()
    def step(self) -> dict[int, int]:
        active_pairs: list[tuple[int, InferenceSession]] = [
            (rid, sess) for rid, sess in self._sessions.items() if not sess.finished
        ]
        if not active_pairs:
            return {}
        num_active = len(active_pairs)
        batch_size = min(self.max_batch_size, num_active)
        start = self._round_robin_pos % num_active
        selected_pairs: list[tuple[int, InferenceSession]] = []
        for i in range(batch_size):
            idx = (start + i) % num_active
            selected_pairs.append(active_pairs[idx])
        self._round_robin_pos = (start + batch_size) % num_active
        sessions = [sess for _, sess in selected_pairs]
        last_logits = self.engine.decode_step_sessions(sessions)
        step_tokens: dict[int, int] = {}
        for idx, (rid, sess) in enumerate(selected_pairs):
            logits_row = last_logits[idx]
            token_id = sess.apply_batch_logits(logits_row)
            if token_id is not None:
                step_tokens[rid] = token_id
                if sess.finished:
                    sess.release_kv_blocks()
        return step_tokens

    def get_response(self, request_id: int) -> str:
        session = self._sessions[request_id]
        return session.decode_text()

    def pop_response(self, request_id: int) -> str:
        session = self._sessions.pop(request_id)
        session.release_kv_blocks()
        return session.decode_text()


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
        self._block_storage: dict[
            int,
            tuple[torch.Tensor, torch.Tensor],
        ] = {}  # global_id -> (key_block, value_block)
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
        key: torch.Tensor,  # [1, H, D, T]
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
            k_block = torch.zeros(
                (
                    self.num_heads,
                    block_size,
                    self.head_dim,
                ),
                dtype=self.dtype,
                device=self.device,
            )
            v_block = torch.zeros_like(k_block)
            k_block[:, :length, :] = k_slice[0]
            v_block[:, :length, :] = v_slice[0]
            self._block_storage[global_id] = (k_block, v_block)
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
            self._block_storage.pop(global_id, None)

    def append_token(
        self,
        layer_idx: int,
        block_ids: list[int],
        key_new: torch.Tensor,
        value_new: torch.Tensor,
    ) -> None:
        assert 0 <= layer_idx < self.num_layers
        assert key_new.size(2) == 1
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
            k_block = torch.zeros(
                (
                    self.num_heads,
                    self.block_size,
                    self.head_dim,
                ),
                dtype=self.dtype,
                device=self.device,
            )
            v_block = torch.zeros_like(k_block)
            self._block_storage[global_id] = (k_block, v_block)
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
            k_block_old, v_block_old = self._block_storage[last_id]
            k_block = k_block_old.clone()
            v_block = v_block_old.clone()
            self._block_storage[new_global_id] = (k_block, v_block)
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
            k_block = torch.zeros(
                (
                    self.num_heads,
                    self.block_size,
                    self.head_dim,
                ),
                dtype=self.dtype,
                device=self.device,
            )
            v_block = torch.zeros_like(k_block)
            self._block_storage[global_id] = (k_block, v_block)
            self._block_refcounts[global_id] = 1
            block_ids.append(global_id)
            last_id = global_id
        info = self._block_infos[last_id]
        k_block, v_block = self._block_storage[last_id]
        pos = info.length
        k_block[:, pos, :] = key_new[0, :, 0, :]
        v_block[:, pos, :] = value_new[0, :, 0, :]
        new_info = KVBlockInfo(
            layer=info.layer,
            block_index=info.block_index,
            start=info.start,
            length=info.length + 1,
        )
        self._block_infos[last_id] = new_info

    def gather_sequence(
        self,
        layer_idx: int,
        block_ids: list[int],
        total_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert 0 <= layer_idx < self.num_layers
        k_seq = torch.zeros(
            (
                1,
                self.num_heads,
                total_len,
                self.head_dim,
            ),
            dtype=self.dtype,
            device=self.device,
        )
        v_seq = torch.zeros_like(k_seq)
        cur = 0
        for global_id in block_ids:
            info = self._block_infos[global_id]
            if info is None:
                continue
            if info.layer != layer_idx:
                continue
            k_block, v_block = self._block_storage[global_id]
            length = info.length
            if length <= 0:
                continue
            end = min(cur + length, total_len)
            take = end - cur
            if take <= 0:
                break
            k_seq[0, :, cur:end, :] = k_block[:, :take, :]
            v_seq[0, :, cur:end, :] = v_block[:, :take, :]
            cur = end
            if cur >= total_len:
                break
        return k_seq, v_seq
