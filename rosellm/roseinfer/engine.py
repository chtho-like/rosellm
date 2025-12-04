from typing import Optional

import torch
from rosetrainer.config import GPTConfig
from rosetrainer.dataset import build_tokenizer
from rosetrainer.model import GPTModel


class InferenceEngine:
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_name: str = "gpt2",
        device: Optional[str] = None,
        use_amp: bool = True,
        max_position_embeddings: Optional[int] = None,
        bf16: bool = False,
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
        self.kv_cache = None
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
            batch.append(ids + [pad_id] * pad_len)
            masks.append([1] * len(ids) + [0] * pad_len)
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
    def prefill(
        self,
        prompt_ids: torch.Tensor,  # [..., T0]
    ):
        from torch.amp import autocast

        input_ids = self._maybe_truncate(prompt_ids)
        if self.use_amp:
            with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                logits, _, presents = self.model(
                    input_ids=input_ids,
                    attention_mask=None,
                    labels=None,
                    past_key_values=None,
                    use_cache=True,
                )
        else:
            logits, _, presents = self.model(
                input_ids=input_ids,
                attention_mask=None,
                labels=None,
                past_key_values=None,
                use_cache=True,
            )
        self.kv_cache = presents
        return logits  # [..., T0, vocab]

    @torch.no_grad()
    def prefill_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from torch.amp import autocast

        input_ids = self._maybe_truncate(input_ids)
        if attention_mask is not None and input_ids.size(1) < attention_mask.size(1):
            attention_mask = attention_mask[:, -input_ids.size(1) :]
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
                )
        else:
            logits, _, presents = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                past_key_values=None,
                use_cache=True,
            )
        self.kv_cache = presents
        if attention_mask is None:
            last_logits = logits[:, -1, :]  # [batch, vocab]
        else:
            batch_size = logits.size(0)
            lengths = attention_mask.sum(dim=1).to(dtype=torch.long)  # [B]
            last_indices = lengths - 1  # [B]
            batch_indices = torch.arange(
                batch_size,
                device=logits.device,
            )
            last_logits = logits[batch_indices, last_indices, :]  # [B, vocab]
        return last_logits

    @torch.no_grad()
    def decode_step(self, last_token_id: int) -> torch.Tensor:
        assert self.kv_cache is not None
        from torch.amp import autocast

        input_ids = torch.tensor(  # [1, 1]
            [[last_token_id]],
            dtype=torch.long,
            device=self.device,
        )
        if self.use_amp:
            with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                logits, _, presents = self.model(
                    input_ids=input_ids,
                    attention_mask=None,
                    labels=None,
                    past_key_values=self.kv_cache,
                    use_cache=True,
                )
        else:
            logits, _, presents = self.model(
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
    def decode_step_batch(
        self,
        last_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        assert self.kv_cache is not None
        from torch.amp import autocast

        input_ids = last_token_ids.view(-1, 1)  # [B, 1]
        if self.use_amp:
            with autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
            ):
                logits, _, presents = self.model(
                    input_ids=input_ids,
                    attention_mask=None,
                    labels=None,
                    past_key_values=self.kv_cache,
                    use_cache=True,
                )
        else:
            logits, _, presents = self.model(
                input_ids=input_ids,
                attention_mask=None,
                labels=None,
                past_key_values=self.kv_cache,
                use_cache=True,
            )
        self.kv_cache = presents
        next_logits = logits[:, -1, :]  # [B, V]
        return next_logits  # [B, V]

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
        input_ids = self._encode_prompt(prompt)  # [1, T0]
        input_ids = self._maybe_truncate(input_ids)  # [1, T]
        logits = self.prefill(input_ids)  # [1, T, V]
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
            next_logits = self.decode_step(last_token_id)
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
        input_ids, attn_mask = self._encode_prompts_batch(prompts)
        batch_size = input_ids.size(0)
        last_logits = self.prefill_batch(
            input_ids,
            attention_mask=attn_mask,
        )
        lengths = attn_mask.sum(dim=1).tolist()
        generated_ids = [input_ids[b, : lengths[b]].tolist() for b in range(batch_size)]
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
            next_logits = self.decode_step_batch(last_token_ids)
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
