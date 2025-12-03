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
    ) -> None:
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.use_amp = use_amp and self.device.type == "cuda"
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
        if self.config.vocab_size < self.tokenizer.vocab_size:
            raise ValueError("the model vocab_size is less than tokenizer vocab_size")

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            ids = [self.eos_token_id]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        return input_ids  # [1, T0]

    def _decode_tokens(self, token_ids: torch.Tensor) -> str:
        ids = token_ids.tolist()
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return text

    def _maybe_truncate(self, input_ids: torch.Tensor) -> torch.Tensor:
        max_pos = self.config.max_position_embeddings
        if input_ids.size(1) > max_pos:
            input_ids = input_ids[:, -max_pos:]
        return input_ids

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_on_eos: bool = True,
    ) -> str:
        self.model.eval()
        input_ids = self._encode_prompt(prompt)  # [1, T0]
        input_ids = self._maybe_truncate(input_ids)  # [1, T]
        from torch.amp import autocast

        for _ in range(max_new_tokens):
            if self.use_amp:
                with autocast(device_type=self.device.type):
                    logits, _ = self.model(  # [1, T, V]
                        input_ids=input_ids,
                        attention_mask=None,
                        labels=None,
                    )
            else:
                logits, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=None,
                    labels=None,
                )
            next_logits = logits[:, -1, :]  # [1, V]
            next_token = torch.argmax(next_logits, dim=-1)  # [1]
            next_id = next_token.item()
            next_token_t = next_token.view(1, 1)  # [1, 1]
            input_ids = torch.cat(  # [1, T+1]
                [input_ids, next_token_t],
                dim=1,
            )
            input_ids = self._maybe_truncate(input_ids)
            if (
                stop_on_eos
                and self.eos_token_id is not None
                and next_id == self.eos_token_id
            ):
                break
        generated = input_ids[0]  # [T1]
        text = self._decode_tokens(generated)
        return text
