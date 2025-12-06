from __future__ import annotations

import codecs
from typing import List

from transformers import PreTrainedTokenizerBase

try:
    import tiktoken
except ImportError:
    tiktoken = None


class BaseDetokenizer:
    def reset(self) -> None:
        raise NotImplementedError

    def start_prompt(
        self,
        prompt_ids: List[int],
    ) -> None:
        raise NotImplementedError

    def on_token(
        self,
        token_id: int,
    ) -> str:
        raise NotImplementedError

    def flush(self) -> str:
        return ""


class GPT2ByteDetokenizer(BaseDetokenizer):
    def __init__(
        self,
        hf_tokenizer: PreTrainedTokenizerBase,
        model_name: str = "gpt2",
    ) -> None:
        if tiktoken is None:
            raise ImportError("need to install tiktoken")
        self.hf_tokenizer = hf_tokenizer
        self.enc = tiktoken.encoding_for_model(model_name)
        self.decoder = codecs.getincrementaldecoder(
            "utf-8",
        )(errors="replace")

    def reset(self) -> None:
        self.decoder = codecs.getincrementaldecoder(
            "utf-8",
        )(errors="replace")

    def start_prompt(
        self,
        prompt_ids: List[int],
    ) -> None:
        self.reset()

    def on_token(
        self,
        token_id: int,
    ) -> str:
        token_bytes = self.enc.decode_single_token_bytes(token_id)
        text_piece = self.decoder.decode(token_bytes)
        return text_piece

    def flush(self) -> str:
        return self.decoder.decode(
            b"",
            final=True,
        )


class PrefixDiffDetokenizer(BaseDetokenizer):
    def __init__(
        self,
        hf_tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.tok = hf_tokenizer
        self.generated_ids: List[int] = []
        self.last_text: str = ""

    def reset(self) -> None:
        self.generated_ids = []
        self.last_text = ""

    def start_prompt(
        self,
        prompt_ids: List[int],
    ) -> None:
        self.reset()
        self.generated_ids = prompt_ids.copy()
        self.last_text = self.tok.decode(
            self.generated_ids,
            skip_special_tokens=True,
        )

    def on_token(
        self,
        token_id: int,
    ) -> str:
        self.generated_ids.append(token_id)
        full = self.tok.decode(
            self.generated_ids,
            skip_special_tokens=True,
        )
        delta = full[len(self.last_text) :]
        self.last_text = full
        return delta
