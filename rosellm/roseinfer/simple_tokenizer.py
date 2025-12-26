from __future__ import annotations


class ByteTokenizer:
    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = int(vocab_size)
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be >= 2")
        self.eos_token_id = 0
        self.pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if not text:
            return [self.eos_token_id]
        out: list[int] = []
        for b in text.encode("utf-8"):
            out.append(int(b % (self.vocab_size - 1)) + 1)
        return out

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(i) for i in ids)
