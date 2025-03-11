from typing import Optional


class AutoConfig:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        _commit_hash: Optional[str] = None,
    ):
        pass
