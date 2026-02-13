import pytest

from rosellm.roseinfer.engine import _kv_cache_max_tokens_from_total_utilization
from rosellm.roseinfer.server import parse_args


def test_kv_cache_max_tokens_from_total_utilization_basic() -> None:
    out = _kv_cache_max_tokens_from_total_utilization(
        free_bytes=600,
        total_bytes=1000,
        gpu_memory_utilization=0.9,
        bytes_per_token=10,
    )
    assert out == 50


def test_kv_cache_max_tokens_from_total_utilization_raises_on_no_budget() -> None:
    with pytest.raises(ValueError, match="not enough free memory"):
        _kv_cache_max_tokens_from_total_utilization(
            free_bytes=100,
            total_bytes=1000,
            gpu_memory_utilization=0.8,
            bytes_per_token=10,
        )


def test_parse_args_accepts_gpu_memory_utilization(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "server.py",
            "--hf-model-id",
            "gpt2",
            "--gpu-memory-utilization",
            "0.9",
        ],
    )
    args = parse_args()
    assert float(getattr(args, "gpu_memory_utilization", 0.0) or 0.0) == 0.9
