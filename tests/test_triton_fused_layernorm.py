import importlib

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("hidden_size", [768, 1024, 1600])
def test_triton_layer_norm_matches_torch(hidden_size: int, monkeypatch) -> None:
    monkeypatch.setenv("ROSELLM_TRITON_FUSED_LAYERNORM", "1")
    monkeypatch.setenv("ROSELLM_TRITON_FUSED_ADD_LAYERNORM", "1")
    import rosellm.rosetrainer.fused_layernorm as fused

    importlib.reload(fused)
    if not fused.TRITON_AVAILABLE:
        pytest.skip("requires Triton")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    x = torch.randn((4, 3, hidden_size), device=device, dtype=dtype)
    weight = torch.randn((hidden_size,), device=device, dtype=dtype)
    bias = torch.randn((hidden_size,), device=device, dtype=dtype)
    eps = 1e-5

    y = fused.layer_norm(x, weight, bias, eps=eps)
    y_ref = F.layer_norm(x, (hidden_size,), weight, bias, eps)
    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("hidden_size", [768, 1024, 1600])
def test_triton_add_layer_norm_inplace_matches_torch(
    hidden_size: int, monkeypatch
) -> None:
    monkeypatch.setenv("ROSELLM_TRITON_FUSED_LAYERNORM", "1")
    monkeypatch.setenv("ROSELLM_TRITON_FUSED_ADD_LAYERNORM", "1")
    import rosellm.rosetrainer.fused_layernorm as fused

    importlib.reload(fused)
    if not fused.TRITON_AVAILABLE:
        pytest.skip("requires Triton")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    x0 = torch.randn((2, 5, hidden_size), device=device, dtype=dtype)
    residual = torch.randn_like(x0)
    weight = torch.randn((hidden_size,), device=device, dtype=dtype)
    bias = torch.randn((hidden_size,), device=device, dtype=dtype)
    eps = 1e-5

    x_ref = x0.clone()
    x_ref.add_(residual)
    y_ref = F.layer_norm(x_ref, (hidden_size,), weight, bias, eps)

    x = x0.clone()
    y = fused.add_layer_norm_(x, residual, weight, bias, eps=eps)

    torch.testing.assert_close(x, x_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)
