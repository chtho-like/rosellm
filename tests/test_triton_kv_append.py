import importlib

import pytest
import torch

from rosellm.roseinfer.engine import KVBlockManager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kv_append_triton_fast_path_writes_correct_values(monkeypatch) -> None:
    monkeypatch.setenv("ROSELLM_TRITON_KV_APPEND", "1")
    monkeypatch.setenv("ROSELLM_TRITON_KV_APPEND_MIN_BATCH", "1")
    import rosellm.roseinfer.kv_append_triton as kv_mod

    importlib.reload(kv_mod)
    if not kv_mod.TRITON_AVAILABLE:
        pytest.skip("requires Triton")

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_layers = 1
    num_heads = 2
    head_dim = 8
    block_size = 4
    batch_size = 5
    kvm = KVBlockManager(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_blocks_per_layer=32,
        device=device,
        dtype=torch.float16,
    )

    block_ids_list: list[list[int]] = [[] for _ in range(batch_size)]
    for b in range(batch_size):
        k0 = torch.randn((num_heads, head_dim), device=device, dtype=torch.float16)
        v0 = torch.randn_like(k0)
        kvm.append_token(0, block_ids_list[b], k0, v0)

    old_meta: list[tuple[int, int]] = []
    for b in range(batch_size):
        gid = block_ids_list[b][-1]
        info = kvm._block_infos[gid]
        old_meta.append((info.block_index, info.length))
        assert info.length == 1

    key_new = torch.randn(
        (batch_size, num_heads, head_dim), device=device, dtype=torch.float16
    )
    value_new = torch.randn_like(key_new)
    kvm.append_token_batch(0, block_ids_list, key_new, value_new)

    k_layer = kvm._k_cache[0]
    v_layer = kvm._v_cache[0]
    for b in range(batch_size):
        block_idx, old_len = old_meta[b]
        gid = block_ids_list[b][-1]
        new_info = kvm._block_infos[gid]
        assert new_info.length == old_len + 1
        torch.testing.assert_close(
            k_layer[block_idx, :, old_len, :],
            key_new[b],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            v_layer[block_idx, :, old_len, :],
            value_new[b],
            rtol=0.0,
            atol=0.0,
        )
