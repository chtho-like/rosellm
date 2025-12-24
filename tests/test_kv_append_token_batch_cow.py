import torch

from rosellm.roseinfer.engine import KVBlockManager


def test_append_token_batch_clones_on_shared_last_block() -> None:
    kvm = KVBlockManager(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        block_size=8,
        max_blocks_per_layer=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Build one base block with 3 tokens.
    block_ids: list[int] = []
    for t in range(3):
        x = float(t + 1)
        kvm.append_token(
            layer_idx=0,
            block_ids=block_ids,
            key_new=torch.full((2, 4), x, dtype=kvm.dtype),
            value_new=torch.full((2, 4), -x, dtype=kvm.dtype),
        )
    assert len(block_ids) == 1
    base_gid = int(block_ids[0])

    # Simulate: base_gid is held by a cache entry (ref=1) + 2 sessions (ref+=2).
    kvm.incref_blocks([base_gid, base_gid])
    assert kvm._block_refcounts[base_gid] == 3

    block_ids_list = [[base_gid], [base_gid]]
    key_new = torch.stack(
        [
            torch.full((2, 4), 10.0, dtype=kvm.dtype),
            torch.full((2, 4), 20.0, dtype=kvm.dtype),
        ],
        dim=0,
    )
    value_new = -key_new

    kvm.append_token_batch(
        layer_idx=0,
        block_ids_list=block_ids_list,
        key_new=key_new,
        value_new=value_new,
    )

    gid0 = int(block_ids_list[0][-1])
    gid1 = int(block_ids_list[1][-1])
    assert gid0 != base_gid
    assert gid1 != base_gid
    assert gid0 != gid1

    # After both sessions clone, the original block is only held by the cache.
    assert kvm._block_refcounts[base_gid] == 1

    base_info = kvm._block_infos[base_gid]
    assert base_info.length == 3
    base_k = kvm._k_cache[0, base_info.block_index]
    base_v = kvm._v_cache[0, base_info.block_index]

    for new_gid, appended in ((gid0, 10.0), (gid1, 20.0)):
        info = kvm._block_infos[new_gid]
        assert info.length == 4
        k = kvm._k_cache[0, info.block_index]
        v = kvm._v_cache[0, info.block_index]
        assert torch.allclose(k[:, :3, :], base_k[:, :3, :])
        assert torch.allclose(v[:, :3, :], base_v[:, :3, :])
        assert torch.allclose(
            k[:, 3, :],
            torch.full((2, 4), appended, dtype=kvm.dtype),
        )
        assert torch.allclose(
            v[:, 3, :],
            torch.full((2, 4), -appended, dtype=kvm.dtype),
        )

    kvm.free_blocks(0, [gid0])
    kvm.free_blocks(0, [gid1])
