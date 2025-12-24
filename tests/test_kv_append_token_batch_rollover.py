import torch

from rosellm.roseinfer.engine import KVBlockManager


def test_append_token_batch_rollover_allocates_new_block() -> None:
    kvm = KVBlockManager(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        block_size=4,
        max_blocks_per_layer=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Build a full block (length == block_size).
    base_ids: list[int] = []
    for t in range(kvm.block_size):
        x = float(t + 1)
        kvm.append_token(
            layer_idx=0,
            block_ids=base_ids,
            key_new=torch.full((2, 4), x, dtype=kvm.dtype),
            value_new=torch.full((2, 4), -x, dtype=kvm.dtype),
        )
    assert len(base_ids) == 1
    base_gid = int(base_ids[0])
    base_info = kvm._block_infos[base_gid]
    assert base_info.length == kvm.block_size

    # Two sessions share the same full block (ref=2).
    kvm.incref_blocks([base_gid])
    assert kvm._block_refcounts[base_gid] == 2
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

    # Each session gets a new block appended (no COW on the full block).
    assert block_ids_list[0][0] == base_gid
    assert block_ids_list[1][0] == base_gid
    assert len(block_ids_list[0]) == 2
    assert len(block_ids_list[1]) == 2
    gid0 = int(block_ids_list[0][1])
    gid1 = int(block_ids_list[1][1])
    assert gid0 != base_gid and gid1 != base_gid and gid0 != gid1

    # Base refcount unchanged (still shared).
    assert kvm._block_refcounts[base_gid] == 2

    # New blocks contain the appended token at position 0.
    for gid, appended in ((gid0, 10.0), (gid1, 20.0)):
        info = kvm._block_infos[gid]
        assert info.length == 1
        assert info.start == base_info.start + base_info.length
        k = kvm._k_cache[0, info.block_index]
        v = kvm._v_cache[0, info.block_index]
        assert torch.allclose(
            k[:, 0, :],
            torch.full((2, 4), appended, dtype=kvm.dtype),
        )
        assert torch.allclose(
            v[:, 0, :],
            torch.full((2, 4), -appended, dtype=kvm.dtype),
        )

    kvm.free_blocks(0, [gid0])
    kvm.free_blocks(0, [gid1])
