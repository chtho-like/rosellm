import argparse
import time

import torch

from rosellm.roseinfer.engine import KVBlockManager


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--old-len", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this microbench")

    # Use GPT-2 small-ish shapes by default.
    num_layers = 12
    num_heads = 12
    head_dim = 64
    block_size = 64
    # Keep this small: KVBlockManager preallocates a dense tensor of
    # [L, max_blocks_per_layer, H, block_size, D].
    max_blocks_per_layer = 64

    kvm = KVBlockManager(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_blocks_per_layer=max_blocks_per_layer,
        device=torch.device("cuda"),
        dtype=torch.float16,
    )

    old_len = int(args.old_len)
    if old_len <= 0 or old_len >= block_size:
        raise ValueError(f"--old-len must be in [1,{block_size - 1}]")

    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    base_gids: list[int] = []
    for layer_idx in range(num_layers):
        block_ids: list[int] = []
        for t in range(old_len):
            x = float(t + 1)
            kvm.append_token(
                layer_idx=layer_idx,
                block_ids=block_ids,
                key_new=torch.full(
                    (num_heads, head_dim), x, device="cuda", dtype=kvm.dtype
                ),
                value_new=torch.full(
                    (num_heads, head_dim), -x, device="cuda", dtype=kvm.dtype
                ),
            )
        assert len(block_ids) == 1
        base_gids.append(block_ids[0])

    key_new = torch.ones(
        (batch_size, num_heads, head_dim),
        device="cuda",
        dtype=kvm.dtype,
    )
    value_new = torch.zeros_like(key_new)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(int(args.iters)):
        for layer_idx, base_gid in enumerate(base_gids):
            kvm.incref_blocks([base_gid] * batch_size)  # simulate cache + sessions
            block_ids_list = [[base_gid] for _ in range(batch_size)]
            kvm.append_token_batch(
                layer_idx=layer_idx,
                key_new=key_new,
                value_new=value_new,
                block_ids_list=block_ids_list,
            )
            for ids in block_ids_list:
                kvm.free_blocks(layer_idx, ids)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    total = t1 - t0
    n_cow = int(args.iters) * num_layers * batch_size
    print("=== kv COW clone microbench ===")
    print(
        f"old_len: {old_len} batch: {batch_size} iters: {int(args.iters)} "
        f"layers: {num_layers}"
    )
    print(f"total: {total*1e3:.2f} ms")
    print(f"avg per COW (clone+append+free): {total / n_cow * 1e6:.2f} us")


if __name__ == "__main__":
    main()
