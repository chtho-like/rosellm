import argparse
import random
import time

import torch

from rosellm.roseinfer.engine import KVBlockManager, PrefixCache


class _DummySession:
    def __init__(
        self,
        *,
        kv_manager: KVBlockManager,
        prompt_length: int,
    ) -> None:
        self.kv_manager = kv_manager
        self.prompt_length = int(prompt_length)
        self.block_ids_per_layer = [[] for _ in range(int(kv_manager.num_layers))]


def _build_cache(
    *,
    num_entries: int,
    key_len: int,
    block_size: int,
    seed: int,
) -> tuple[PrefixCache, list[tuple[int, ...]]]:
    rng = random.Random(seed)
    kv = KVBlockManager(
        num_layers=1,
        num_heads=1,
        head_dim=1,
        block_size=int(block_size),
        max_blocks_per_layer=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    cache = PrefixCache(
        kv_manager=kv,
        max_entries=0,  # unbounded
    )
    dummy = _DummySession(
        kv_manager=kv,
        prompt_length=key_len,
    )
    last_logits = torch.zeros((1, 1), dtype=torch.float32)

    keys: list[tuple[int, ...]] = []
    for _ in range(int(num_entries)):
        k = tuple(rng.randrange(0, 50257) for _ in range(int(key_len)))
        keys.append(k)
        cache.put(k, dummy, last_logits)
    return cache, keys


def _bench(
    *,
    cache: PrefixCache,
    query: tuple[int, ...],
    iters: int,
) -> float:
    t0 = time.perf_counter()
    for _ in range(int(iters)):
        cache.find_longest_token_prefix(query)
    t1 = time.perf_counter()
    return (t1 - t0) / float(iters)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-entries", type=int, default=2048)
    p.add_argument("--key-len", type=int, default=512)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cache, keys = _build_cache(
        num_entries=args.num_entries,
        key_len=args.key_len,
        block_size=args.block_size,
        seed=args.seed,
    )

    miss_query = keys[0][:-1] + (999_999,)
    miss_s = _bench(
        cache=cache,
        query=miss_query,
        iters=args.iters,
    )
    hit_query = keys[0] + (999_999,)
    hit_s = _bench(
        cache=cache,
        query=hit_query,
        iters=args.iters,
    )

    print("=== prefix cache lookup microbench ===")
    print(f"entries: {int(args.num_entries)} key_len: {int(args.key_len)}")
    print(f"miss: {miss_s * 1e6:.2f} us/lookup")
    print(f"hit : {hit_s * 1e6:.2f} us/lookup")


if __name__ == "__main__":
    main()
