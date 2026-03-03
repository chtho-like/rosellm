import argparse
import math

import torch


def _torch_sdpa_auto(q, k, v, causal: bool, scale: float):
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    return sdpa(q, k, v, is_causal=causal, scale=scale)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="v9")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--causal", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--init", type=str, default="randn")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    causal = bool(args.causal)
    scale = 1.0 / math.sqrt(args.head_dim)

    shape = (args.batch, args.heads, args.seq_len, args.head_dim)
    if args.init == "empty":
        q = torch.empty(shape, device="cuda", dtype=dtype)
        k = torch.empty_like(q)
        v = torch.empty_like(q)
    elif args.init == "randn":
        q = torch.randn(shape, device="cuda", dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
    else:
        raise ValueError("--init must be 'randn' or 'empty'")

    if args.backend in ("v6", "v7", "v8", "v9"):
        mod_name = {
            "v6": "triton_flashattn_6_autotune_cutoff",
            "v7": "triton_flashattn_7_fixed_best_cutoff",
            "v8": "triton_flashattn_8_dispatch_fixed_best",
            "v9": "triton_flashattn_9_autotune_table",
        }[args.backend]
        mod = __import__(mod_name)

        def fn():
            mod.flash_attn(q, k, v, causal=causal, sm_scale=scale)

    elif args.backend in ("torch_sdpa", "torch_sdpa_auto"):
        def fn():
            _torch_sdpa_auto(q, k, v, causal, scale)

    elif args.backend in (
        "torch_sdpa_flash_pytorch",
        "torch_sdpa_efficient_cutlass",
        "torch_sdpa_cudnn",
    ):
        mod = __import__(args.backend)

        def fn():
            mod.flash_attn(q, k, v, causal=causal, sm_scale=scale)

    elif args.backend == "flash_attn":
        from flash_attn.flash_attn_interface import flash_attn_func

        q_bshd = q.transpose(1, 2).contiguous()
        k_bshd = k.transpose(1, 2).contiguous()
        v_bshd = v.transpose(1, 2).contiguous()

        def fn():
            flash_attn_func(
                q_bshd,
                k_bshd,
                v_bshd,
                dropout_p=0.0,
                softmax_scale=scale,
                causal=causal,
            )

    else:
        raise ValueError(f"unknown backend: {args.backend}")

    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()

    fn()
    torch.cuda.synchronize()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

