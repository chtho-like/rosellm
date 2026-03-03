from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rosellm.rosetrainer.hf_deepseek_v2_fp8 import load_deepseek_v2_fp8_from_hf_pretrained


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=local_rank)
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")

        model_id = "gaunernst/DeepSeek-V2-Lite-Chat-FP8"
        model, _gpt_cfg, tok, stats = load_deepseek_v2_fp8_from_hf_pretrained(
            model_id,
            device=device,
            dtype=torch.bfloat16,
            ep_size=world,
            verbose=(rank == 0),
        )
        if rank == 0:
            print("[smoke] load stats:", stats)

        if rank == 0:
            prompt = "Hello! Please answer briefly: what is 2+2?"
            inputs = tok(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        else:
            input_ids = torch.tensor([[int(tok.eos_token_id)]], device=device, dtype=torch.long)
            attention_mask = None

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        if rank == 0:
            next_id = int(out.logits[0, -1].argmax().item())
            print("[smoke] next token:", next_id, tok.decode([next_id]))
        dist.barrier()
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
