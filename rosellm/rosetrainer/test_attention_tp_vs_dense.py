import os

import torch
import torch.distributed as dist
import torch.nn as nn
from config import GPTConfig
from model import MultiHeadSelfAttention
from tensor_parallel import init_tensor_parallel


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return device, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def build_attention(config_base: GPTConfig, device: torch.device):
    dense_cfg = GPTConfig(
        vocab_size=config_base.vocab_size,
        max_position_embeddings=config_base.max_position_embeddings,
        n_layers=config_base.n_layers,
        n_heads=config_base.n_heads,
        d_model=config_base.d_model,
        d_ff=config_base.d_ff,
        dropout=config_base.dropout,
        use_tensor_parallel=False,
    )
    attn_dense = MultiHeadSelfAttention(dense_cfg).to(device)
    tp_cfg = GPTConfig(
        vocab_size=config_base.vocab_size,
        max_position_embeddings=config_base.max_position_embeddings,
        n_layers=config_base.n_layers,
        n_heads=config_base.n_heads,
        d_model=config_base.d_model,
        d_ff=config_base.d_ff,
        dropout=config_base.dropout,
        use_tensor_parallel=True,
    )
    attn_tp = MultiHeadSelfAttention(tp_cfg).to(device)
    return attn_dense, attn_tp


def copy_qkv_from_dense_to_tp(
    attn_dense: MultiHeadSelfAttention,
    attn_tp: MultiHeadSelfAttention,
    world_size: int,
    rank: int,
):
    with torch.no_grad():
        linear_dense: nn.Linear = attn_dense.qkv_proj
        col_tp = attn_tp.qkv_proj

        d_model = attn_dense.d_model
        n_heads = attn_dense.n_heads
        d_head = attn_dense.d_head
        assert d_model == n_heads * d_head

        local_heads = n_heads // world_size
        local_dim = local_heads * d_head
        head_start = rank * local_heads
        head_end = head_start + local_heads

        q_offset = 0
        k_offset = d_model
        v_offset = 2 * d_model

        q_start = q_offset + head_start * d_head
        q_end = q_offset + head_end * d_head
        k_start = k_offset + head_start * d_head
        k_end = k_offset + head_end * d_head
        v_start = v_offset + head_start * d_head
        v_end = v_offset + head_end * d_head

        q_weight = linear_dense.weight[q_start:q_end, :]
        k_weight = linear_dense.weight[k_start:k_end, :]
        v_weight = linear_dense.weight[v_start:v_end, :]

        col_tp.weight[:local_dim, :].copy_(q_weight)
        col_tp.weight[local_dim : 2 * local_dim, :].copy_(k_weight)
        col_tp.weight[2 * local_dim : 3 * local_dim, :].copy_(v_weight)

        if col_tp.bias is not None:
            q_bias = linear_dense.bias[q_start:q_end]
            k_bias = linear_dense.bias[k_start:k_end]
            v_bias = linear_dense.bias[v_start:v_end]
            col_tp.bias.copy_(torch.cat([q_bias, k_bias, v_bias], dim=0))


def copy_out_proj_from_dense_to_tp(
    attn_dense: MultiHeadSelfAttention,
    attn_tp: MultiHeadSelfAttention,
    world_size: int,
    rank: int,
):
    with torch.no_grad():
        linear_dense: nn.Linear = attn_dense.out_proj
        row_tp = attn_tp.out_proj
        in_features = linear_dense.in_features
        in_per_rank = in_features // world_size
        start = rank * in_per_rank
        end = start + in_per_rank
        row_tp.weight.copy_(linear_dense.weight[:, start:end])
        if row_tp.bias is not None:
            row_tp.bias.copy_(linear_dense.bias)


def main():
    device, local_rank = setup_distributed()
    init_tensor_parallel()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank == 0:
        print(f"world_size = {world_size}")
    base_cfg = GPTConfig(
        vocab_size=10000,
        max_position_embeddings=128,
        n_layers=1,
        n_heads=4,
        d_model=64,
        d_ff=256,
        dropout=0.0,
    )
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    attn_dense, attn_tp = build_attention(base_cfg, device)
    copy_qkv_from_dense_to_tp(attn_dense, attn_tp, world_size, rank)
    copy_out_proj_from_dense_to_tp(attn_dense, attn_tp, world_size, rank)
    batch_size = 2
    seq_len = 8
    x = torch.randn(batch_size, seq_len, base_cfg.d_model, device=device)
    attention_mask = torch.ones(
        batch_size,
        seq_len,
        dtype=torch.long,
        device=device,
    )
    attn_dense.eval()
    attn_tp.eval()
    with torch.no_grad():
        y_dense = attn_dense(x, attention_mask=attention_mask)
        y_tp = attn_tp(x, attention_mask=attention_mask)
    diff = (y_dense - y_tp).abs().max()
    diff_val = diff.item()
    if rank == 0:
        print("y_dense shape:", y_dense.shape)
        print("y_tp shape:", y_tp.shape)
        print("max |y_dense - y_tp| = ", diff_val)
    cleanup_distributed()


if __name__ == "__main__":
    main()
