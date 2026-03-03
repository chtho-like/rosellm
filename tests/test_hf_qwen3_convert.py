import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from rosellm.rosetrainer.hf_qwen3 import gpt_config_from_hf_qwen3
from rosellm.rosetrainer.qwen3 import Qwen3ForCausalLM as RoseQwen3ForCausalLM


def test_qwen3_gpt_config_matches_kv_head_dim() -> None:
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        head_dim=8,
        attention_dropout=0.0,
        attention_bias=False,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=1000000,
        tie_word_embeddings=True,
        use_cache=False,
    )
    ours = gpt_config_from_hf_qwen3(cfg)
    assert ours.n_layers == cfg.num_hidden_layers
    assert ours.n_heads == cfg.num_key_value_heads
    assert ours.d_model // ours.n_heads == cfg.head_dim


def test_qwen3_state_dict_logits_match() -> None:
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        head_dim=8,
        attention_dropout=0.0,
        attention_bias=False,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=1000000,
        tie_word_embeddings=True,
        use_cache=False,
    )
    hf = Qwen3ForCausalLM(cfg)
    hf.eval()

    ours = RoseQwen3ForCausalLM(cfg)
    ours.eval()
    ours.load_state_dict(hf.state_dict(), strict=True)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        hf_logits = hf(input_ids).logits
        our_logits, _ = ours(input_ids, attention_mask=None, labels=None, use_cache=False)

    max_err = (hf_logits - our_logits).abs().max().item()
    assert max_err < 1e-4
