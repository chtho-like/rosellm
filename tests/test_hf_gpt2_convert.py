import torch
from transformers import GPT2Config, GPT2LMHeadModel

from rosellm.rosetrainer.hf_gpt2 import (
    convert_hf_gpt2_state_dict,
    gpt_config_from_hf_gpt2,
)
from rosellm.rosetrainer.model import GPTModel


def test_convert_hf_gpt2_state_dict_logits_match() -> None:
    torch.manual_seed(0)
    hf_cfg = GPT2Config(
        vocab_size=128,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        activation_function="gelu_new",
        use_cache=False,
    )
    hf = GPT2LMHeadModel(hf_cfg)
    hf.eval()

    cfg = gpt_config_from_hf_gpt2(hf_cfg)
    ours = GPTModel(cfg)
    ours.eval()

    mapped = convert_hf_gpt2_state_dict(hf.state_dict(), n_layers=cfg.n_layers)
    ours.load_state_dict(mapped, strict=True)

    input_ids = torch.randint(0, hf_cfg.vocab_size, (2, 16))
    with torch.no_grad():
        hf_logits = hf(input_ids).logits
        our_logits, _ = ours(
            input_ids, attention_mask=None, labels=None, use_cache=False
        )

    max_err = (hf_logits - our_logits).abs().max().item()
    assert max_err < 1e-4
