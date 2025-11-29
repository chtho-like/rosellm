import argparse

import torch
from checkpoint import load_checkpoint
from config import GPTConfig
from dataset import build_tokenizer
from model import GPTModel


def top_k_logits(
    logits: torch.Tensor,  # [..., vocab]
    k: int,
) -> torch.Tensor:
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)  # values.shape: [..., k]
    min_values = values[..., -1, None]  # min_values.shape: [..., 1]
    return torch.where(
        logits < min_values,
        torch.full_like(logits, float("-inf")),
        logits,
    )


def top_p_logits(
    logits: torch.Tensor,  # [..., vocab]
    top_p: float,
) -> torch.Tensor:
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
    )
    probs = torch.softmax(sorted_logits, dim=-1)  # [..., vocab]
    cum_probs = torch.cumsum(probs, dim=-1)  # [..., vocab]
    mask = cum_probs > top_p  # [..., vocab]
    mask[..., 0] = False  # keep at least one token
    masked_logits = sorted_logits.masked_fill(
        mask,
        float("-inf"),
    )
    _, original_indices = torch.sort(
        sorted_indices,
        dim=-1,
    )
    return torch.gather(masked_logits, -1, original_indices)


@torch.no_grad()
def generate(
    model: GPTModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    do_sample: bool = False,
    device: torch.device = torch.device("cpu"),
) -> str:
    model.eval()
    enc = tokenizer.encode(prompt, add_special_tokens=False)
    if len(enc) == 0:
        enc = [tokenizer.eos_token_id]
    input_ids = torch.tensor(
        [enc],
        dtype=torch.long,
        device=device,
    )
    max_pos = model.config.max_position_embeddings
    if input_ids.size(1) >= max_pos:
        input_ids = input_ids[:, -max_pos + 1 :]
    for _ in range(max_new_tokens):
        logits, _ = model(input_ids)
        next_logits = logits[:, -1, :]
        if not do_sample or temperature <= 0.0:
            next_token = torch.argmax(next_logits, dim=-1)
        else:
            logits_scaled = next_logits / temperature
            logits_filtered = logits_scaled
            if top_k > 0:
                logits_filtered = top_k_logits(logits_filtered, top_k)
            if top_p > 0.0:
                logits_filtered = top_p_logits(logits_filtered, top_p)
            probs = torch.softmax(logits_filtered, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)[:, 0]
        input_ids = torch.cat(
            [input_ids, next_token.view(1, 1)],
            dim=1,
        )
        next_token_id = next_token.item()
        if (
            hasattr(tokenizer, "eos_token_id")
            and next_token_id == tokenizer.eos_token_id
        ):
            break
        if input_ids.size(1) > max_pos:
            input_ids = input_ids[:, -max_pos:]
    output_ids = input_ids[0].tolist()
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return text


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print(f"Using device: {device}")
    tokenizer = build_tokenizer(args.tokenizer_name)
    ckpt = torch.load(args.checkpoint_path, map_location=device.type)
    cfg_dict = ckpt.get("config")
    if cfg_dict is not None:
        print("Found config in checkpoint, ignore cli configs")
        config = GPTConfig(**cfg_dict)
    else:
        print("No config found in checkpoint, use cli configs")
        config = GPTConfig(
            vocab_size=args.vocab_size,
            max_position_embeddings=args.max_position_embeddings,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_model=args.d_model,
            d_ff=args.d_ff,
            dropout=args.dropout,
            use_tensor_parallel=args.use_tensor_parallel,
        )
    model = GPTModel(config).to(device)
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    load_checkpoint(
        args.checkpoint_path,
        model=model,
        optimizer=None,
        scaler=None,
        map_location=device.type,
    )
    print(f"Prompt: {args.prompt}")
    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        device=device,
    )
    print(f"Generated text: {text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a model.")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Vocabulary size.",
    )
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=10000,
        help="Max sequence length.",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=2,
        help="Number of Transformer layers.",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Model hidden size.",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=512,
        help="FFN hidden size.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability.",
    )
    parser.add_argument(
        "--use-tensor-parallel",
        action="store_true",
        help="Enable tensor parallel blocks.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to checkpoint file.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="gpt2",
        help="Tokenizer name.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        default="Hello, ",
        help="Prompt to generate text from.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="Top-p sampling.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling to generate text (or else greedy).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
