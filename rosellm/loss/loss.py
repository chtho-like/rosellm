import torch
import torch.nn as nn


def fixed_cross_entropy(
    source,
    target,
    num_items_in_batch: int = 1,
    ignore_index: int = -100,
    **kwargs,
):
    reduction = "sum" if num_items_in_batch != 1 else "mean"
    loss = nn.functional.cross_entropy(
        source,
        target,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def causal_lm_loss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: int = 1,
    ignore_index: int = -100,
    **kwargs,
):
    logits = logits.float()
    labels = labels.to(logits.device)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    shift_labels = shift_labels.to(shift_logits.device)
    loss = fixed_cross_entropy(
        shift_logits,
        shift_labels,
        num_items_in_batch,
        ignore_index,
        **kwargs,
    )
    return loss


if __name__ == "__main__":
    print("Example: ForCausalLMLoss with sequence data")
    batch_size = 2
    seq_length = 10
    vocab_size = 100
    causal_logits = torch.randn(batch_size, seq_length, vocab_size)
    print("causal_logits.shape", causal_logits.shape)
    causal_labels = torch.randint(0, vocab_size, (batch_size, seq_length))
    print("causal_labels.shape", causal_labels.shape)

    causal_loss = causal_lm_loss(causal_logits, causal_labels, vocab_size)
    print(f"Causal language modeling loss: {causal_loss.item():.4f}")
