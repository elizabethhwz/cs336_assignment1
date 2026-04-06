import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy_loss(
    logits: Float[Tensor, "batch_size ... sequence_length vocab_size"],
    targets: Int[Tensor, "batch_size ... sequence_length"],
) -> Float[Tensor, ""]:
    """
    Compute the cross-entropy loss between the predicted logits and the true targets.

    Args:
        logits: A tensor of shape (batch_size, ..., sequence_length, vocab_size) containing the predicted logits for each class.
        targets: A tensor of shape (batch_size, ..., sequence_length) containing the true class indices. Each value should be in the range [0, vocab_size - 1].
    Returns:
        A scalar tensor representing the average cross-entropy loss over the batch.
    """
    log_normalizer = torch.logsumexp(logits, dim=-1)
    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - target_logits).mean()
