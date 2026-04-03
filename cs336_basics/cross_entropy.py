from cs336_basics.softmax import softmax
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
    softmax_logits = softmax(logits)
    log_softmax_logits = torch.log(softmax_logits + 1e-9)  # Add a small constant for numerical stability
    return -torch.gather(log_softmax_logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1).mean()
