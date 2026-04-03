import torch
from typing import Iterable

@torch.no_grad()
def gradient_clipping(gradients: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6):
    """
    Clips the gradients to have a maximum L2 norm of max_norm.
    Args:
        gradients: A collection of trainable parameters for which to clip the gradients.
        max_norm: The maximum allowed L2 norm of the gradients.
    Returns:
        A list of clipped gradients.
    """
    parameters = list(gradients)
    total_norm = torch.sqrt(
        sum(param.grad.pow(2).sum() for param in parameters if param.grad is not None)
    )
    if total_norm > max_norm:
        factor = max_norm / (total_norm + eps)
        for param in parameters:
            if param.grad is None:
                continue
            param.grad.mul_(factor)
    return gradients
