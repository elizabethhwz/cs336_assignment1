import torch
from cs336_basics.softmax import softmax

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the scaled dot product attention.

    Args:
        query: Float[Tensor, "batch_size ... sequence_length d_k"]: The query tensor.
        key: Float[Tensor, "batch_size ... sequence_length d_k"]: The key tensor.
        value: Float[Tensor, "batch_size ... sequence_length d_v"]: The value tensor.
        mask: Bool[Tensor, "batch_size ... sequence_length sequence_length"]]: A boolean mask tensor. `True` entries are kept and `False` entries are masked out before softmax.

    Returns:
        Float[Tensor, "batch_size ... sequence_length d_v"]: The output of the attention mechanism.
    """

    dk = query.shape[-1]
    scale = query.new_tensor(dk).sqrt()
    scores = torch.einsum("... i k, ... j k -> ... i j", query, key) / scale
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attention_weights = softmax(scores, dim=-1)
    return torch.einsum("... i j, ... j v -> ... i v", attention_weights, value)


    

