from cs336_basics.softmax import softmax_temperature
import torch
from jaxtyping import Float, Int
from torch import Tensor

def top_p_sampling(
    logits: Float[Tensor, "batch_size ... sequence_length vocab_size"],
    p: float,
    temperature: float = 1.0,
) -> Int[Tensor, "batch_size ... sequence_length"]:
    """
    Perform top-p (nucleus) sampling on the given logits.

    Args:
        logits: A tensor of shape (batch_size, ..., sequence_length, vocab_size)
            containing the unnormalized log probabilities for each token.
        p: The cumulative probability threshold for top-p sampling. Must be in the range (0, 1].
        temperature: The temperature for scaling the logits. Must be greater than 0.
    Returns:
        A tensor of sampled token indices with shape (batch_size, ..., sequence_length).
    """
    if not (0 < p <= 1):
        raise ValueError("p must be in the range (0, 1].")
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    if logits.ndim < 1:
        raise ValueError("logits must have at least one dimension.")

    probs = softmax_temperature(logits, temperature=temperature, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus_mask = cumulative_probs <= p
    nucleus_mask[..., 0] = True

    filtered_probs = torch.where(nucleus_mask, sorted_probs, torch.zeros_like(sorted_probs))
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    vocab_size = filtered_probs.shape[-1]
    flat_probs = filtered_probs.reshape(-1, vocab_size)
    sampled_index = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)

    flat_sorted_indices = sorted_indices.reshape(-1, vocab_size)
    sampled_tokens = flat_sorted_indices.gather(-1, sampled_index.unsqueeze(-1)).squeeze(-1)

    return sampled_tokens.reshape(logits.shape[:-1])
