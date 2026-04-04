import numpy.typing as npt
import torch
from torch import Tensor
from jaxtyping import Int
import numpy as np

def data_loader(x: npt.NDArray, batch_size: int, context_length: int, device: str = 'mps') -> tuple[Int[Tensor, 'batch_size context_length'], Int[Tensor, 'batch_size context_length']]:
    """
    A simple data loader that yields batches of input and target sequences for training a language model.

    Args:
        x (np.ndarray): The input data as a 1D array of token indices.
        batch_size (int): The number of sequences in each batch.
        context_length (int): The length of the input sequences (context).
        device (str): The device to which the tensors will be moved ('cpu' or 'cuda').
    Returns:
        A tuple of two tensors: (input_batch, target_batch), where:
        - input_batch (Tensor[Int, 'batch_size context_length']): A tensor containing the input sequences.
        - target_batch (Tensor[Int, 'batch_size context_length']): A tensor containing the target sequences.
    """
    random_indices = torch.randint(0, len(x) - context_length, (batch_size,))
    input_batch = torch.tensor(np.stack([x[i:i+context_length] for i in random_indices]), dtype=torch.long, device=device)
    target_batch = torch.tensor(np.stack([x[i+1:i+context_length+1] for i in random_indices]), dtype=torch.long, device=device)
    return input_batch, target_batch
