import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Initializes the embedding layer.    
        Args:
            num_embeddings (int): The size of the vocabulary (number of unique tokens).
            embedding_dim (int): The dimensionality of the embedding vectors.
            device (torch.device | None): The device on which to place the embedding weights. If None, defaults to the current device.
            dtype (torch.dtype | None): The data type of the embedding weights. If None, defaults to torch.float32.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the embedding layer.
        Args:
            token_ids (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing the token IDs to be embedded.
        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, embedding_dim) containing the corresponding embedding vectors.
        """
        return self.weight[token_ids]
     
