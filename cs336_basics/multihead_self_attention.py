import torch
import torch.nn as nn
from cs336_basics.linear import Linear
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.rope import RotaryPositionalEmbedding

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None, device: torch.device | None = None, token_positions: torch.Tensor | None = None):
        """
        Initializes the Multihead Self-Attention module.

        Args:
            d_model (int): The dimensionality of the input and output features.
            num_heads (int): The number of attention heads.
            theta (float | None): The base frequency for the rotary positional embeddings.
            max_seq_len (int | None): The maximum sequence length that the model will be trained on.
            device (torch.device | None): The device to store the precomputed positional embeddings (optional).
            token_positions (torch.Tensor | None): The positions of the tokens in the sequence (optional).
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert self.d_k * num_heads == d_model, f"d_model must be divisible by num_heads, but got d_model={d_model} and num_heads={num_heads}"
        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
        if token_positions is not None:
            self.register_buffer("token_positions", token_positions, persistent=False)
            
        self.W_Q = Linear(d_model, d_model, device=device)
        self.W_K = Linear(d_model, d_model, device=device)
        self.W_V = Linear(d_model, d_model, device=device)
        self.W_O = Linear(d_model, d_model, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the multihead self-attention output for the input tensor `x`.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model). Assumes that the input is already embedded and has positional encodings applied. 
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, d_model) after applying multihead self-attention.
        """
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        *leading_dims, sequence_length, _ = x.shape
        Q = Q.reshape(*leading_dims, sequence_length, self.num_heads, self.d_k).transpose(-3, -2)
        K = K.reshape(*leading_dims, sequence_length, self.num_heads, self.d_k).transpose(-3, -2)
        V = V.reshape(*leading_dims, sequence_length, self.num_heads, self.d_k).transpose(-3, -2)

        if hasattr(self, 'rope'):
            if hasattr(self, "token_positions"):
                token_positions = self.token_positions
            else:
                token_positions = torch.arange(sequence_length, device=x.device).expand(*leading_dims, sequence_length)
            if token_positions.shape == (*leading_dims, sequence_length):
                token_positions = token_positions.unsqueeze(-2).expand(*leading_dims, self.num_heads, sequence_length)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.tril(
            torch.ones(*leading_dims, self.num_heads, sequence_length, sequence_length, dtype=torch.bool, device=x.device)
        )

        attention_output = scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(-3, -2).reshape(*leading_dims, sequence_length, self.d_model)
        output = self.W_O(attention_output)
        return output
