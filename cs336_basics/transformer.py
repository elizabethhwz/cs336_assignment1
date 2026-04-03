import torch
import torch.nn as nn
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.multihead_self_attention import MultiheadSelfAttention   
from cs336_basics.swiglu import SwiGLU

class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int | None = None, theta: float | None = None, device: torch.device | None = None, token_positions: torch.Tensor | None = None):
        """
        Initializes the Transformer module.

        Args:
            d_model (int): The dimensionality of the input and output features.
            num_heads (int): The number of attention heads.
            d_ff (int): The dimensionality of the feedforward network.
            max_seq_len (int | None): The maximum sequence length that the model will be trained on.
            theta (float | None): The base frequency for the rotary positional embeddings.
            device (torch.device | None): The device to store the precomputed positional embeddings (optional).
            token_positions (torch.Tensor | None): The positions of the tokens in the sequence (optional).
        """
        super().__init__()
        self.RMSNorm1 = RMSNorm(d_model, device=device)
        self.MHA = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, token_positions=token_positions)
        self.RMSNorm2 = RMSNorm(d_model, device=device)
        self.SwiGLU = SwiGLU(d_model, d_ff, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the Transformer module for the input tensor `x`.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model). Assumes that the input is already embedded and has positional encodings applied.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, d_model) after applying the Transformer module.
        """
        x = x + self.MHA(self.RMSNorm1(x))
        x = x + self.SwiGLU(self.RMSNorm2(x))
        return x
