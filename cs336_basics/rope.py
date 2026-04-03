import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        """
        Args:
            theta: The base frequency for the rotary positional embeddings
            d_k: The dimensionality of the key/query vectors in the attention mechanism
            max_seq_len: The maximum sequence length that the model will be trained on
            device: The device to store the precomputed positional embeddings (optional)
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        k = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        inv_freq = theta ** (-k / d_k)
        self.register_buffer("inv_freq", inv_freq, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_k)
            token_positions: Tensor of shape (batch_size, seq_len) containing the positions of each token in the sequence
        Returns:
            Tensor of shape (batch_size, seq_len, d_k) with the rotary positional embeddings applied
        """
        assert x.shape[-1] == self.d_k, f"Input tensor's last dimension must be {self.d_k}, but got {x.shape[-1]}"
        assert x.shape[-2] == token_positions.shape[-1], f"Input tensor's sequence length must match token_positions' sequence length, but got {x.shape[-2]} and {token_positions.shape[-1]}"
        assert token_positions.max() < self.max_seq_len, f"Token positions must be less than max_seq_len ({self.max_seq_len}), but got {token_positions.max()}"
        assert x.shape[-1] % 2 == 0, f"Input tensor's last dimension must be even, but got {x.shape[-1]}"

        angles = token_positions.unsqueeze(-1).to(x.dtype) * self.inv_freq.to(x.dtype)
        cos_angles = torch.cos(angles)  # (batch_size, seq_len, d_k // 2)
        sin_angles = torch.sin(angles)  # (batch_size, seq_len, d_k // 2)
        x_even, x_odd = x[..., ::2], x[..., 1::2]  # (batch_size, seq_len, d_k // 2) each
        rotated_even = x_even * cos_angles - x_odd * sin_angles  # (batch_size, seq_len, d_k // 2)
        rotated_odd = x_even * sin_angles + x_odd * cos_angles  # (batch_size, seq_len, d_k // 2)
        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = rotated_even
        x_rotated[..., 1::2] = rotated_odd
        return x_rotated

