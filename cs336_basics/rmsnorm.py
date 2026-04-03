import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Initializes the RMSNorm layer.
        Args:
            d_model (int): The dimensionality of the input and output vectors.
            eps (float): A small value added to the denominator for numerical stability. Default is 1e-5.
            device (torch.device | None): The device on which to place the parameters. If None, defaults to the current device.
            dtype (torch.dtype | None): The data type of the parameters. If None, defaults to torch.float32.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the RMSNorm layer.
        Args:
            x (torch.Tensor): A tensor of shape (batch_size, sequence_length, d_model) containing the input vectors.
        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, d_model) containing the normalized output vectors.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        x = x / torch.sqrt(mean_square + self.eps)
        result = x * self.weight.to(torch.float32)
        return result.to(in_dtype)



