import torch
import torch.nn as nn
from cs336_basics.linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Initialize the SwiGLU layer.

        Args:
            d_model (int): The dimensionality of the input and output.
            d_ff (int): The dimensionality of the feedforward layer.
            device (torch.device | None): The device to initialize the layer on.
            dtype (torch.dtype | None): The data type to initialize the layer with.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)

    def SiLU(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the SiLU activation function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying SiLU.
        """
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SwiGLU layer.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Output tensor of shape (..., d_model).
        """
        x1 = self.linear1(x)
        x3 = self.linear3(x)
        return self.linear2(self.SiLU(x1) * x3)
