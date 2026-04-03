import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module): 
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Initializes a linear layer with the given input and output features.
        :param in_features: The number of input features.
        :param out_features: The number of output features.
        :param device: The device on which to place the layer's parameters (optional).
        :param dtype: The data type of the layer's parameters (optional).
        """
        super().__init__()
        self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weights, mean=0, std=2./(in_features + out_features) ** 0.5, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the linear layer.
        :param x: The input tensor of shape (batch_size, in_features).
        :return: The output tensor of shape (batch_size, out_features).
        """
        return einsum(x, self.weights, "... in_features, out_features in_features -> ... out_features")
