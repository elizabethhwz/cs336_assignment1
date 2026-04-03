import torch

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute the softmax of the input tensor along the specified dimension.
    """
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_max = torch.where(torch.isfinite(x_max), x_max, torch.zeros_like(x_max))
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return torch.where(x_sum > 0, x_exp / x_sum, torch.zeros_like(x_exp))
