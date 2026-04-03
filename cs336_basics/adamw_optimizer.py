import torch
from typing import Optional
from collections.abc import Callable


class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)

                state["t"] += 1
                t = state["t"]
                m = state["m"]
                v = state["v"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                lr_t = lr * ((1 - beta2 ** t) ** 0.5) / (1 - beta1 ** t)

                param.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)
                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)
        return loss
