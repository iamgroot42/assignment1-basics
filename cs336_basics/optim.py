from collections.abc import Callable
from typing import Optional, Tuple
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 weight_decay: float = 0,
                 eps: float = 1e-8):

        defaults = {
            'lr': lr,
            'beta_1': betas[0],
            'beta_2': betas[1],
            'weight_decay': weight_decay,
            'eps': eps
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta_1 = group['beta_1']
            beta_2 = group['beta_2']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad.data

                m = state.get("m", torch.zeros_like(grad))
                v = state.get("v", torch.zeros_like(grad))
                t = state.get("t", 1)

                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * torch.pow(grad, 2)

                lr_adjusted = lr * math.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t)
                
                p.data -= lr_adjusted * m / (torch.sqrt(v) + eps)
                # Apply weight decay
                if weight_decay != 0:
                    p.data -= lr * weight_decay * grad
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss
