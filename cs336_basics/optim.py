from collections.abc import Callable
from typing import Optional
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3,
                beta_1: float = 0.9,
                beta_2: float = 0.999,
                weight_decay: float = 0,
                epsilon: float = 1e-8):

        defaults = {
            'lr': lr,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'weight_decay': weight_decay,
            'epsilon': epsilon
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta_1 = group['beta_1']
            beta_2 = group['beta_2']
            epsilon = group['epsilon']
            weight_decay = group['weight_decay']
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad.data

                m = state.get("m", torch.zeors_like(grad))
                v = state.get("v", torch.zeors_like(grad))
                t = state.get("t", 1)

                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * torch.pow(grad, 2)

                lr_adjusted = lr * torch.sqrt(1 - torch.pow(beta_2, t)) / (1 - torch.pow(beta_1, t))
                
                p.data -= lr_adjusted * m / (torch.sqrt(v) + epsilon)
                # Apply weight decay
                if weight_decay != 0:
                    p.data -= lr * weight_decay * grad
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss
