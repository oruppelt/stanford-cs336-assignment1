
import torch
import torch.nn as nn
from collections.abc import Callable, Iterable
from typing import Optional
import math

import einx
from einops import rearrange, einsum, reduce
from transformer_blocks import softmax

def cross_entropy(oi, xi):

    batch_shape = oi.shape[:-1]  # All dimensions except vocab_size
    vocab_size = oi.shape[-1]

    oi_max = oi.max(dim=1, keepdim=True)[0]

    oi_shifted = oi - oi_max

    logZ = torch.log(torch.sum(torch.exp(oi_shifted), dim=-1))

    # target_logits = oi_shifted[torch.arange(oi_shifted.shape[0]), xi]

    # I do not fully understand how torch.gather works
    # but it says it is better to do it this way

    target_logits = torch.gather(oi_shifted, dim=-1, index=xi.unsqueeze(-1)).squeeze(-1)

    loss = (logZ - target_logits).mean()

    return loss

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if betas[1] < 0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0

                    state["exp_avg"] = torch.zeros_like(p.data)

                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                grad = p.grad.data

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                lr_t = lr * math.sqrt(1 - beta2 ** state["step"]) / (1 - beta1 ** state["step"])

                p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-lr_t)

                if weight_decay > 0:
                    p.data.mul_(1 - lr * weight_decay)

        return loss

def cos_annealing(t, alpha_max, alpha_min, Tw, Tc):

    lr = None

    if t < Tw:
        lr = alpha_max * (t / Tw)

    elif t <= Tc:
        annealing = 1 + math.cos(math.pi * (t - Tw) / (Tc - Tw))
        lr = alpha_min + 0.5 * annealing * (alpha_max - alpha_min)

    else:
        lr = alpha_min

    return lr

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):

    eps = 1e-6
    param_list = list(parameters)

    total_norm_squared = 0.0
    for param in param_list:
        if param.grad is not None:
            param_norm_squared = torch.sum(param.grad.data ** 2)
            total_norm_squared += param_norm_squared

    total_norm = torch.sqrt(total_norm_squared)

    if total_norm > max_l2_norm:
        # Compute clipping factor
        clip_factor = max_l2_norm / (total_norm + eps)

        for param in param_list:
            if param.grad is not None:
                param.grad.data.mul_(clip_factor)
