# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from transformers.utils.versions import require_version

from .coso_projector import CoSOProjector


class CoSOAdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. "
                "Use torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _is_coso_group(group):
        return "rank" in group

    def _get_or_create_projector(self, group, state):
        if "projector" not in state:
            state["projector"] = CoSOProjector(
                group["proj_rank"],
                group["rank"],
                db_decay=group["db_decay"],
                update_proj_gap=group["update_proj_gap"],
            )
        return state["projector"]

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[param]
                if "step" not in state:
                    state["step"] = 0

                if self._is_coso_group(group):
                    projector = self._get_or_create_projector(group, state)
                    grad = projector.project(grad, state["step"])

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                norm_grad = exp_avg / denom
                if self._is_coso_group(group):
                    norm_grad = projector.project_back(norm_grad)

                param.add_(norm_grad, alpha=-step_size)

                if group["weight_decay"] > 0.0:
                    param.add_(param, alpha=-group["lr"] * group["weight_decay"])

        return loss

    def resetting_lr(self, lr, fc_lr):
        for group in self.param_groups:
            group["lr"] = lr if self._is_coso_group(group) else fc_lr
