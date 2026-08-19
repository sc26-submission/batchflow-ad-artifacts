from __future__ import annotations

from torch import nn


def add_weight_decay(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    """Apply weight decay to matrix/tensor weights but not biases or norm terms."""
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": float(weight_decay)},
    ]
