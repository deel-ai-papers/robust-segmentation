from typing import Optional, Union
import math
import torch
from torch import Tensor, nn
from torch.autograd import grad


def pgd(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    masks: Tensor = None,
    epsilon: float = 0.03,
    alpha: float = 0.01,
    num_steps: int = 40,
    p: Union[int, float] = 2,
) -> Tensor:
    """Projected Gradient Descent (PGD) attack"""
    device = inputs.device
    batch_size = len(inputs)

    labels_ = labels.clone()
    if masks is not None:
        labels_[masks] = -100

    # LR schedule setup
    target_sum = 1.25 * epsilon
    max_lr = (2 * target_sum) / (num_steps + 1)

    lrs = []
    for t in range(num_steps):
        lr_t = 0.5 * max_lr * (1 + math.cos(t * math.pi / num_steps))
        lrs.append(lr_t)
    lrs = torch.tensor(lrs, device=device)

    # Variables
    δ = torch.zeros_like(inputs)
    δ.requires_grad = True

    # This ensures code works for Images (4D), Audio (3D), or Flat features (2D)
    view_shape = [batch_size] + [1] * (inputs.ndim - 1)

    for i in range(num_steps):
        adv_inputs = inputs + δ
        logits = model(adv_inputs)

        if isinstance(logits, tuple):
            logits = logits[0]

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits, labels_)

        gradients = grad(loss, δ)[0]

        grad_norms = torch.norm(gradients.view(batch_size, -1), p=p, dim=1) + 1e-10
        gradients = gradients / grad_norms.view(view_shape)

        δ.data = δ.data + lrs[i] * gradients

        if p == float("inf"):
            δ.data = torch.clamp(δ.data, -epsilon, epsilon)
        else:
            # Calculate norms of the current perturbation
            delta_norms = δ.data.view(batch_size, -1).norm(p=p, dim=1)

            # Calculate the scaling factor: min(1, epsilon / norm)
            # We treat division by zero (if norm is 0) by adding epsilon
            factor = torch.min(
                torch.ones_like(delta_norms), epsilon / (delta_norms + 1e-12)
            )

            # Apply factor. If norm <= epsilon, factor is 1 (no change).
            # If norm > epsilon, factor shrinks the vector to length epsilon.
            δ.data = δ.data * factor.view(view_shape)

    best_adv = inputs + δ.detach()
    return best_adv
