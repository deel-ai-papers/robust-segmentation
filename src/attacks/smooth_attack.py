from typing import Optional, Union
import math
import torch
from torch import Tensor, nn
from torch.autograd import grad
import torch.nn.functional as F


def smoothadv_pgd(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    masks: Tensor = None,
    epsilon: float = 0.5,
    alpha: float = 0.01,
    num_steps: int = 40,
    sigma: float = 0.25,
    n_samples: int = 32,
    p: Union[int, float] = 2,
) -> Tensor:
    """
    PGD attack against randomized smoothing (SMOOTHADV objective).

    Supports:
    - classification
    - segmentation with ignore masks
    - L2 or Linf
    """

    device = inputs.device
    batch_size = len(inputs)

    labels_ = labels.clone()
    if masks is not None:
        labels_[masks] = -100  # ignore index

    # cosine LR schedule (same style as your PGD)
    target_sum = 1.25 * epsilon
    max_lr = (2 * target_sum) / (num_steps + 1)

    lrs = []
    for t in range(num_steps):
        lr_t = 0.5 * max_lr * (1 + math.cos(t * math.pi / num_steps))
        lrs.append(lr_t)
    lrs = torch.tensor(lrs, device=device)

    δ = torch.zeros_like(inputs)
    δ.requires_grad = True

    view_shape = [batch_size] + [1] * (inputs.ndim - 1)

    # ---- reuse Gaussian noise across steps (important) ----
    noise = torch.randn(
        (n_samples,) + inputs.shape, device=device
    ) * sigma  # shape: (n_samples, B, C, H, W)

    for i in range(num_steps):
        perturbed = inputs + δ

        # ---- Monte Carlo smoothing ----
        # expand inputs for noise sampling
        expanded = perturbed.unsqueeze(0) + noise  # (n_samples,B,...)
        flat = expanded.view(-1, *inputs.shape[1:])  # merge sample+batch

        logits = model(flat)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.view(n_samples, batch_size, *logits.shape[1:])

        probs = F.softmax(logits, dim=2 if logits.ndim > 3 else -1)

        if probs.ndim == 3:
            # (n_samples, B, C)
            true_probs = probs.gather(
                2, labels_.view(1, batch_size, 1).expand(n_samples, -1, -1)
            ).squeeze(-1)

            mean_probs = true_probs.mean(dim=0)  # (B,)

            loss = -torch.log(mean_probs + 1e-12).mean()
        else:
            # (n_samples,B,C,H,W)
            y_rep = labels_.unsqueeze(0).expand(n_samples, -1, -1, -1)

            true_probs = probs.gather(
                2, y_rep.unsqueeze(2)
            ).squeeze(2)  # (n_samples,B,H,W)

            mean_probs = true_probs.mean(dim=0)  # (B,H,W)

            valid_mask = (labels_ != -100)

            valid_probs = mean_probs[valid_mask]
            loss = -torch.log(valid_probs + 1e-12).mean()

        gradients = grad(loss, δ)[0]

        grad_norms = torch.norm(gradients.view(batch_size, -1), p=p, dim=1) + 1e-10
        gradients = gradients / grad_norms.view(view_shape)

        δ.data = δ.data + lrs[i] * gradients

        if p == float("inf"):
            δ.data = torch.clamp(δ.data, -epsilon, epsilon)
        else:
            delta_norms = δ.data.view(batch_size, -1).norm(p=p, dim=1)
            factor = torch.min(
                torch.ones_like(delta_norms), epsilon / (delta_norms + 1e-12)
            )
            δ.data = δ.data * factor.view(view_shape)

    best_adv = inputs + δ.detach()
    return best_adv
