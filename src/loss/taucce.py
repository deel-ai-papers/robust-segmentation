import torch
import torch.nn as nn
import torch.nn.functional as F


class TauCCE(nn.Module):
    def __init__(self, tau=1.0, ignore_index=None, weight=None):
        super().__init__()
        assert tau > 0, "Tau must be superior to 0."
        self.tau = float(tau)
        self.ignore_index = ignore_index if ignore_index is not None else -100
        weight = (
            torch.tensor(weight, dtype=torch.float32) if weight is not None else None
        )
        self.weight = weight

    def forward(self, inputs, targets):
        """
        inputs:  [B, C, H, W] (logits)
        targets: [B, H, W] (int labels with possible ignore_index)
                 or [B, C, H, W] one-hot (ignore mask not inferable here)
        """
        if targets.ndim == 4:
            targets_idx = targets.argmax(1).long()
        else:
            targets_idx = targets.long()

        weight = (
            self.weight.to(dtype=inputs.dtype, device=inputs.device)
            if self.weight is not None
            else None
        )
        tce_loss = (
            F.cross_entropy(
                inputs * self.tau,
                targets_idx,
                reduction="mean",
                ignore_index=self.ignore_index,
                weight=weight,
            )
            / self.tau
        )
        return tce_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 4, 3, 3
    logits = torch.randn(B, C, H, W, requires_grad=True)
    labels = torch.randint(0, C, (B, H, W))
    labels[0, 1, 1] = -100  # ignore pixel

    weights = [1.0, 2.0, 0.5, 1.0]
    loss_fn = TauCCE(tau=4.0, ignore_index=-100, weight=weights)
    loss = loss_fn(logits, labels)
    print("Loss:", loss.item())
    loss.backward()
    print("Grad OK:", logits.grad is not None)
