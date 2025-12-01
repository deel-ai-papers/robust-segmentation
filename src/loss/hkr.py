import torch
import torch.nn as nn
import torch.nn.functional as F

from deel.torchlip import SoftHKRMulticlassLoss


class HKRLoss(nn.Module):
    def __init__(self, alpha, min_margin, temperature, ignore_index=255):
        super().__init__()
        self.hkr = SoftHKRMulticlassLoss(
            alpha=alpha,
            min_margin=min_margin,
            temperature=temperature,
            reduction="mean",
        )
        self.ignore_index = int(ignore_index)

    def forward(self, inputs, targets):
        # inputs: [B,C,H,W], targets: [B,H,W]
        assert inputs.ndim == 4 and targets.ndim == 3
        B, C, H, W = inputs.shape
        targets = targets.long()
        valid = targets != self.ignore_index
        if not valid.any():
            return inputs.sum() * 0.0

        # guard against labels >= C slipping through
        max_lab = targets[valid].max()
        if max_lab.item() >= C:
            raise RuntimeError(f"Label {max_lab.item()} >= num_classes {C}")

        onehot = (
            F.one_hot(
                torch.where(valid, targets, torch.zeros_like(targets)), num_classes=C
            )
            .permute(0, 3, 1, 2)
            .to(inputs.dtype)
        )

        x = inputs.permute(0, 2, 3, 1)[valid].view(-1, C)  # [N_valid, C]
        y = onehot.permute(0, 2, 3, 1)[valid].view(-1, C)  # [N_valid, C]
        return self.hkr(x, y)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 4, 3, 3
    logits = torch.randn(B, C, H, W, requires_grad=True)
    labels = torch.randint(0, C, (B, H, W))
    labels[0, 1, 1] = -100  # ignore pixel

    weights = [1.0, 2.0, 0.5, 1.0]
    loss_fn = HKRLoss(alpha=0.95, min_margin=0.1, temperature=5.0, ignore_index=-100)
    loss = loss_fn(logits, labels)
    print("Loss:", loss.item())
    loss.backward()
    print("Grad OK:", logits.grad is not None)
