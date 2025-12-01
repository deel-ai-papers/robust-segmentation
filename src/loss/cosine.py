import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicCosineLoss(nn.Module):
    def __init__(self, ignore_index=-100, eps=1e-8, weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps
        self.weight = (
            None if weight is None else torch.as_tensor(weight, dtype=torch.float32)
        )

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W], targets: [B, H, W]
        assert inputs.ndim == 4
        assert targets.ndim == 3
        targets = targets.long()

        B, C, H, W = inputs.shape
        mask = targets != self.ignore_index
        if not mask.any():
            return inputs.sum() * 0.0

        onehot = (
            F.one_hot(
                torch.where(mask, targets, torch.zeros_like(targets)), num_classes=C
            )
            .permute(0, 3, 1, 2)
            .to(inputs.dtype)
        )
        x = inputs.permute(0, 2, 3, 1)[mask]
        y = onehot.permute(0, 2, 3, 1)[mask]
        t = targets[mask]

        x = x / (x.norm(p=2, dim=1, keepdim=True) + self.eps)
        y = y / (y.norm(p=2, dim=1, keepdim=True) + self.eps)
        loss = 1 - (x * y).sum(1)

        if self.weight is not None:
            w = self.weight.to(inputs.device, inputs.dtype)
            loss = loss * w[t]

        return loss.mean()


class NaiveCosineLoss(nn.Module):
    def __init__(self, ignore_index=255, eps=1e-8):
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)

    def forward(self, inputs, targets):
        # inputs: [B,C,H,W], targets: [B,H,W] with ignore_index
        assert inputs.ndim == 4 and targets.ndim == 3
        B, C, H, W = inputs.shape
        targets = targets.long()
        valid = targets != self.ignore_index
        if not valid.any():
            return inputs.sum() * 0.0

        # guard: any kept label must be < C
        tmax = targets[valid].max()
        if tmax.item() >= C:
            raise RuntimeError(f"Label {tmax.item()} >= num_classes {C}")

        valid_f = valid.to(inputs.dtype)  # [B,H,W]

        # dot(x_b, y_b): y is one-hot → pick true class logit then sum on valid pixels
        t_safe = targets.clamp(0, C - 1)  # safe for ignored pixels
        true_logit = torch.gather(inputs, 1, t_safe.unsqueeze(1)).squeeze(1)  # [B,H,W]
        dot = (true_logit * valid_f).sum(dim=(1, 2))  # [B]

        # ||x_b||: sum of squared logits over channels & valid pixels
        x_sq = inputs.pow(2)  # [B,C,H,W]
        x_norm = (
            (x_sq * valid_f.unsqueeze(1))
            .sum(dim=(1, 2, 3))
            .clamp_min(self.eps**2)
            .sqrt()
        )  # [B]

        # ||y_b||: one-hot over valid pixels → sqrt(#valid pixels)
        n_valid = valid_f.sum(dim=(1, 2))  # [B]
        keep = n_valid > 0
        if not keep.any():
            return inputs.sum() * 0.0
        y_norm = n_valid.clamp_min(1.0).sqrt()  # [B]

        cos = dot[keep] / (x_norm[keep] * y_norm[keep])  # [B_kept]
        return (1 - cos).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 4, 3, 3
    logits = torch.randn(B, C, H, W, requires_grad=True)
    labels = torch.randint(0, C, (B, H, W))
    labels[0, 1, 1] = -100  # ignore pixel

    weights = [1.0, 2.0, 0.5, 1.0]
    for loss_fn in [
        ClassicCosineLoss(ignore_index=-100, weight=weights),
        NaiveCosineLoss(ignore_index=-100),
    ]:
        loss = loss_fn(logits, labels)
        print("Loss:", loss.item())
        loss.backward()
        print("Grad OK:", logits.grad is not None)
