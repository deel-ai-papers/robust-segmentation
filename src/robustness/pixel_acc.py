import torch
from torchmetrics import Metric

import torch.nn.functional as F
from tqdm import tqdm


# Certifiable pixel accuracy for segmentation tasks
class CertifiedPixelAcc(Metric):
    def __init__(self, epsilon=0.1, lipconstant=1.0):
        super().__init__()
        self.epsilon = float(epsilon)
        self.lipconstant = float(lipconstant)
        self.add_state(
            "robust_pixel_acc",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "clean_pixel_acc",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_imgs",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor, ignore_index=None):
        assert logits.dim() == 4, "logits must be (B, C, H, W)"
        assert target.dim() == 3, "target must be (B, H, W)"
        target = target.long()
        B, C, H, W = logits.shape
        assert target.shape == (
            B,
            H,
            W,
        ), "target shape must match logits spatial dimensions"

        preds = torch.argmax(logits, dim=1)  # (B, H, W)
        correct = preds == target
        nb_correct = correct.sum(dim=(1, 2)).long()  # (B,)

        if ignore_index is None:
            total_pixels = (B * H * W) * torch.ones((B,))
        else:
            total_pixels = (target != ignore_index).sum(dim=(1, 2)).long()

        # Compute certified robust pixels
        topk = torch.topk(logits, k=2, dim=1).values
        margin_top = topk[:, 0] - topk[:, 1]  # (B, H, W)

        # Only consider correctly predicted pixels (cannot be correct for ignore_index)
        margin = margin_top * correct
        margin = margin.view(B, -1)  # (B, H*W)
        margin = (margin**2) / 2.0
        margin, _ = margin.sort(dim=1)
        margin = torch.cumsum(margin, dim=1)
        margin = torch.sqrt(margin)

        # Find the first index where margin >= epsilon * lipconstant
        threshold = self.epsilon * self.lipconstant
        robust_mask = (margin >= threshold).long()  # (B, H*W)
        robust_counts = robust_mask.sum(dim=1)  # (B,)

        clean_pacc_imgs = (nb_correct / total_pixels).sum()
        robust_pacc_imgs = (robust_counts / total_pixels).sum()

        self.num_imgs += B
        self.clean_pixel_acc += clean_pacc_imgs.float()
        self.robust_pixel_acc += robust_pacc_imgs.float()

    def compute(self) -> dict:
        if self.num_imgs.item() == 0:
            return {
                "certified_pixel_acc": float("nan"),
                "clean_pixel_acc": float("nan"),
            }
        clean_acc = (self.clean_pixel_acc / self.num_imgs).to(torch.float32)
        robust_acc = (self.robust_pixel_acc / self.num_imgs).to(torch.float32)
        return {
            "certified_pixel_acc": robust_acc.item(),
            "clean_pixel_acc": clean_acc.item(),
        }
