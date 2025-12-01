import math
import torch
from torchmetrics import Metric

import torch.nn.functional as F
from tqdm import tqdm


class FNRThresholdEpsilon(Metric):
    def __init__(self, lipconstant=1.0, threshold=0.8):
        super().__init__()
        self.lipconstant = float(lipconstant)
        self.threshold = float(threshold)
        self.add_state(
            "epsilon",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_imgs",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor):
        assert logits.dim() == 4, "logits must be (B, C, H, W)"
        assert target.dim() == 3, "target must be (B, H, W)"
        assert logits.size(1) == 2, "Valid for binary segmentation"
        target = target.long()
        B, C, H, W = logits.shape
        assert target.shape == (
            B,
            H,
            W,
        ), "target shape must match logits spatial dimensions"
        lbls = target.bool()
        preds = logits.argmax(dim=1).bool()
        pos = target.float().sum(dim=(1, 2))

        # Compute robustness margins
        margins = F.relu(logits[:, 1, :, :] - logits[:, 0, :, :]) ** 2
        margins /= 2
        margins /= self.lipconstant

        # Attack only positives
        mask = torch.ones_like(target)
        margins[~lbls] = int(1e10)

        # Robustness certificate on true positives
        margins = margins.view(B, -1)
        margins, _ = margins.sort(dim=1)
        margins = torch.cumsum(margins, dim=1)

        # Compute robustness up to required number of false negatives
        idxs = self.threshold * pos
        idxs = [math.ceil(i) for i in idxs]
        eps_imgs = torch.sqrt(margins[torch.arange(B), idxs])
        eps_imgs.masked_fill_(torch.isinf(eps_imgs), 0)

        # Update values
        self.epsilon += torch.sum(eps_imgs)
        self.num_imgs += B

    def compute(self) -> dict:
        if self.num_imgs.item() == 0:
            return {
                "epsilon": float("nan"),
            }
        epsilon = self.epsilon / self.num_imgs
        return {
            "epsilon": epsilon.item(),
        }


class CertifiedFNR(Metric):
    def __init__(self, epsilon=0.1, lipconstant=1.0):
        super().__init__()
        self.epsilon = float(epsilon)
        self.lipconstant = float(lipconstant)
        self.add_state(
            "clean_fnr_imgs",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "robust_fnr_imgs",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_imgs",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor):
        assert logits.dim() == 4, "logits must be (B, C, H, W)"
        assert target.dim() == 3, "target must be (B, H, W)"
        assert logits.size(1) == 2, "Valid for binary segmentation"
        target = target.long()
        B, C, H, W = logits.shape
        assert target.shape == (
            B,
            H,
            W,
        ), "target shape must match logits spatial dimensions"

        # In a clean setting
        clean_tp_imgs = (logits.argmax(dim=1) * target).sum(dim=(1, 2)).to(torch.long)
        total_pos = target.sum(dim=(1, 2))

        self.num_imgs += B
        self.clean_fnr_imgs += (
            ((total_pos - clean_tp_imgs) / total_pos).sum().to(torch.float32)
        )

        # To compute a bound on the FNR we minimize the number of TP
        # since FN = T - TP
        margin = F.relu(logits[:, 1, :, :] - logits[:, 0, :, :])
        margin = margin * target  # Ensure the margin is 0 is the pixel is a negative
        margin = margin.view(B, -1)
        margin = (margin**2) / 2.0
        margin, _ = margin.sort(dim=1)
        margin = torch.cumsum(margin, dim=1)
        margin = torch.sqrt(margin)

        # Find the first index where margin >= epsilon * lipconstant
        threshold = self.epsilon * self.lipconstant
        robust_mask = (margin >= threshold).long()  # (B, H*W)

        robust_tp_imgs = robust_mask.sum(dim=1)
        self.robust_fnr_imgs += (
            ((total_pos - robust_tp_imgs) / total_pos).sum().to(torch.float32)
        )

    def compute(self) -> dict:
        if self.num_imgs.item() == 0:
            return {
                "certified_fnr": float("nan"),
                "clean_fnr": float("nan"),
            }
        clean_fnr = self.clean_fnr_imgs / self.num_imgs
        robust_fnr = self.robust_fnr_imgs / self.num_imgs
        return {
            "certified_fnr": robust_fnr.item(),
            "clean_fnr": clean_fnr.item(),
        }
