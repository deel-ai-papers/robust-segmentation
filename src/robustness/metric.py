import torch
from torchmetrics import Metric

import torch.nn.functional as F
from tqdm import tqdm


class StabilityMeasure(Metric):
    def __init__(self, alphas=(0.25, 0.5, 0.75), lipconstant=1.0):
        super().__init__()
        self.alphas = [float(a) for a in alphas]
        assert all(0.0 <= a <= 1.0 for a in self.alphas), "alphas must be in [0,1]"
        self.lipconstant = float(lipconstant)
        self.add_state(
            "cum_curve_sum",
            default=torch.tensor([], dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "n_samples", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    @torch.no_grad()
    def update(self, logits: torch.Tensor):
        assert logits.dim() == 4, "logits must be (B, C, H, W)"
        B, C, H, W = logits.shape
        topk = torch.topk(logits, k=2, dim=1).values
        margin = topk[:, 0] - topk[:, 1]
        margin = (margin**2) / 2.0
        margin = margin.view(B, -1)
        margin_sorted, _ = torch.sort(margin, dim=1)
        margin_cum = torch.cumsum(margin_sorted, dim=1)
        batch_sum = margin_cum.sum(dim=0, dtype=torch.float64)  # (N,)

        if self.cum_curve_sum.numel() == 0:
            self.cum_curve_sum = batch_sum
        else:
            assert (
                batch_sum.numel() == self.cum_curve_sum.numel()
            ), "All inputs must keep H*W constant."
            self.cum_curve_sum = self.cum_curve_sum + batch_sum

        self.n_samples = self.n_samples + B

    def compute(self) -> torch.Tensor:
        if self.cum_curve_sum.numel() == 0 or self.n_samples.item() == 0:
            return torch.full((len(self.alphas),), float("nan"))

        mean_curve = self.cum_curve_sum / self.n_samples.to(torch.float64)  # (N,)
        N = mean_curve.numel()
        idxs = [min(int(a * (N - 1)), N - 1) for a in self.alphas]
        vals = {
            "stability@" + str(int(a * 100)): torch.sqrt(mean_curve[idx])
            .to(torch.float32)
            .item()
            / self.lipconstant
            for a, idx in zip(self.alphas, idxs)
        }
        return vals


class RobustnessMeasure(Metric):
    def __init__(self, alphas=(0.25, 0.5, 0.75), lipconstant=1.0):
        super().__init__()
        self.alphas = [float(a) for a in alphas]
        assert all(0.0 <= a <= 1.0 for a in self.alphas), "alphas must be in [0,1]"
        self.lipconstant = float(lipconstant)
        self.add_state(
            "cum_curve_sum",
            default=torch.tensor([], dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "n_samples", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor, ignore_index: int):
        assert logits.dim() == 4, "logits must be (B, C, H, W)"
        B, C, H, W = logits.shape
        correct = (logits.argmax(1) == labels) * (labels != ignore_index)
        topk = torch.topk(logits, k=2, dim=1).values
        margin = topk[:, 0] - topk[:, 1]
        margin = ((margin**2) / 2.0) * correct
        margin = margin.view(B, -1)
        margin_sorted, _ = torch.sort(margin, dim=1)
        margin_cum = torch.cumsum(margin_sorted, dim=1)
        batch_sum = margin_cum.sum(dim=0, dtype=torch.float64)  # (N,)

        if self.cum_curve_sum.numel() == 0:
            self.cum_curve_sum = batch_sum
        else:
            assert (
                batch_sum.numel() == self.cum_curve_sum.numel()
            ), "All inputs must keep H*W constant."
            self.cum_curve_sum = self.cum_curve_sum + batch_sum

        self.n_samples = self.n_samples + B

    def compute(self) -> torch.Tensor:
        if self.cum_curve_sum.numel() == 0 or self.n_samples.item() == 0:
            return torch.full((len(self.alphas),), float("nan"))

        mean_curve = self.cum_curve_sum / self.n_samples.to(torch.float64)  # (N,)
        N = mean_curve.numel()
        idxs = [min(int(a * (N - 1)), N - 1) for a in self.alphas]
        vals = {
            "robustness@" + str(int(a * 100)): torch.sqrt(mean_curve[idx])
            .to(torch.float32)
            .item()
            / self.lipconstant
            for a, idx in zip(self.alphas, idxs)
        }
        return vals
