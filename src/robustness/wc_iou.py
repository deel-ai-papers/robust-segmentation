import torch
from torchmetrics import Metric
import torch.nn.functional as F

class WCClassIoU(Metric):
    def __init__(self, epsilon=0.1, lipconstant=1.0, class_num=0, ignore_index=255):
        super().__init__()
        self.epsilon = float(epsilon)
        self.lipconstant = float(lipconstant)
        self.class_num = int(class_num)
        self.ignore_index = int(ignore_index)

        # Micro-accumulators (sum of numerators/denominators across images)
        self.add_state(
            "correct_pixels",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )  # clean TP
        self.add_state(
            "clean_union",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )  # clean denom
        self.add_state(
            "robust_pixels",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )  # adv TP
        self.add_state(
            "adv_union", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )  # adv denom

        # Kept for API symmetry (unused)
        self.add_state(
            "total_pixels",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor):
        assert logits.dim() == 4, "logits must be (B, C, H, W)"
        assert target.dim() == 3, "target must be (B, H, W)"
        target = target.long()
        B, C, H, W = logits.shape
        assert target.shape == (B, H, W), "target shape must match logits spatial dims"

        k = self.class_num
        preds = torch.argmax(logits, dim=1)  # (B,H,W)
        logits_k = logits[:, k]  # (B,H,W)

        tmp = logits.clone()
        tmp[:, k] = float("-inf")
        max_others = tmp.amax(dim=1)  # (B,H,W)

        budget_sq = float((self.epsilon * self.lipconstant) ** 2)

        for b in range(B):
            valid_mask = target[b] != self.ignore_index
            
            is_k_gt = target[b] == k
            is_k_pred = preds[b] == k
            
            # Apply valid_mask strictly to all sets
            A = is_k_gt & is_k_pred & valid_mask         # TP
            Bm = (~is_k_gt) & is_k_pred & valid_mask     # FP
            Cm = is_k_gt & (~is_k_pred) & valid_mask     # FN
            D = (~is_k_gt) & (~is_k_pred) & valid_mask   # TN

            t = int(A.sum().item())
            b_ = int(Bm.sum().item())
            f_ = int(Cm.sum().item())
            Z0 = t + b_ + f_  # IoU denominator for this image/class

            if Z0 == 0:
                continue  # nothing for this image/class

            # Clean accumulators
            self.correct_pixels += torch.tensor(
                t, dtype=torch.long, device=logits.device
            )
            self.clean_union += torch.tensor(Z0, dtype=torch.long, device=logits.device)

            # TP->not-k and TN->k squared output costs: ((gap)^2)/2
            tp_costs = ((logits_k[b] - max_others[b])[A]).reshape(-1)
            tn_costs = ((max_others[b] - logits_k[b])[D]).reshape(-1)
            tp_sq = (tp_costs**2) / 2.0
            tn_sq = (tn_costs**2) / 2.0

            # Cumulative sums of sorted costs for both TP and TN missclassings
            if tp_sq.numel() > 0:
                tp_sorted = torch.sort(tp_sq).values
                PA = torch.cat([tp_sorted.new_zeros(1), torch.cumsum(tp_sorted, dim=0)])
            else:
                PA = torch.zeros(1, device=logits.device, dtype=logits.dtype)

            if tn_sq.numel() > 0:
                tn_sorted = torch.sort(tn_sq).values
                PD = torch.cat([tn_sorted.new_zeros(1), torch.cumsum(tn_sorted, dim=0)])
            else:
                PD = torch.zeros(1, device=logits.device, dtype=logits.dtype)

            # Sweep m (TPs broken), choose max feasible n (TNs->FPs) within budget
            best_num, best_den = t, Z0
            PA_cpu = PA.detach().to("cpu")
            PD_cpu = PD.detach().to("cpu")

            tA = PA_cpu.numel() - 1  # number of TPs
            for m in range(tA + 1):
                rem = budget_sq - float(PA_cpu[m])
                if rem < 0:
                    break
                n = (
                    int(
                        torch.searchsorted(PD_cpu, torch.tensor(rem), right=True).item()
                    )
                    - 1
                )
                if n < 0:
                    n = 0
                if n > (PD_cpu.numel() - 1):
                    n = PD_cpu.numel() - 1

                num = t - m
                den = Z0 + n
                if den > 0 and (num * best_den) < (best_num * den):  # minimize fraction
                    best_num, best_den = num, den

            self.robust_pixels += torch.tensor(
                best_num, dtype=torch.long, device=logits.device
            )
            self.adv_union += torch.tensor(
                best_den, dtype=torch.long, device=logits.device
            )

    def compute(self) -> dict:
        clean_den = int(self.clean_union.item())
        adv_den = int(self.adv_union.item())

        clean_iou = (
            float("nan") if clean_den == 0 else (self.correct_pixels.item() / clean_den)
        )
        adv_iou = (
            float("nan") if adv_den == 0 else (self.robust_pixels.item() / adv_den)
        )

        return {
            "clean_iou": clean_iou,
            "certified_worst_class_iou": adv_iou,
        }
