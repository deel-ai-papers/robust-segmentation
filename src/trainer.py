import torch
import numpy as np
from torchmetrics import Accuracy, JaccardIndex

from src.robustness import (
    StabilityMeasure,
    CertifiedPixelAcc,
    WCClassIoU,
    RobustnessMeasure,
)
from tqdm import tqdm
from torchvision.utils import save_image
import os


class LipschitzConstraint(torch.nn.Module):
    def __init__(self, coeff=0.1, lipconstant=None):
        super().__init__()
        self.coef = coeff
        self.lipconstant = lipconstant

    def forward(self, lip):
        if self.lipconstant is None:
            self.lipconstant = lip.detach()
            return 0.0
        else:
            return self.coef * torch.log(lip / self.lipconstant).abs()


default = LipschitzConstraint()


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        scheduler,
        train_loader,
        val_loader,
        num_classes,
        ignore_index=255,
        device="cuda",
        lipschitz=False,
        constraint=default,
        epsilon=None,
        wc_iou_class=None,
        sigma_train=0.0,
        img_save_dir="",
    ):
        self.model = model.to(device)
        self.opt = optimizer
        self.crit = criterion
        self.sched = scheduler
        self.constraint = constraint
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.epsilon = epsilon
        self.lipschitz = lipschitz
        self.sigma_train = sigma_train
        self.img_save_dir = img_save_dir

        self.acc = Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=ignore_index
        ).to(device)
        self.miou = JaccardIndex(
            task="multiclass", num_classes=num_classes, ignore_index=ignore_index
        ).to(device)
        self.cls_ious = JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
            average=None,
        ).to(device)

        self.cpacc = (
            CertifiedPixelAcc(
                epsilon=epsilon,
                lipconstant=1.0,
            ).to(device)
            if epsilon is not None
            else None
        )
        self.wc_iou = (
            WCClassIoU(epsilon=epsilon, lipconstant=1.0, class_num=wc_iou_class).to(
                device
            )
            if wc_iou_class is not None
            else None
        )
        self.robustness = (
            StabilityMeasure(alphas=(0.25, 0.5, 0.75), lipconstant=1.0).to(device)
            if lipschitz
            else None
        )

    def _step(self, batch):
        imgs, labels = (x.to(self.device) for x in batch)

        if self.sigma_train != 0.0:
            imgs += torch.randn_like(imgs) * self.sigma_train

        preds = self.model(imgs)
        if isinstance(preds, (list, tuple)):
            if preds[1].requires_grad:
                loss = self.crit(preds[0], labels)
                loss += self.constraint(preds[1])
            else:
                loss = self.crit(preds[0], labels)
        else:
            loss = self.crit(preds, labels)

        return preds, labels, loss

    def train_one_epoch(self, wandb_run=None):
        self.model.train()
        total_loss = 0.0

        for imgs, labels in tqdm(self.train_loader):
            preds, labels, loss = self._step((imgs, labels))
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            total_loss += loss.item()

            if isinstance(preds, tuple):
                pacc = (preds[0].argmax(dim=1) == labels).float().mean()
            else:
                pacc = (preds.argmax(dim=1) == labels).float().mean()

            if wandb_run is not None:
                wandb_run.log({"tr_loss": loss.item()})
                wandb_run.log({"pacc_train": pacc.item()})
                if isinstance(preds, tuple):
                    wandb_run.log({"lipconstant": preds[1]})
                else:
                    wandb_run.log({"lipconstant": 1.0})

        if self.sched:
            self.sched.step()
            if wandb_run is not None:
                wandb_run.log({"lr": self.sched.get_last_lr()[0]})

        return total_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        # Proper TM usage: reset -> update in loop -> compute once
        self.acc.reset()
        self.miou.reset()
        self.cls_ious.reset()
        if self.robustness:
            self.robustness.reset()
        if self.cpacc:
            self.cpacc.reset()
        if self.wc_iou:
            self.wc_iou.reset()

        for imgs, labels in tqdm(self.val_loader, desc="Eval certification"):
            preds, labels, _ = self._step((imgs, labels))
            if isinstance(preds, (list, tuple)):
                lip_ = float(preds[1].item())
                preds = preds[0]
            else:
                lip_ = 1.0
            self.acc.update(preds, labels)
            self.miou.update(preds, labels)
            self.cls_ious.update(preds, labels)
            if self.robustness:
                self.robustness.lipconstant = lip_
                self.robustness.update(preds)
            if self.cpacc:
                self.cpacc.lipconstant = lip_
                self.cpacc.update(preds, labels, ignore_index=self.ignore_index)
            if self.wc_iou:
                self.wc_iou.lipconstant = lip_
                self.wc_iou.update(preds, labels)

        acc = self.acc.compute().item()
        miou = self.miou.compute().item()
        cls = torch.nan_to_num(self.cls_ious.compute())  # per-class (C,)

        metrics = {"acc": acc, "mIoU": miou, "cls_IoUs": cls.cpu(), "lip": lip_}
        if self.robustness:
            rob = self.robustness.compute()
            metrics.update(rob)
        if self.cpacc:
            cpacc = self.cpacc.compute()
            metrics[f"pacc_@{self.cpacc.epsilon}"] = cpacc["certified_pixel_acc"]
        if self.wc_iou:
            wciou = self.wc_iou.compute()
            metrics[f"clean_IoU{self.wc_iou.class_num}"] = wciou["clean_iou"]
            metrics[f"wcIoU_{self.wc_iou.class_num}@{self.wc_iou.epsilon}"] = wciou[
                "certified_worst_class_iou"
            ]
        return metrics

    def attack_q1(
        self,
        attacks,
        adv_thresholds,
        p=2,
        num_batches=float("inf"),
    ):
        """
        Evaluates the model's performance against a set of adversarial attacks.

        This function first calculates the clean and certified accuracy of the model.
        It then applies a list of adversarial attacks to the validation data and
        computes the worst-case accuracy among all attacks for each sample.

        Args:
            attacks (list): A list of attack objects to be applied.
            adv_thresholds (list): A list of thresholds corresponding to the attacks.
            p (int, optional): The p-norm to use for measuring perturbations. Defaults to 2.
            num_batches (float, optional): The maximum number of batches to process.
                                         Defaults to float("inf") (all batches).

        Returns:
            dict: A dictionary containing performance metrics:
                  - 'attacked_acc': Robust accuracy under the worst of the given attacks.
                  - 'clean_acc': Accuracy on original, unperturbed images.
                  - f'CRPA@{self.cpacc.epsilon}': Certified Robust Pixel Accuracy.
        """
        self.model.eval()
        self.acc.reset()
        self.miou.reset()
        self.cpacc.reset()

        cpt = 0

        # First pass: Calculate clean and certified accuracy
        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            labels = labels.long()
            with torch.no_grad():
                outputs = self.model(imgs)
            if isinstance(outputs, (list, tuple)):
                lip_c = float(outputs[1].item())
                clean_preds = outputs[0]
            else:
                if self.lipschitz:
                    lip_c = 1.0
                else:
                    lip_c = float("inf")
                clean_preds = outputs
            self.acc.update(clean_preds, labels)
            self.cpacc.lipconstant = lip_c
            self.cpacc.update(clean_preds, labels, ignore_index=self.ignore_index)
            cpt += 1
            if cpt > num_batches:
                break

        cpacc = self.cpacc.compute()
        pacc = self.acc.compute().item()
        wc_pacc = cpacc["certified_pixel_acc"]

        # Second pass: Apply attacks and calculate robust accuracy
        cpt = 0
        corrects, totals = [], []

        for imgs, labels in tqdm(self.val_loader, desc="Attacking"):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            labels = labels.long()
            masks = labels != self.ignore_index
            adv_outputs = []

            for att, thresh in zip(attacks, adv_thresholds):
                if thresh and self.lipschitz:
                    max_adv_threshold = 1 - wc_pacc
                    adv_imgs = att(
                        model=self.model,
                        inputs=imgs,
                        labels=labels,
                        masks=masks,
                        adv_threshold=max_adv_threshold,
                    )
                else:
                    adv_imgs = att(
                        model=self.model,
                        inputs=imgs,
                        labels=labels,
                        masks=masks,
                    )

                # Ensure the attack respects the epsilon constraint
                norms = torch.norm((adv_imgs - imgs).view(imgs.size(0), -1), p=p, dim=1)
                if not torch.all(norms <= self.epsilon + 1e-4):
                    # print(
                    #     "[WARNING] Attack failed to meet the epsilon constraint - clipping."
                    # )
                    norms = norms.view(-1, 1, 1, 1)
                    factor = self.epsilon / (norms + 1e-8)
                    factor = torch.clamp(factor, max=1.0)
                    adv_imgs = imgs + (adv_imgs - imgs) * factor
                    norms_ = torch.norm(
                        (adv_imgs - imgs).view(imgs.size(0), -1), p=p, dim=1
                    )
                    assert norms_.max() <= self.epsilon + 1e-4, "Clipping failed!"

                with torch.no_grad():
                    outs = self.model(adv_imgs)
                    if isinstance(outs, tuple):
                        outs = outs[0]
                    adv_outputs.append(outs)

            # Shape (N_attacks, B, C, H, W)
            attacked_out = torch.stack(adv_outputs, dim=0)

            # For each image, keep only the attack that achieved the worst pixel accuracy
            total = (labels != self.ignore_index).sum(dim=(1, 2))  # (B,)
            # Shape (N_attacks, B)
            correct_per_attack = (attacked_out.argmax(2) == labels).sum(dim=(2, 3))

            correct = correct_per_attack.min(dim=0).values  # (B,)
            totals.append(total)
            corrects.append(correct)

            cpt += 1
            if cpt > num_batches:
                break

        # This creates a single tensor of results from all processed batches.
        correct_all = torch.cat(corrects, dim=0)
        total_all = torch.cat(totals, dim=0)

        # Calculate final metrics
        metrics = {
            "attacked_acc": (correct_all.sum() / (total_all.sum() + 1e-8)).item(),
            "clean_acc": pacc,
        }
        metrics[f"CRPA@{self.cpacc.epsilon}"] = wc_pacc

        # Sanity check: robust accuracy should not be better than certified accuracy
        assert (
            wc_pacc <= metrics["attacked_acc"] + 1e-8
        ), f"Certified accuracy ({wc_pacc}) should not exceed attacked accuracy ({metrics['attacked_acc']})"

        return metrics

    def attack_q2(
        self,
        attacks,
        adv_threshold,
        p=2,
        num_batches=float("inf"),
    ):
        """
        Evaluates the model's performance against a set of adversarial attacks.

        This function first calculates the clean and certified accuracy of the model.
        It then applies a list of adversarial attacks to the validation data and
        computes the worst-case accuracy among all attacks for each sample.

        Args:
            attacks (list): A list of attack objects to be applied.
            adv_threshold (list): The desired adversarial threshold.
            p (int, optional): The p-norm to use for measuring perturbations. Defaults to 2.
            num_batches (float, optional): The maximum number of batches to process.
                                         Defaults to float("inf") (all batches).

        Returns:
            dict: A dictionary containing performance metrics:
                  - 'attacked_acc': Robust accuracy under the worst of the given attacks.
                  - 'clean_acc': Accuracy on original, unperturbed images.
                  - f'CRPA@{self.cpacc.epsilon}': Certified Robust Pixel Accuracy.
        """
        self.model.eval()
        self.acc.reset()
        self.cpacc.reset()

        # Q2 certificate
        wc_eps = RobustnessMeasure(alphas=[adv_threshold], lipconstant=1.0).to(
            self.device
        )
        wc_eps.reset()

        cpt = 0

        # First pass: Calculate clean and certified accuracy
        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            labels = labels.long()
            with torch.no_grad():
                outputs = self.model(imgs)
            if isinstance(outputs, (list, tuple)):
                lip_c = float(outputs[1].item())
                clean_preds = outputs[0]
            else:
                if self.lipschitz:
                    lip_c = 1.0
                else:
                    lip_c = float("inf")
                clean_preds = outputs
            wc_eps.lipconstant = lip_c
            self.acc.update(clean_preds, labels)
            self.cpacc.lipconstant = lip_c
            self.cpacc.update(clean_preds, labels, ignore_index=self.ignore_index)
            wc_eps.update(clean_preds, labels, ignore_index=self.ignore_index)

            cpt += 1
            if cpt > num_batches:
                break

        cpacc = self.cpacc.compute()
        wc_pacc = cpacc["certified_pixel_acc"]
        q2_cert = wc_eps.compute()

        # Second pass: Apply attacks and calculate robust accuracy
        cpt = 0
        min_norms = []

        for imgs, labels in tqdm(self.val_loader, desc="Attacking"):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            labels = labels.long()
            masks = labels != self.ignore_index
            adv_outputs = []

            norms = float("inf") * torch.ones(imgs.size(0))
            for att in attacks:
                max_adv_threshold = 1 - wc_pacc
                adv_imgs = att(
                    model=self.model,
                    inputs=imgs,
                    labels=labels,
                    masks=masks,
                    adv_threshold=adv_threshold,
                )

                with torch.no_grad():
                    out = self.model(adv_imgs)

                    if isinstance(out, tuple):
                        out = out[0]

                    valid = (
                        (
                            (out.argmax(1) == labels).sum(dim=(1, 2))
                            / (labels != self.ignore_index).sum(dim=(1, 2))
                        )
                        <= 1 - adv_threshold
                    ).cpu()
                    norms_ = torch.norm(
                        (adv_imgs - imgs).view(imgs.size(0), -1), p=p, dim=1
                    ).cpu()
                    norms_ = torch.where(
                        valid, norms_, float("inf") * torch.ones_like(valid)
                    )
                    norms = torch.where(norms_ < norms, norms_, norms)

            min_norms.append(norms.cpu().numpy())

            cpt += 1
            if cpt > num_batches:
                break

        # Calculate final metrics
        min_norms = np.array(min_norms)
        emp_eps = np.mean(min_norms[np.isfinite(min_norms)])
        theo_eps = q2_cert["robustness@" + str(int(adv_threshold * 100))]
        assert emp_eps >= theo_eps
        metrics = {
            "empirical_robustness": emp_eps,
            "theoretical_robustness": theo_eps,
        }
        return metrics


if __name__ == "__main__":
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    model = nn.Conv2d(3, 8, 3, 1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    sched = None
    num_epochs = 3
    imgs = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 8, (4, 32, 32))
    loader = DataLoader(TensorDataset(imgs, labels), batch_size=2)
    T = Trainer(model, opt, crit, sched, loader, loader, num_classes=8)
    for e in range(num_epochs):
        print(f"Epoch {e + 1}/{num_epochs}")
        print("loss:", T.train_one_epoch())
        print(T.evaluate())
