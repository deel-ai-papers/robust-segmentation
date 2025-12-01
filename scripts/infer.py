import torch
import torch.nn.functional as F
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets import (
    VOCSegmentation,
    Cityscapes,
    NoisySquaresSegmentationDataset,
    OxfordIIITPet,
)
from src.datasets import get_dataset
from src import get_model, get_optimizer, get_loss
from src.trainer import Trainer
from src.utils.logging import brief_epoch_log
from src.loss.taucce import TauCCE
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
import matplotlib.colors as colors


parser = argparse.ArgumentParser(description="train a segmentation model.")
parser.add_argument(
    "--dataset",
    type=str,
    choices=["voc", "cityscapes", "toy", "iiit_pets", "kvasir"],
    required=True,
    help="dataset to use for training.",
)
parser.add_argument(
    "--img_size", type=int, default=512, help="image size for training."
)
parser.add_argument("--square_imgs", action="store_true", help="use square images.")
parser.add_argument(
    "--data_root", type=str, required=True, help="root directory of the dataset."
)
parser.add_argument(
    "--year", type=str, default="2012", help="year of the voc dataset (if applicable)."
)
parser.add_argument(
    "--download", action="store_true", help="download the dataset if not present."
)
parser.add_argument(
    "--batch_size", type=int, default=8, help="batch size for training."
)
parser.add_argument(
    "--type_param",
    type=str,
    choices=["ortho", "spectral", "unconstrained"],
    default="ortho",
)
parser.add_argument("--klip", action="store_true", help="use klip model.")
parser.add_argument("--learnable_coeffs", action="store_true", help="use klip model.")
parser.add_argument("--wandb_log", action="store_true", help="use klip model.")
parser.add_argument(
    "--config", type=str, default="S", help="configuration for the model."
)

args = parser.parse_args()

assert args.dataset == "kvasir", "Due to custom colormap code, should be cleaned up"
train_ds, test_ds, means, stds, num_classes, ignore_index = get_dataset(args)
print(f"training dataset size: {len(train_ds)}")
print(f"testing dataset size: {len(test_ds)}")

# create dataloaders
train_loader = DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
)
test_loader = DataLoader(
    test_ds, batch_size=2 * args.batch_size, shuffle=False, num_workers=4
)

# get model
model = get_model(means, stds, in_channels=3, num_classes=num_classes, args=args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Model has {sum(p.numel() for p in model.parameters())} parameters.")

model.to(device)

# Find the files with the right prefix
lipconfig = (
    "rms"
    if args.klip and not args.learnable_coeffs
    else "klip"
    if args.klip
    else "1lip"
)
prefix = f"{args.dataset}_{args.type_param}_{args.config}"
model_files = [
    f
    for f in os.listdir("checkpoints")
    if f.startswith(prefix) and f.endswith(".pth") and "optimizer" not in f
]
model_files = [f for f in model_files if lipconfig in f]

# Make the user select the right model if multiple are found
if len(model_files) > 1:
    print("Multiple model files found:")
    for i, f in enumerate(model_files):
        print(f"{i}: {f}")
    idx = int(input("Select the model file to use: "))
    model_file = model_files[idx]
    print(f"Using model file: {model_file}")
elif len(model_files) == 1:
    model_file = model_files[0]
    print(f"Using model file: {model_file}")
else:
    raise ValueError("No model file found.")

with torch.no_grad():
    for img, lbl in train_loader:
        out = model(img.to(device))
        if isinstance(out, (list, tuple)):
            out = out[0] / out[1]
        print(f"Image shape: {img.shape}")
        print(f"Label shape: {lbl.shape}")
        print(f"Output shape: {out.shape}")
        break

model.load_state_dict(torch.load(f"checkpoints/{model_file}", map_location=device))
optimizer_ = torch.optim.SGD(
    model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
)
scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_, T_max=1)
criterion_ = TauCCE(ignore_index=ignore_index, tau=1.0)
model.to(device)

trainer = Trainer(
    model,
    optimizer_,
    criterion_,
    scheduler_,
    train_loader,
    test_loader,
    num_classes,
    ignore_index,
    device,
    lipschitz=args.type_param in ["ortho", "spectral"],
)
model.eval()
metrics = trainer.evaluate()
brief_epoch_log(1, 1, 0.0, metrics, show_cls=False)


def normalize(img):
    img = img - img.min()
    img = img / img.max()
    return img


def simple_cmap_pets():
    """Transform labels in {0, 1, 2} to RGB colors."""
    cmap = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )
    return cmap


def simple_cmap_kvasir():
    """Transform labels in {0, 1} to RGB colors."""
    cmap = np.array(
        [
            0,
            1,
        ]
    )
    return cmap


with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)

        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0] / outputs[1]

        if args.dataset == "voc":
            colormap = VOCSegmentation.cmap
        elif args.dataset == "cityscapes":
            colormap = Cityscapes.train_id_to_color
        elif args.dataset == "iiit_pets":
            colormap = simple_cmap_pets()
        elif args.dataset == "kvasir":
            colormap = simple_cmap_kvasir()
        else:
            raise NotImplementedError("Colormap not implemented for this dataset.")

        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        if ignore_index is not None:
            predictions[labels.cpu().numpy() == ignore_index] = colormap.shape[0] - 1
            labels[labels == ignore_index] = colormap.shape[0] - 1

        labels = labels.cpu().numpy()

        # plt.figure(figsize=(4, 12), dpi=300)
        for i in range(min(len(imgs), 4)):
            plt.subplot(3, 4, i + 1)
            plt.imshow(normalize(imgs[i].cpu().permute(1, 2, 0)))
            plt.axis("off")
            plt.title("Input Image")

            y_true = colormap[labels[i]]
            y_hat = colormap[predictions[i]]
            if y_true.max() > 1.0:
                y_true = y_true / 255.0
                y_hat = y_hat / 255.0

            plt.subplot(3, 4, i + 5)
            plt.imshow(y_true)
            plt.axis("off")
            plt.title("Ground Truth")

            plt.subplot(3, 4, i + 9)
            plt.imshow(y_hat)
            plt.axis("off")
            plt.title("Prediction")

        plt.tight_layout()
        os.makedirs("figs", exist_ok=True)
        plt.savefig("figs/inference_results.png", dpi=300)
        plt.close()

        break

from src.robustness.visualize import find_large_connected_components

# Robustness visualization - Inference time
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)

        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0] / outputs[1]

        idx = 2
        N_min = 200
        robustness_threshold = 0.25

        img = imgs[idx]
        C, H, W = img.shape
        logit = outputs[idx]
        lbl = labels[idx]
        pred = torch.argmax(outputs, dim=1)[idx]

        attacked_pred = pred.view(-1).clone()
        margins = ((F.relu(logit[1, :, :] - logit[0, :, :]) ** 2) / 2).view(-1)
        margins, idxs = torch.sort(margins, descending=False)
        margins = torch.sqrt(torch.cumsum(margins, dim=0))
        idx_r = torch.searchsorted(margins, robustness_threshold)
        to_missclassify = idxs[:idx_r]
        attacked_pred[to_missclassify] = 0
        attacked_pred = attacked_pred.view(H, W)
        attacked_np = attacked_pred.detach().cpu().numpy()
        gamma_emp = (margins[:idx_r] > 0).sum().item() / (pred == 1).sum().item()

        pred_np = pred.detach().cpu().numpy()

        pixel_subsets = find_large_connected_components(pred.cpu(), N_min=N_min)

        visual = torch.ones((img.size(1), img.size(2))).cpu() * 0.5
        threshold_visual = np.ones((img.size(1), img.size(2), 3), dtype=np.float32)

        classes = sorted(np.unique(pred_np).tolist())
        class_to_color = {
            0: 0,
            1: 1,
        }

        epsilons = []
        for subset in pixel_subsets:
            if not subset:
                continue

            coords = torch.tensor(subset, dtype=torch.long)
            rows, cols = coords[:, 0], coords[:, 1]

            logits_subset = logit.detach().cpu()[:, rows, cols]
            top2 = torch.topk(logits_subset, k=2, dim=0)
            margins = top2.values[0] - top2.values[1]

            margins, _ = torch.sort(margins, descending=False)
            margins = (margins**2) / 2.0
            margins = torch.sqrt(torch.cumsum(margins, dim=0))

            rob_idx = int(0.9 * len(subset))
            rob = margins[rob_idx]
            epsilons.append(rob.item() if top2.indices[0, rob_idx] == 1 else 0.0)

            visual[rows, cols] = rob

            if rob < robustness_threshold:
                threshold_visual[rows, cols] = [1.0, 1.0, 1.0]  # White
            else:
                class_label = pred_np[rows[0], cols[0]]
                color = class_to_color[class_label]
                threshold_visual[rows, cols] = color

        large_mask = np.zeros_like(pred_np, dtype=bool)
        for subset in pixel_subsets:
            if not subset:
                continue
            coords = np.asarray(subset, dtype=np.int64)
            large_mask[coords[:, 0], coords[:, 1]] = True

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), dpi=300)

        # Plot 1: Input Image with Robustness Heatmap Overlay
        ax1.imshow(img.cpu().permute(1, 2, 0).numpy())
        visual_numpy = visual.numpy()
        robustness = np.unique(visual_numpy)
        if len(robustness) > 1:
            colorbar = True
            log_norm = None
            if visual_numpy.max() > 0:
                log_norm = colors.LogNorm(
                    vmin=max(visual_numpy[visual_numpy > 0].min(), 1e-6),
                    vmax=visual_numpy.max(),
                )
        else:
            colorbar = False

        # Overlay the heatmap with transparency
        im = ax1.imshow(visual_numpy, cmap="jet", alpha=0.4)
        # norm=log_norm, alpha=0.4)
        ax1.set_title(r"Robust Pixel Subsets, $\gamma_\mathrm{stab}$=0.1", fontsize=14)
        ax1.text(
            0.5,
            -0.1,
            rf"Requires a budget larger than $\epsilon$={max(epsilons):.2f}",
            ha="center",
            transform=ax1.transAxes,
            fontsize=14,
        )
        ax1.axis("off")

        # Plot 2: WC FNR Image
        ax2.imshow(colormap[attacked_np])
        ax2.set_title(
            rf"Worst-Case Vanishing Attack, $\epsilon$={robustness_threshold}",
            fontsize=14,
        )
        ax2.text(
            0.5,
            -0.1,
            rf"{100*gamma_emp:.2f}% of 'polyp' pixel predictions changed to 'benign'",
            ha="center",
            transform=ax2.transAxes,
            fontsize=14,
        )
        ax2.axis("off")
        plt.tight_layout()
        os.makedirs("figs", exist_ok=True)
        plt.savefig("figs/robustness_visualization_inference.png", dpi=300)
        plt.close()
        break


with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)

        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0] / outputs[1]

        idx = 2
        N_min = 200
        robustness_threshold = 0.25

        img = imgs[idx]
        logit = outputs[idx]
        lbl = labels[idx]
        pred = torch.argmax(outputs, dim=1)[idx]
        C, H, W = img.shape

        attacked_pred = pred.view(-1).clone()
        attacked_lbl = lbl.view(-1).clone()
        margins = ((F.relu(lbl * (logit[1, :, :] - logit[0, :, :])) ** 2) / 2).view(-1)
        margins, idxs = torch.sort(margins, descending=False)
        margins = torch.sqrt(torch.cumsum(margins, dim=0))
        idx_r = torch.searchsorted(margins, robustness_threshold)
        idx_attack = idxs[:idx_r]
        mask_attack = attacked_lbl[idx_attack] == 1
        attacked_pred[idx_attack[mask_attack]] = 0
        attacked_pred = attacked_pred.view(H, W)

        tp = ((pred == 1) & (lbl == 1)).cpu()
        fp = ((pred == 1) & (lbl == 0)).cpu()
        fn = ((pred == 0) & (lbl == 1)).cpu()
        tn = ((pred == 0) & (lbl == 0)).cpu()

        tp_attack = ((attacked_pred == 1) & (lbl == 1)).cpu()
        fp_attack = ((attacked_pred == 1) & (lbl == 0)).cpu()
        fn_attack = ((attacked_pred == 0) & (lbl == 1)).cpu()

        fnr_clean = (fn.sum() / (tp.sum() + fn.sum())).item()
        fnr_attacked = (fn_attack.sum() / (tp_attack.sum() + fn_attack.sum())).item()

        colormap_st = {
            "tp": [0, 1, 0],
            "fp": [0, 0, 1],
            "fn": [1, 0, 0],
        }

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8), dpi=300)

        # Plot 1: Input Image with Predictions Overlay
        ax1.imshow(img.cpu().permute(1, 2, 0).numpy())
        ax1.imshow(pred.cpu().numpy(), cmap="tab20", alpha=0.5)
        ax1.set_title("Predictions", fontsize=14)
        ax1.axis("off")

        # Plot 2: TP, FP, FN Visualization
        st_map = np.ones((H, W, 3), dtype=np.float32)
        st_map[tp] = colormap_st["tp"]
        st_map[fp] = colormap_st["fp"]
        st_map[fn] = colormap_st["fn"]
        ax2.imshow(st_map)
        ax2.set_title("Clean Predictions", fontsize=14)
        # TP (Green), FP (Blue), FN (Red)", fontsize=14)
        ax2.text(
            0.5,
            -0.1,
            rf"FNR($X,Y$) = {fnr_clean:.2f}",
            ha="center",
            transform=ax2.transAxes,
            fontsize=14,
        )
        ax2.axis("off")

        # Plot 3: WC FNR Image
        st_map_attacked = np.ones((H, W, 3), dtype=np.float32)
        st_map_attacked[tp_attack] = colormap_st["tp"]
        st_map_attacked[fp_attack] = colormap_st["fp"]
        st_map_attacked[fn_attack] = colormap_st["fn"]
        ax3.imshow(st_map_attacked)
        ax3.set_title(
            rf"Worst-Case FNR Attack, $\epsilon$={robustness_threshold}", fontsize=14
        )
        plot_title = (
            r"$\max_{\tilde{X} \in \mathcal{B}_\epsilon(X)}$ FNR($\tilde{X},Y$) = "
            + f"{fnr_attacked:.2f}"
        )
        ax3.text(
            0.5,
            -0.1,
            plot_title,
            ha="center",
            transform=ax3.transAxes,
            fontsize=14,
        )
        ax3.axis("off")

        plt.tight_layout()
        os.makedirs("figs", exist_ok=True)
        plt.savefig("figs/robustness_visualization_labels.png", dpi=300)
        break
