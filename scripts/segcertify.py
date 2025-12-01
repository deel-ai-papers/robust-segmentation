import torch
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
from tqdm import tqdm
import time
from pathlib import Path
from src.smoothing.model import SmoothedModel
from src.smoothing import certify
import scipy.stats as sps


parser = argparse.ArgumentParser(description="train a segmentation model.")
parser.add_argument(
    "--dataset",
    type=str,
    choices=["voc", "cityscapes", "toy", "iiit_pets"],
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
    "--batch_size", type=int, default=100, help="batch size for training."
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
parser.add_argument(
    "--n_samples",
    type=int,
    default=100,
    help="number of MC iterations for voting + smoothing.",
)
parser.add_argument(
    "--prop_voting",
    type=float,
    default=0.1,
    help="proportion of MC iterations for majority class voting.",
)
parser.add_argument(
    "--sigma", type=float, default=0.5, help="standard deviation of Gaussian noise."
)
parser.add_argument(
    "--alpha", type=float, default=0.001, help="probabilistic error rate for smoothing."
)
parser.add_argument(
    "--tau",
    type=float,
    default=0.0,
    help="tau hyperparameter for the segcertify method.",
)
parser.add_argument(
    "--epsilon", type=float, default=0.17, help="epsilon radius to certify for."
)
parser.add_argument(
    "--max_bs",
    type=int,
    default=1000,
    help="maximum batch size to use for memory-safe smoothing inference.",
)
parser.add_argument(
    "--num_batches_eval",
    type=float,
    default=float("inf"),
    help="number of validation batches to use for eval",
)
args = parser.parse_args()

train_ds, test_ds, means, stds, num_classes, ignore_index = get_dataset(args)
print(f"training dataset size: {len(train_ds)}")

if args.tau == 0.0:
    from scipy.optimize import bisect

    def objective(tau):
        return args.sigma * sps.norm.ppf(tau) - args.epsilon

    print("Compute tau from radius and sigma:")
    new_tau = bisect(objective, a=0.5, b=1.0)

    print("New tau: ", new_tau)
    args.tau = new_tau

assert 0.5 <= args.tau < 1.0, "tau must be in [0.5, 1.0)"
assert 0.0 < args.alpha < 1.0, "alpha must be in (0, 1)"
assert 0.0 < args.epsilon, "epsilon must be positive"
assert 0.0 <= args.prop_voting <= 1.0, "prop_voting must be in [0, 1]"

# create dataloaders
train_loader = DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
)
test_loader = DataLoader(
    test_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
)

# get model
model = get_model(means, stds, in_channels=3, num_classes=num_classes, args=args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Find the files with the right prefix
lipconfig = (
    "rms"
    if args.klip and not args.learnable_coeffs
    else "klip"
    if args.klip
    else "1lip"
    if args.type_param != "unconstrained"
    else "nonlip"
)
prefix = f"{args.dataset}_{args.type_param}_{args.config}"
print("Prefix: ", prefix)
model_files = [
    f
    for f in os.listdir("checkpoints")
    if f.startswith(prefix) and f.endswith(".pth") and "optimizer" not in f
]
model_files = [f for f in model_files if lipconfig in f]
print(f"{model_files=}")

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

assert os.path.exists(
    f"checkpoints/{model_file}"
), f"Model file: {model_file} does not exist."
model.load_state_dict(torch.load(f"checkpoints/{model_file}", map_location=device))
optimizer_ = torch.optim.SGD(
    model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
)
scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_, T_max=1)
criterion_ = TauCCE(ignore_index=ignore_index, tau=1.0)
model.to(device)

# smoothing wrapper
smooth_model = SmoothedModel(
    model, sigma=args.sigma, n_mc=args.n_samples, max_bs=args.max_bs
)

with torch.no_grad():
    cra_smooth_imgs, cra_lip_imgs, num_imgs = 0, 0, 0
    cpt = 1
    for imgs, lbls in tqdm(test_loader):
        imgs, lbls = imgs.to(device), lbls.to(device)
        B, C, H, W = imgs.shape
        total_valid = (lbls != ignore_index).sum(dim=(1, 2))  # (B,)

        start_lip = time.time()
        clean_outputs = model(imgs)

        if isinstance(clean_outputs, tuple):
            clean_outputs = clean_outputs[0] / clean_outputs[1]

        clean_preds = clean_outputs.argmax(dim=1)

        ### Lipschitz Certification Process ###
        topk = torch.topk(clean_outputs, k=2, dim=1).values
        margin_top = topk[:, 0] - topk[:, 1]  # (B, H, W)
        correct = (clean_preds == lbls).float()
        margin = margin_top * correct
        margin = margin.view(B, -1)
        margin = (margin**2) / 2.0
        margin, _ = margin.sort(dim=1)
        margin = torch.cumsum(margin, dim=1)
        margin = torch.sqrt(margin)
        threshold = args.epsilon
        robust_mask = (margin >= threshold).long()

        num_imgs += B
        cra_lip_imgs += (robust_mask.sum(dim=1) / total_valid).sum()

        stop_lip = time.time()
        delta_lip = stop_lip - start_lip

        ### Smoothing Certification Process ###
        start_smooth = time.time()
        smooth_preds = smooth_model(imgs)

        ### Smoothing Certification Process ###
        for smooth_pred, lbl in zip(smooth_preds, lbls):
            pred = smooth_pred.view(args.n_samples, H * W)
            lbl = lbl.view(H * W).long()
            n0 = int(args.prop_voting * args.n_samples)
            n = args.n_samples - n0
            classes, radius, times = certify(
                num_classes,
                pred.cpu().numpy(),
                n0,
                n,
                args.sigma,
                args.tau,
                args.alpha,
                abstain=-1,
                parallel=False,
                correction="holm",
                kfwer_k=1,
            )
            cra_smooth_imgs += (classes == lbl.cpu().numpy()).sum() / (
                lbl != ignore_index
            ).sum()

        stop_smooth = time.time()
        delta_smooth = stop_smooth - start_smooth

        if cpt >= args.num_batches_eval:
            break
        else:
            cpt += 1

import pandas as pd

time_lip = float(delta_lip)
time_smooth = float(delta_smooth)

lip_cra = float(100 * cra_lip_imgs / num_imgs)
smooth_cra = float(100 * cra_smooth_imgs / num_imgs)

results_file = f"data/segcertify_results_{args.dataset}_{args.type_param}_{args.config}_{lipconfig}.csv"

row = {
    "num_samples": int(args.n_samples),
    "smooth_cra": smooth_cra,
    "lip_cra": lip_cra,
    "time_lip": time_lip,
    "time_smooth": time_smooth,
    "sigma": args.sigma,
    "epsilon": args.epsilon,
}
df = pd.DataFrame([row])

results_path = Path(results_file)
df.to_csv(results_path, mode="a", header=not results_path.exists(), index=False)
