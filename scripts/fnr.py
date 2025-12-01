import torch
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets import get_dataset
from src import get_model, get_optimizer, get_loss
from src.trainer import Trainer
from src.utils.logging import brief_epoch_log
from src.loss.taucce import TauCCE
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="train a segmentation model.")
parser.add_argument(
    "--dataset",
    type=str,
    default="kvasir",
    choices=["kvasir"],
    help="dataset to use for training.",
)
parser.add_argument(
    "--img_size", type=int, default=256, help="image size for training."
)
parser.add_argument("--square_imgs", action="store_true", help="use square images.")
parser.add_argument(
    "--data_root",
    type=str,
    default="../DATA/Kvasir-SEG",
    help="root directory of the dataset.",
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
parser.add_argument(
    "--epsilon", type=float, default=0.1, help="epsilon robustness radius."
)
parser.add_argument(
    "--fnr_threshold", type=float, default=0.95, help="epsilon robustness radius."
)
parser.add_argument(
    "--class_wc_iou", type=int, default=-1, help="class for worst-case iou."
)
parser.add_argument("--klip", action="store_true", help="use klip model.")
parser.add_argument("--learnable_coeffs", action="store_true", help="use klip model.")
parser.add_argument("--wandb_log", action="store_true", help="use klip model.")
parser.add_argument(
    "--config", type=str, default="S", help="configuration for the model."
)

args = parser.parse_args()

train_ds, test_ds, means, stds, num_classes, ignore_index = get_dataset(args)
print(f"training dataset size: {len(train_ds)}")

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
            lipconstant = out[1]
            out = out[0] / out[1]
        else:
            lipconstant = 1.0
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

model.eval()

from src.robustness.fnr import CertifiedFNR, FNRThresholdEpsilon

qone_metric = CertifiedFNR(epsilon=args.epsilon, lipconstant=lipconstant)
qtwo_metric = FNRThresholdEpsilon(lipconstant=lipconstant, threshold=args.fnr_threshold)


with torch.no_grad():
    for imgs, msks in test_loader:
        imgs, msks = imgs.to(device), msks.to(device)
        logits = model(imgs)
        if isinstance(logits, (list, tuple)):
            logits = logits[0] / logits[1]
        else:
            pass
        logits, msks = logits.cpu(), msks.cpu()
        qone_metric.update(logits, msks)
    metrics = qone_metric.compute()

print("Metrics (Q1): ", metrics)

with torch.no_grad():
    for imgs, msks in test_loader:
        imgs, msks = imgs.to(device), msks.to(device)
        logits = model(imgs)
        if isinstance(logits, (list, tuple)):
            logits = logits[0] / logits[1]
        else:
            pass
        logits, msks = logits.cpu(), msks.cpu()
        qtwo_metric.update(logits, msks)
    metrics = qtwo_metric.compute()

print("Metrics (Q2): ", metrics)
