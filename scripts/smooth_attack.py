import torch
import argparse
import os
import sys
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets import get_dataset
from src import get_model
from src.trainer import Trainer
from src.loss.taucce import TauCCE
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="Evaluate smoothed classifier robustness.")
parser.add_argument("--dataset", type=str, choices=["voc", "cityscapes", "toy", "iiit_pets"], required=True)
parser.add_argument("--img_size", type=int, default=512)
parser.add_argument("--square_imgs", action="store_true")
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--year", type=str, default="2012")
parser.add_argument("--download", action="store_true")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--type_param", type=str, choices=["ortho", "spectral", "unconstrained"], default="unconstrained")
parser.add_argument("--klip", action="store_true")
parser.add_argument("--learnable_coeffs", action="store_true")
parser.add_argument("--config", type=str, default="S")
parser.add_argument("--model_file", type=str, default=None)
parser.add_argument("--sigma", type=float, default=0.25)
parser.add_argument("--n0", type=int, default=100)
parser.add_argument("--n", type=int, default=1000)
parser.add_argument("--tau", type=float, default=0.75)
parser.add_argument("--alpha", type=float, default=0.001)
parser.add_argument("--epsilon", type=float, default=0.5)
parser.add_argument("--num_steps", type=int, default=40)
parser.add_argument("--n_attack_samples", type=int, default=32)
parser.add_argument("--p_norm", type=float, default=2)
parser.add_argument("--num_batches", type=int, default=15)
parser.add_argument("--output_csv", type=str, default=None)
args = parser.parse_args()

train_ds, test_ds, means, stds, num_classes, ignore_index = get_dataset(args)
print(f"Test dataset size: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

model = get_model(means, stds, in_channels=3, num_classes=num_classes, args=args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if args.model_file is not None:
    model_file = args.model_file
    print(f"Using model file: {model_file}")
else:
    lipconfig = (
        "rms" if args.klip and not args.learnable_coeffs
        else "klip" if args.klip
        else "1lip" if args.type_param != "unconstrained"
        else "nonlip"
    )
    prefix = f"{args.dataset}_{args.type_param}_{args.config}"
    model_files = [f for f in os.listdir("checkpoints") if f.startswith(prefix) and f.endswith(".pth")]
    model_files = [f for f in model_files if lipconfig in f if "optimizer" not in f]

    if len(model_files) > 1:
        print("Multiple model files found:")
        for i, f in enumerate(model_files):
            print(f"{i}: {f}")
        idx = int(input("Select the model file to use: "))
        model_file = model_files[idx]
    elif len(model_files) == 1:
        model_file = model_files[0]
    else:
        raise ValueError("No model file found.")
    print(f"Using model file: {model_file}")

model.load_state_dict(torch.load(f"checkpoints/{model_file}", map_location=device))
model.eval()

optimizer_ = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_, T_max=1)
criterion_ = TauCCE(ignore_index=ignore_index, tau=1.0)

trainer = Trainer(
    model, optimizer_, criterion_, scheduler_,
    train_loader, test_loader, num_classes, ignore_index, device,
)

metrics = trainer.attack_smooth(
    sigma=args.sigma,
    n0=args.n0,
    n=args.n,
    tau=args.tau,
    alpha=args.alpha,
    epsilon=args.epsilon,
    num_steps=args.num_steps,
    n_attack_samples=args.n_attack_samples,
    p=args.p_norm,
    num_batches=args.num_batches,
)

print("Smooth attack metrics:", metrics)

if args.output_csv is not None:
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    file_exists = os.path.exists(args.output_csv)

    row = {
        "model_file": model_file,
        "dataset": args.dataset,
        "sigma": args.sigma,
        "tau": args.tau,
        "epsilon": args.epsilon,
        "num_steps": args.num_steps,
        "num_batches": args.num_batches,
        **metrics,
    }

    with open(args.output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Metrics saved to {args.output_csv}")
