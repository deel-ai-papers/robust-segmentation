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
from torch.utils.data import DataLoader
from src.loss.taucce import TauCCE

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
parser.add_argument("--epochs", type=int, default=50, help="number of training epochs.")
parser.add_argument(
    "--optimizer", type=str, choices=["sgd", "adam", "adamw"], default="sgd"
)
parser.add_argument("--lr", type=float, default=0.01, help="learning rate.")
parser.add_argument("--momentum", type=float, default=0.9, help="sgd momentum.")
parser.add_argument("--wd", type=float, default=1e-4, help="weight decay.")
parser.add_argument(
    "--loss",
    type=str,
    choices=["tau_cce", "cosine", "naive_cosine", "hkr"],
    default="tau_cce",
    help="loss function to use.",
)
parser.add_argument(
    "--tau", type=float, default=0.5, help="tau parameter for taucce loss."
)
parser.add_argument(
    "--alpha", type=float, default=0.95, help="alpha parameter for hkr loss."
)
parser.add_argument(
    "--min_margin", type=float, default=0.01, help="min margin for hkr loss."
)
parser.add_argument(
    "--temperature", type=float, default=5.0, help="temperature for hkr loss."
)
parser.add_argument(
    "--model_weights", type=str, default="", help="load some pre-trained weights."
)
parser.add_argument(
    "--sigma_train",
    type=float,
    default=0.0,
    help="gaussian noise to add during training.",
)

args = parser.parse_args()

train_ds, test_ds, means, stds, num_classes, ignore_index = get_dataset(args)

# create dataloaders
train_loader = DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
)
test_loader = DataLoader(
    test_ds, batch_size=2 * args.batch_size, shuffle=False, num_workers=4
)

# get model
model = get_model(means, stds, in_channels=3, num_classes=num_classes, args=args)

if args.dataset != "kvasir":
    criterion = get_loss(args, ignore_index)
else:
    criterion = TauCCE(tau=args.tau, weight=[0.8, 1.3])

optimizer = get_optimizer(model, args)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=1e-6
)

if args.model_weights != "":
    assert os.path.isfile(args.model_weights)
    opt_file_path = args.model_weights.replace(".pth", "_optimizer.pth")

    model.train()
    with torch.no_grad():
        for imgs, lbls in train_loader:
            out = model(imgs)
            break
    model.load_state_dict(torch.load(args.model_weights))
    print(f"loaded model weights from {args.model_weights}")
    if os.path.isfile(opt_file_path):
        optimizer.load_state_dict(torch.load(opt_file_path))
        print(f"loaded optimizer state from {opt_file_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer = Trainer(
    model,
    optimizer,
    criterion,
    scheduler,
    train_loader,
    test_loader,
    num_classes,
    ignore_index,
    device,
    lipschitz=args.type_param in ["ortho", "spectral"],
    sigma_train=args.sigma_train,
)

lipconfig = (
    "rms"
    if args.klip and not args.learnable_coeffs
    else "klip"
    if args.klip
    else "1lip"
    if args.type_param != "unconstrained"
    else "nonlip"
)
model_name = f"{args.dataset}_{args.type_param}_{args.config}_{args.loss}_{lipconfig}"
if args.wandb_log:
    import wandb

    run = wandb.init(
        project="cvpr-lipschitz-segmentation",
        name=model_name,
        config=args,
    )
else:
    run = None

for epoch in range(1, args.epochs + 1):
    train_loss = trainer.train_one_epoch(run)
    metrics = trainer.evaluate()
    brief_epoch_log(epoch, args.epochs, train_loss, metrics, show_cls=True)

    if args.wandb_log:
        run.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in metrics.items()},
            }
        )

# save the model
os.makedirs("checkpoints", exist_ok=True)

torch.save(model.state_dict(), f"checkpoints/{model_name}.pth")
torch.save(optimizer.state_dict(), f"checkpoints/{model_name}_optimizer.pth")
