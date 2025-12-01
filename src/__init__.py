import torch
from klip.deeplab import KLipDeepLabV3, deeplabv3_klipschitz
from onelip.deeplab import LipDeepLabV3, deeplabv3_lipschitz
from allnets.deeplab import DeepLabV3, deeplabv3_unconstrained


def get_model(mean, std, in_channels, num_classes, args):
    if args.type_param in ["ortho", "spectral"]:
        if args.klip:
            model = deeplabv3_klipschitz(
                mean,
                std,
                in_channels,
                num_classes,
                config=args.config,
                param=args.type_param,
                learnable_coeffs=args.learnable_coeffs,
            )
        else:
            model = deeplabv3_lipschitz(
                mean,
                std,
                in_channels,
                num_classes,
                config=args.config,
                param=args.type_param,
            )
        return model
    elif args.type_param == "unconstrained":
        model = deeplabv3_unconstrained(
            mean, std, in_channels, num_classes, config=args.config
        )
        return model
    else:
        raise ValueError(f"Unknown type_param: {args.type_param}")


from loss.taucce import TauCCE
from loss.cosine import ClassicCosineLoss, NaiveCosineLoss
from loss.hkr import HKRLoss


def get_loss(args, ignore_index=-100):
    if args.loss == "tau_cce":
        return TauCCE(tau=args.tau, ignore_index=ignore_index, weight=None)
    elif args.loss == "cosine":
        return ClassicCosineLoss(ignore_index=ignore_index)
    elif args.loss == "naive_cosine":
        return NaiveCosineLoss(ignore_index=ignore_index)
    elif args.loss == "hkr":
        return HKRLoss(
            alpha=args.alpha,
            min_margin=args.min_margin,
            temperature=args.temperature,
            ignore_index=ignore_index,
        )
    else:
        raise ValueError(f"Unknown loss: {args.loss}")


def get_optimizer(model, args):
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd
        )
    elif args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.95, 0.95)
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
