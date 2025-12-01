import torch
from .voc import VOCSegmentation
from .cityscapes import Cityscapes
from .toy_data import NoisySquaresSegmentationDataset
from .iiit_pets import OxfordIIITPet
from .kvasir import Kvasir
from PIL import Image

from src.utils import ext_transforms as et


def get_dataset(opts):
    """Dataset And Augmentation"""

    means = {
        "toy": [0.5, 0.5, 0.5],
        "voc": [0.485, 0.456, 0.406],
        "cityscapes": [0.286, 0.325, 0.283],
        "iiit_pets": [0.485, 0.456, 0.406],
        "kvasir": [0.5, 0.5, 0.5],
    }[opts.dataset]
    stds = {
        "toy": [0.2, 0.2, 0.2],
        "voc": [0.229, 0.224, 0.225],
        "cityscapes": [0.176, 0.180, 0.177],
        "iiit_pets": [0.229, 0.224, 0.225],
        "kvasir": [0.5, 0.5, 0.5],
    }[opts.dataset]
    num_classes = {
        "toy": 2,
        "voc": 21,
        "cityscapes": 19,
        "iiit_pets": 3,
        "kvasir": 2,
    }[opts.dataset]

    if opts.dataset == "toy":
        full_ds = NoisySquaresSegmentationDataset(
            length=10_000,
            image_size=(64, 64),
            square_size=(10, 24),
            gaussian_noise_std=0.02,
        )
        train_dst, val_dst = torch.utils.data.random_split(
            full_ds, [8000, 2000], generator=torch.Generator().manual_seed(42)
        )
        ignore_index = 255

    if not opts.square_imgs and opts.dataset == "cityscapes":
        img_sz = (opts.img_size, 2 * opts.img_size)
    else:
        img_sz = (opts.img_size, opts.img_size)

    if opts.dataset == "kvasir":
        train_transform = et.ExtCompose(
            [
                et.ExtResize(size=img_sz, interpolation=Image.NEAREST),
                et.ExtRandomScale((0.9, 1.2)),
                et.ExtRandomCrop(size=img_sz, pad_if_needed=True),
                et.ExtResize(size=img_sz, interpolation=Image.NEAREST),
                et.ExtRandomRotation(30),
                et.ExtRandomHorizontalFlip(),
                et.ExtRandomVerticalFlip(),
                et.ExtToTensor(),
                et.ExtBinaryLabels(),
            ]
        )
        val_transform = et.ExtCompose(
            [
                et.ExtResize(size=img_sz, interpolation=Image.NEAREST),
                et.ExtToTensor(),
                et.ExtBinaryLabels(),
            ]
        )
        train_dst = Kvasir(
            root=opts.data_root, split="train", transform=train_transform
        )
        val_dst = Kvasir(root=opts.data_root, split="test", transform=val_transform)
        ignore_index = 255

    if opts.dataset == "voc":
        train_transform = et.ExtCompose(
            [
                et.ExtResize(size=img_sz),
                et.ExtRandomScale((0.5, 2.0)),
                et.ExtRandomCrop(size=img_sz, pad_if_needed=True),
                et.ExtRandomHorizontalFlip(),
                et.ExtToTensor(),
            ]
        )
        val_transform = et.ExtCompose(
            [
                et.ExtResize(size=img_sz),
                et.ExtToTensor(),
            ]
        )
        train_dst = VOCSegmentation(
            root=opts.data_root,
            year=opts.year,
            image_set="train",
            download=opts.download,
            transform=train_transform,
        )
        val_dst = VOCSegmentation(
            root=opts.data_root,
            year=opts.year,
            image_set="val",
            download=False,
            transform=val_transform,
        )
        ignore_index = 255

    if opts.dataset == "cityscapes":
        train_transform = et.ExtCompose(
            [
                et.ExtResize(size=img_sz),
                et.ExtRandomCrop(size=img_sz),
                et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                et.ExtRandomHorizontalFlip(),
                et.ExtRandomRotation(10),
                et.ExtToTensor(),
            ]
        )

        val_transform = et.ExtCompose(
            [
                et.ExtResize(size=img_sz),
                et.ExtToTensor(),
            ]
        )

        train_dst = Cityscapes(
            root=opts.data_root, split="train", transform=train_transform
        )
        val_dst = Cityscapes(root=opts.data_root, split="val", transform=val_transform)
        ignore_index = 255

    if opts.dataset == "iiit_pets":
        train_transform = et.ExtCompose(
            [
                et.ExtResize(size=img_sz),
                et.ExtRandomCrop(size=img_sz),
                et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                et.ExtRandomHorizontalFlip(),
                et.ExtRandomRotation(10),
                et.ExtToTensor(),
            ]
        )

        val_transform = et.ExtCompose(
            [
                et.ExtResize(size=img_sz),
                et.ExtToTensor(),
            ]
        )

        train_dst = OxfordIIITPet(
            root=opts.data_root, train=True, transforms=train_transform
        )
        val_dst = OxfordIIITPet(
            root=opts.data_root, train=False, transforms=val_transform
        )
        ignore_index = 255

    return train_dst, val_dst, means, stds, num_classes, ignore_index
