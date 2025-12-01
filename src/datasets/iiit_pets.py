import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms.v2 as v2
from PIL import Image
import numpy as np


class OxfordIIITPet(data.Dataset):
    to_img = v2.ToPILImage()

    def __init__(self, root, train: bool, transforms: callable):
        super(OxfordIIITPet, self).__init__()
        assert isinstance(train, bool), "train must be a boolean value"
        if train:
            train_dataset = torchvision.datasets.OxfordIIITPet(
                root=root,
                transform=self.to_img,
                target_transform=self.to_img,
                download=True,
                target_types="segmentation",
                split="trainval",
            )
            self.dataset = train_dataset
        else:
            test_dataset = torchvision.datasets.OxfordIIITPet(
                root=root,
                transform=self.to_img,
                target_transform=self.to_img,
                download=True,
                target_types="segmentation",
                split="test",
            )
            self.dataset = test_dataset
        self.trfms = transforms

    def __getitem__(self, index):
        img, mask = self.dataset[index]
        if self.trfms is not None:
            img, mask = self.trfms(img, mask)
        mask = mask - 1
        return img, mask

    def __len__(self):
        return len(self.dataset)
