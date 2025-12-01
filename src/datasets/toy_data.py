import math
import random
from typing import Tuple, Sequence, Optional

import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class NoisySquaresSegmentationDataset(Dataset):
    def __init__(
        self,
        length: int = 10_000,
        image_size: Tuple[int, int] = (64, 64),
        square_size: Sequence[int] = (8, 24),
        n_squares: Sequence[int] = (1, 3),
        bg_val_range: Tuple[float, float] = (0.0, 0.35),
        fg_val_range: Tuple[float, float] = (0.65, 1.0),
        gaussian_noise_std: float = 0.0,
        transforms: Optional[callable] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.length = length
        self.H, self.W = image_size

        if len(square_size) == 2 and isinstance(square_size[0], int):
            self.square_size_range = square_size
        else:
            raise ValueError("square_size must be a 2-tuple of ints")

        self.n_squares_range = n_squares
        self.bg_val_range = bg_val_range
        self.fg_val_range = fg_val_range
        self.gaussian_noise_std = gaussian_noise_std
        self.transforms = transforms

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        assert self.bg_val_range[1] < self.fg_val_range[0], "Classes must be separable"

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self._sample_background()
        mask = torch.zeros((self.H, self.W), dtype=torch.long)

        n_sq = random.randint(*self.n_squares_range)
        for _ in range(n_sq):
            self._draw_square(img, mask)

        if self.gaussian_noise_std > 0:
            noise = torch.randn_like(img) * self.gaussian_noise_std
            img = torch.clamp(img + noise, 0.0, 1.0)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, mask

    def _sample_background(self) -> torch.Tensor:
        """Return a (1, H, W) tensor of background noise."""
        low, high = self.bg_val_range
        bg = torch.empty((1, self.H, self.W)).uniform_(low, high)
        return bg

    def _draw_square(self, img: torch.Tensor, mask: torch.Tensor) -> None:
        """Draw one filled square with random size, position, and intensity."""
        side = random.randint(*self.square_size_range)

        top = random.randint(0, self.H - side)
        left = random.randint(0, self.W - side)
        bottom = top + side
        right = left + side

        val = random.uniform(*self.fg_val_range)

        img[:, top:bottom, left:right] = val
        mask[top:bottom, left:right] = 1


if __name__ == "__main__":
    ds = NoisySquaresSegmentationDataset(length=4, gaussian_noise_std=0.05)
    img, mask = ds[0]
    print("Image shape:", img.shape, img.min().item(), img.max().item())
    print("Mask  shape:", mask.shape, mask.unique())
