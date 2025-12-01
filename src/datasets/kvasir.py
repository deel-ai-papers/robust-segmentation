from pathlib import Path
import random
from PIL import Image
from torch.utils.data import Dataset

from src.utils import ext_transforms as et


class Kvasir(Dataset):
    """
    Flat layout:
      root/images/*.{jpg,jpeg,png,bmp,tif,tiff}
      root/masks/*.{png,bmp,tif,tiff}

    Reproducible 70/10/20 split via fixed-seed shuffle.
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    MSK_EXTS = {".jpg", ".png", ".bmp", ".tif", ".tiff"}
    _SEED = 1337  # change here if you want a different fixed split

    def __init__(self, root, split="train", transform=None):
        assert split in {"train", "test"}
        self.root = Path(root)
        self.transform = transform
        self.split = split

        images_dir = self.root / "images"
        masks_dir = self.root / "masks"

        def collect(folder: Path, allowed_exts: set):
            out = {}
            if folder.exists():
                for p in folder.iterdir():
                    if p.is_file() and p.suffix.lower() in allowed_exts:
                        out[p.stem] = p
            return out

        img_files = collect(images_dir, self.IMG_EXTS)
        msk_files = collect(masks_dir, self.MSK_EXTS)

        all_ids = sorted(set(img_files) & set(msk_files))
        if not all_ids:
            raise RuntimeError(f"No paired files under {images_dir} and {masks_dir}.")

        # Reproducible shuffle
        rng = random.Random(self._SEED)
        rng.shuffle(all_ids)

        n = len(all_ids)
        n_train = int(n * 0.70)
        n_test = n - n_train
        # test uses the remainder
        if split == "train":
            ids = all_ids[:n_train]
        elif split == "test":
            ids = all_ids[n_train:]

        self.ids = ids
        self.img_files = img_files
        self.msk_files = msk_files

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids[idx]
        image = Image.open(self.img_files[uid])  # original mode
        target = Image.open(self.msk_files[uid]).convert("L")  # id-encoded mask

        if self.transform:
            image, target = self.transform(image, target)

        return image, target


if __name__ == "__main__":
    train_transform = et.ExtCompose(
        [
            et.ExtResize(size=img_sz),
            et.ExtRandomCrop(size=img_sz, pad_if_needed=True),
            et.ExtRandomRotation(10),
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

    train_ds = Kvasir(
        root="../DATA/Kvasir-SEG/", split="train", transform=train_transform
    )
    test_ds = Kvasir(root="../DATA/Kvasir-SEG/", split="test", transform=test_transform)

    # import matplotlib.pyplot as plt
    # for img, msk in train_ds:
    #     plt.figure()
    #     plt.subplot(1,2,1)
    #     plt.imshow(img)
    #     plt.subplot(1,2,2)
    #     plt.imshow(msk)
    #     plt.savefig("train_img.png")
    #     break

    # for img, msk in test_ds:
    #     plt.figure()
    #     plt.subplot(1,2,1)
    #     plt.imshow(img)
    #     plt.subplot(1,2,2)
    #     plt.imshow(msk)
    #     plt.savefig("test_img.png")
    #     break
