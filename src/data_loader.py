import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    import medmnist
    from medmnist import INFO
except Exception as e:  # pragma: no cover
    raise RuntimeError("medmnist package is required. Install via `pip install medmnist`.") from e


DEFAULT_IMAGE_SIZE = 224  # Upsample from 28x28 to 224x224 for parity across CNN/ViT
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_transforms(image_size: int = DEFAULT_IMAGE_SIZE, augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    test_tfms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    # remove identity if augment disabled
    train_tfms = [t for t in train_tfms if not isinstance(t, transforms.Lambda)]
    return transforms.Compose(train_tfms), transforms.Compose(test_tfms)


def get_dataloaders(
    data_dir: str = "data",
    batch_size: int = 128,
    num_workers: int = 2,
    image_size: int = DEFAULT_IMAGE_SIZE,
    augment: bool = True,
    download: bool = True,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Returns train/val/test dataloaders for BloodMNIST and number of classes.

    Ensures consistent preprocessing across CNN and ViT.
    """
    os.makedirs(data_dir, exist_ok=True)

    info = INFO["bloodmnist"]
    n_classes = len(info["label"])

    DataClass = getattr(medmnist, info["python_class"])  # BloodMNIST

    train_tfms, test_tfms = _build_transforms(image_size=image_size, augment=augment)

    train_set = DataClass(split="train", transform=train_tfms, download=download, root=data_dir)
    val_set = DataClass(split="val", transform=test_tfms, download=download, root=data_dir)
    test_set = DataClass(split="test", transform=test_tfms, download=download, root=data_dir)

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }

    train_loader = DataLoader(train_set, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader, n_classes


__all__ = [
    "get_dataloaders",
    "set_seed",
    "DEFAULT_IMAGE_SIZE",
]
