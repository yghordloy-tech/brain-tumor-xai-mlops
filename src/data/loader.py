"""
src/data/loader.py
==================
PyTorch data-loading pipeline for the Brain Tumor MRI Classification project.

Uses ``torchvision.datasets.ImageFolder`` – the raw dataset is expected to
live in class-named sub-folders::

    data/raw/Training/<class_name>/
    data/raw/Testing/<class_name>/

Applies ``random_split`` to carve out an 85 % Train / 15 % Validation
partition from the training directory. Trains with augmentation; validation
and test sets get only resize + normalisation. Builds a
``WeightedRandomSampler`` for the training ``DataLoader`` to mitigate
class imbalance.
"""

from __future__ import annotations

import pathlib
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from torchvision import datasets, transforms

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "config.yaml"


def _load_config(config_path=None):
    """Load and return the YAML configuration dictionary."""
    path = pathlib.Path(config_path) if config_path else _DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_train_transforms(cfg: dict) -> transforms.Compose:
    """Return the training transform pipeline with data augmentation."""
    image_size = cfg["data"]["image_size"]
    aug = cfg["augmentation"]
    norm = aug["normalize"]

    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.3, 0.9)),
        transforms.RandomAffine(
            degrees=aug["rotation_limit"],
            translate=(0.1, 0.1),
            shear=10
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(p=aug["horizontal_flip_prob"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ])


def get_eval_transforms(cfg: dict) -> transforms.Compose:
    """Return the evaluation transform pipeline (validation & test).

    No augmentation — only resize, tensor conversion, and normalisation.
    """
    image_size = cfg["data"]["image_size"]
    norm = cfg["augmentation"]["normalize"]

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ])


class TransformedSubset(Subset):
    """A ``Subset`` wrapper that applies its own transform instead of the
    one baked into the underlying dataset.

    Necessary because ``random_split`` returns plain ``Subset`` objects that
    delegate ``__getitem__`` to the parent ``ImageFolder``. Different
    transforms are needed for train vs. validation splits even though they
    share the same parent dataset.
    """

    def __init__(self, subset: Subset, transform: transforms.Compose) -> None:
        super().__init__(subset.dataset, subset.indices)
        self.transform = transform

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        image, label = self.dataset.samples[real_idx]

        from PIL import Image
        image = Image.open(image).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _build_weighted_sampler(subset: Subset) -> WeightedRandomSampler:
    """Create a ``WeightedRandomSampler`` for a given ``Subset``.

    Each sample receives a weight equal to ``1 / class_count`` so that
    underrepresented classes are up-sampled during training.
    """
    targets = np.array([subset.dataset.targets[i] for i in subset.indices])

    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts.astype(np.float64)
    sample_weights = class_weights[targets]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_dataloaders(
    config_path=None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """Build and return ``(train_loader, val_loader, test_loader, class_to_name)``.

    All hyper-parameters are read from *config_path* (defaults to
    ``configs/config.yaml`` at the project root).
    """
    cfg = _load_config(config_path)

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]
    pin_memory = cfg["data"]["pin_memory"]
    seed = cfg["project"]["seed"]

    raw_dir = _PROJECT_ROOT / cfg["paths"]["raw_data_dir"]
    train_dir = raw_dir / "Training"
    test_dir = raw_dir / "Testing"

    eval_tf = get_eval_transforms(cfg)
    train_tf = get_train_transforms(cfg)

    full_train_dataset = datasets.ImageFolder(root=str(train_dir), transform=eval_tf)
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=eval_tf)

    class_to_name: Dict[int, str] = {v: k for k, v in full_train_dataset.class_to_idx.items()}

    total = len(full_train_dataset)
    val_len = int(total * 0.15)
    train_len = total - val_len

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_train_dataset, [train_len, val_len], generator=generator
    )

    train_subset = TransformedSubset(train_subset, transform=train_tf)
    val_subset = TransformedSubset(val_subset, transform=eval_tf)

    sampler = _build_weighted_sampler(train_subset)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(
        f"[loader] Dataset splits  — "
        f"Train: {train_len:,}  |  Val: {val_len:,}  |  Test: {len(test_dataset):,}"
    )
    print(f"[loader] Classes         — {class_to_name}")
    print(f"[loader] Batch size      — {batch_size}")
    print(f"[loader] Image size      — {cfg['data']['image_size']}×{cfg['data']['image_size']}")

    return train_loader, val_loader, test_loader, class_to_name


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_to_name = get_dataloaders()

    images, labels = next(iter(train_loader))
    print(f"\n[smoke-test] Train batch shape : {images.shape}")
    print(f"[smoke-test] Train label shape : {labels.shape}")
    print(f"[smoke-test] Label dtype       : {labels.dtype}")
    print(f"[smoke-test] Pixel range       : [{images.min():.3f}, {images.max():.3f}]")
