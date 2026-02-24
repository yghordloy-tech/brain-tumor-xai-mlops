"""
src/data/loader.py
==================
PyTorch data-loading pipeline for the Brain Tumor MRI Classification project.

* Reads **all** settings (paths, image size, batch size, augmentation
  parameters, normalization stats) dynamically from ``configs/config.yaml``.
* Uses ``torchvision.datasets.ImageFolder`` – the raw dataset is expected to
  live in class-named sub-folders::

      data/raw/Training/<class_name>/
      data/raw/Testing/<class_name>/

* Applies ``random_split`` to carve out an **85 % Train / 15 % Validation**
  partition from the training directory.
* Trains with augmentation (``RandomHorizontalFlip``, ``RandomRotation``);
  validation and test sets get only resize + normalisation.
* Builds a ``WeightedRandomSampler`` for the training ``DataLoader`` so that
  every class is sampled proportionally, mitigating class imbalance.
"""

from __future__ import annotations

import os
import pathlib
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]  # …/brain-tumor-xai-mlops
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "config.yaml"


def _load_config(config_path: str | pathlib.Path | None = None) -> dict:
    """Load and return the YAML configuration dictionary."""
    path = pathlib.Path(config_path) if config_path else _DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------


def get_train_transforms(cfg: dict) -> transforms.Compose:
    """Return the training transform pipeline **with** ultra-aggressive data augmentation."""
    image_size: int = cfg["data"]["image_size"]
    aug = cfg["augmentation"]
    norm = aug["normalize"]

    return transforms.Compose(
        [
            # 1. Aggressive Zoom: Force it to look at tiny 30% patches
            transforms.RandomResizedCrop(image_size, scale=(0.3, 0.9)), 
            
            # 2. Spatial Distortion: Shift the brain off-center and stretch it
            transforms.RandomAffine(
                degrees=aug["rotation_limit"], 
                translate=(0.1, 0.1), 
                shear=10
            ),
            
            # 3. Contrast Jitter: Stop it from memorizing the brightness of ventricles
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            
            transforms.RandomHorizontalFlip(p=aug["horizontal_flip_prob"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm["mean"], std=norm["std"]),
        ]
    )


def get_eval_transforms(cfg: dict) -> transforms.Compose:
    """Return the evaluation transform pipeline (validation & test).

    No augmentation — only resize → tensor → normalise.
    """
    image_size: int = cfg["data"]["image_size"]
    norm = cfg["augmentation"]["normalize"]

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm["mean"], std=norm["std"]),
        ]
    )


# ---------------------------------------------------------------------------
# Subset with overridden transform
# ---------------------------------------------------------------------------


class TransformedSubset(Subset):
    """A ``Subset`` wrapper that applies its **own** transform instead of the
    one baked into the underlying dataset.

    This is necessary because ``random_split`` returns plain ``Subset``
    objects that delegate ``__getitem__`` to the parent ``ImageFolder``.
    We need different transforms for the train vs. validation splits even
    though they share the same parent dataset.
    """

    def __init__(self, subset: Subset, transform: transforms.Compose) -> None:
        super().__init__(subset.dataset, subset.indices)
        self.transform = transform

    def __getitem__(self, idx: int):
        # Fetch the *original* image (PIL) and label from the parent dataset.
        real_idx = self.indices[idx]
        image, label = self.dataset.samples[real_idx]

        # Load from disk using PIL (ImageFolder stores paths, not tensors).
        from PIL import Image

        image = Image.open(image).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# Weighted sampler
# ---------------------------------------------------------------------------


def _build_weighted_sampler(subset: Subset) -> WeightedRandomSampler:
    """Create a ``WeightedRandomSampler`` for a given ``Subset``.

    Each sample receives a weight equal to ``1 / class_count`` so that
    underrepresented classes are up-sampled during training.
    """
    targets = np.array(
        [subset.dataset.targets[i] for i in subset.indices]
    )

    class_counts = np.bincount(targets)
    # Weight per *class*
    class_weights = 1.0 / class_counts.astype(np.float64)
    # Weight per *sample*
    sample_weights = class_weights[targets]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_dataloaders(
    config_path: str | pathlib.Path | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """Build and return ``(train_loader, val_loader, test_loader, class_to_name)``.

    All hyper-parameters are read from *config_path* (defaults to
    ``configs/config.yaml`` at the project root).

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the YAML configuration file.

    Returns
    -------
    train_loader : DataLoader
        Training split (85 %) with augmentation and ``WeightedRandomSampler``.
    val_loader : DataLoader
        Validation split (15 %) with eval transforms only.
    test_loader : DataLoader
        Full test set with eval transforms only.
    class_to_name : dict[int, str]
        Mapping of class index → human-readable class name (from ImageFolder).
    """

    # ---- 1. Load configuration ------------------------------------------------
    cfg = _load_config(config_path)

    batch_size: int = cfg["training"]["batch_size"]
    num_workers: int = cfg["data"]["num_workers"]
    pin_memory: bool = cfg["data"]["pin_memory"]
    seed: int = cfg["project"]["seed"]

    raw_dir = _PROJECT_ROOT / cfg["paths"]["raw_data_dir"]
    train_dir = raw_dir / "Training"
    test_dir = raw_dir / "Testing"

    # ---- 2. Load full datasets with a *base* (eval) transform -----------------
    #  We use eval transforms on the parent ImageFolder because the train
    #  subset will later get its own augmented transform via TransformedSubset.
    eval_tf = get_eval_transforms(cfg)
    train_tf = get_train_transforms(cfg)

    full_train_dataset = datasets.ImageFolder(root=str(train_dir), transform=eval_tf)
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=eval_tf)

    # Class index → name mapping (e.g. {0: "glioma", 1: "meningioma", …})
    class_to_name: Dict[int, str] = {v: k for k, v in full_train_dataset.class_to_idx.items()}

    # ---- 3. Split training data → 85 % train / 15 % validation ---------------
    total = len(full_train_dataset)
    val_len = int(total * 0.15)
    train_len = total - val_len

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_train_dataset, [train_len, val_len], generator=generator
    )

    # Wrap subsets so each gets its *own* transform.
    train_subset = TransformedSubset(train_subset, transform=train_tf)
    val_subset = TransformedSubset(val_subset, transform=eval_tf)

    # ---- 4. Build WeightedRandomSampler for training --------------------------
    sampler = _build_weighted_sampler(train_subset)

    # ---- 5. Construct DataLoaders ---------------------------------------------
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,           # weighted sampling → do NOT set shuffle=True
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

    # ---- 6. Summary -----------------------------------------------------------
    print(
        f"[loader] Dataset splits  — "
        f"Train: {train_len:,}  |  Val: {val_len:,}  |  Test: {len(test_dataset):,}"
    )
    print(f"[loader] Classes         — {class_to_name}")
    print(f"[loader] Batch size      — {batch_size}")
    print(f"[loader] Image size      — {cfg['data']['image_size']}×{cfg['data']['image_size']}")

    return train_loader, val_loader, test_loader, class_to_name


# ---------------------------------------------------------------------------
# Quick smoke-test when running the file directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_to_name = get_dataloaders()

    # Grab one batch to verify shapes & dtypes
    images, labels = next(iter(train_loader))
    print(f"\n[smoke-test] Train batch shape : {images.shape}")
    print(f"[smoke-test] Train label shape : {labels.shape}")
    print(f"[smoke-test] Label dtype       : {labels.dtype}")
    print(f"[smoke-test] Pixel range       : [{images.min():.3f}, {images.max():.3f}]")
