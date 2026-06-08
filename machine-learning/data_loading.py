from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dataset_kind import (
    DATASET_KIND_NATURAL,
    DATASET_KIND_PROCESSED,
    DATASET_KIND_UNKNOWN,
    dataset_name_is_processed,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset.balancing.registry import get_method

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DatasetSplits:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_images: np.ndarray
    train_labels: np.ndarray
    val_images: np.ndarray
    val_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    image_size: int
    num_classes: int
    balance_metadata: dict[str, Any] = field(default_factory=dict)


class GalaxyImageDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        image_size: int,
    ) -> None:
        if images.ndim != 4:
            raise ValueError(
                f"expected images with shape (N, H, W, C), got {images.shape}"
            )
        self._images = images
        self._labels = labels.astype(np.int64)
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self._images[index]
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        tensor = self._transform(image)
        return tensor, int(self._labels[index])


def resolve_dataset_path(
    dataset_name: str,
    raw_directory: Path,
    processed_directory: Path,
) -> Path:
    if dataset_name.endswith("_raw"):
        stem = dataset_name[: -len("_raw")]
        candidate = raw_directory / f"{stem}.h5"
        if candidate.exists():
            return candidate
    candidate = processed_directory / f"{dataset_name}.h5"
    if candidate.exists():
        return candidate
    raw_candidate = raw_directory / f"{dataset_name}.h5"
    if raw_candidate.exists():
        return raw_candidate
    raise FileNotFoundError(
        f"dataset '{dataset_name}' not found in {raw_directory} or {processed_directory}"
    )


@dataclass
class StratifiedSplit:
    train_images: np.ndarray
    train_labels: np.ndarray
    val_images: np.ndarray
    val_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray


def stratified_train_val_test_split(
    images: np.ndarray,
    labels: np.ndarray,
    random_seed: int,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> StratifiedSplit:
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            "split ratios must sum to 1.0, got "
            f"{train_ratio:.4f}+{validation_ratio:.4f}+{test_ratio:.4f}={ratio_sum:.4f}"
        )
    if min(train_ratio, validation_ratio, test_ratio) <= 0:
        raise ValueError("split ratios must all be positive")

    labels = labels.astype(np.int64)
    holdout_ratio = validation_ratio + test_ratio
    train_images, holdout_images, train_labels, holdout_labels = train_test_split(
        images,
        labels,
        test_size=holdout_ratio,
        random_state=random_seed,
        stratify=labels,
    )
    test_fraction_of_holdout = test_ratio / holdout_ratio
    val_images, test_images, val_labels, test_labels = train_test_split(
        holdout_images,
        holdout_labels,
        test_size=test_fraction_of_holdout,
        random_state=random_seed,
        stratify=holdout_labels,
    )
    return StratifiedSplit(
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        test_images=test_images,
        test_labels=test_labels,
    )


def build_data_loaders(
    images: np.ndarray,
    labels: np.ndarray,
    image_size: int,
    batch_size: int,
    num_workers: int,
    random_seed: int,
    pin_memory: bool,
    split_ratios: dict[str, float] | None = None,
    training_balance_config: dict[str, Any] | None = None,
    dataset_name: str | None = None,
    model_name: str | None = None,
) -> DatasetSplits:
    ratios = split_ratios or {}
    split = stratified_train_val_test_split(
        images,
        labels,
        random_seed,
        train_ratio=float(ratios.get("train", 0.70)),
        validation_ratio=float(ratios.get("validation", 0.15)),
        test_ratio=float(ratios.get("test", 0.15)),
    )
    split, balance_metadata = _apply_train_only_balance(
        split=split,
        training_balance_config=training_balance_config,
        dataset_name=dataset_name,
        model_name=model_name,
        random_seed=random_seed,
    )
    return build_data_loaders_from_split(
        split=split,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        balance_metadata=balance_metadata,
    )


def build_data_loaders_from_split(
    split: StratifiedSplit,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    balance_metadata: dict[str, Any] | None = None,
) -> DatasetSplits:
    train_images = split.train_images
    train_labels = split.train_labels
    val_images = split.val_images
    val_labels = split.val_labels
    test_images = split.test_images
    test_labels = split.test_labels

    train_dataset = GalaxyImageDataset(train_images, train_labels, image_size)
    val_dataset = GalaxyImageDataset(val_images, val_labels, image_size)
    test_dataset = GalaxyImageDataset(test_images, test_labels, image_size)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    all_labels = np.concatenate([train_labels, val_labels, test_labels])
    num_classes = int(all_labels.max()) + 1 if len(all_labels) else 0

    return DatasetSplits(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        test_images=test_images,
        test_labels=test_labels,
        image_size=image_size,
        num_classes=num_classes,
        balance_metadata=balance_metadata or {
            "enabled": False,
            "applied": False,
            "methods": [],
        },
    )


def _apply_train_only_balance(
    *,
    split: StratifiedSplit,
    training_balance_config: dict[str, Any] | None,
    dataset_name: str | None,
    model_name: str | None,
    random_seed: int,
) -> tuple[StratifiedSplit, dict[str, Any]]:
    metadata = {
        "enabled": False,
        "applied": False,
        "methods": [],
        "apply_to": "none",
        "dataset_kind_inferred": _infer_dataset_kind(dataset_name),
        "train_distribution_before": _class_distribution(split.train_labels),
        "train_distribution_after": _class_distribution(split.train_labels),
        "validation_distribution": _class_distribution(split.val_labels),
        "test_distribution": _class_distribution(split.test_labels),
    }
    config = training_balance_config or {}
    if not bool(config.get("enabled", False)):
        return split, metadata

    apply_to = str(config.get("apply_to", "train_only"))
    if apply_to != "train_only":
        raise ValueError(
            "training_balance.apply_to only supports 'train_only'; "
            f"got {apply_to!r}"
        )

    if not _matches_filter(config.get("datasets"), dataset_name):
        metadata["enabled"] = True
        metadata["apply_to"] = apply_to
        return split, metadata
    if not _matches_filter(config.get("models"), model_name):
        metadata["enabled"] = True
        metadata["apply_to"] = apply_to
        return split, metadata

    methods = [str(method_name) for method_name in config.get("methods") or []]
    if not methods:
        metadata["enabled"] = True
        metadata["apply_to"] = apply_to
        return split, metadata

    train_images = split.train_images
    train_labels = split.train_labels
    for method_name in methods:
        method = get_method(method_name)
        train_images, train_labels = method.apply(train_images, train_labels, random_seed)

    balanced_split = StratifiedSplit(
        train_images=train_images,
        train_labels=train_labels,
        val_images=split.val_images,
        val_labels=split.val_labels,
        test_images=split.test_images,
        test_labels=split.test_labels,
    )
    metadata.update({
        "enabled": True,
        "applied": True,
        "methods": methods,
        "apply_to": apply_to,
        "train_distribution_after": _class_distribution(train_labels),
    })
    return balanced_split, metadata


def _matches_filter(raw_filter: Any, value: str | None) -> bool:
    if raw_filter in (None, "", "all"):
        return True
    if isinstance(raw_filter, str):
        return raw_filter == value
    values = {str(item) for item in raw_filter}
    return "all" in values or (value is not None and value in values)


def _class_distribution(labels: np.ndarray) -> dict[int, int]:
    unique_labels, counts = np.unique(labels.astype(np.int64), return_counts=True)
    return {
        int(label): int(count)
        for label, count in zip(unique_labels, counts, strict=True)
    }


def _infer_dataset_kind(dataset_name: str | None) -> str:
    if not dataset_name:
        return DATASET_KIND_UNKNOWN
    if dataset_name.endswith("_raw"):
        return DATASET_KIND_NATURAL
    if dataset_name_is_processed(dataset_name):
        return DATASET_KIND_PROCESSED
    return DATASET_KIND_UNKNOWN


def flatten_normalized(images: np.ndarray) -> np.ndarray:
    flat = images.reshape(images.shape[0], -1).astype(np.float32)
    if flat.max() > 1.0 + 1e-6:
        flat /= 255.0
    return flat
