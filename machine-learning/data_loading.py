from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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
) -> StratifiedSplit:
    labels = labels.astype(np.int64)
    train_images, holdout_images, train_labels, holdout_labels = train_test_split(
        images,
        labels,
        test_size=0.30,
        random_state=random_seed,
        stratify=labels,
    )
    val_images, test_images, val_labels, test_labels = train_test_split(
        holdout_images,
        holdout_labels,
        test_size=0.50,
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
) -> DatasetSplits:
    split = stratified_train_val_test_split(images, labels, random_seed)
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

    num_classes = int(labels.max()) + 1 if len(labels) else 0

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
    )


def flatten_normalized(images: np.ndarray) -> np.ndarray:
    flat = images.reshape(images.shape[0], -1).astype(np.float32)
    if flat.max() > 1.0 + 1e-6:
        flat /= 255.0
    return flat
