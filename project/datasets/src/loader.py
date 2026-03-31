"""Galaxy dataset loader — generic H5 reader with fault tolerance."""

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np


class DatasetLoadError(Exception):
    """Raised when a required field is missing from the H5 file."""


@dataclass
class GalaxyDataset:
    name: str
    labels: np.ndarray
    images_shape: tuple
    images_dtype: np.dtype
    class_names: dict[int, str]
    metadata: dict[str, np.ndarray] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ── class name registries ────────────────────────────────────────────────────

CLASS_NAMES_DECALS: dict[int, str] = {
    0: "Disturbed",
    1: "Merging",
    2: "Round Smooth",
    3: "In-between Smooth",
    4: "Cigar Shaped",
    5: "Barred Spiral",
    6: "Unbarred Tight Spiral",
    7: "Unbarred Loose Spiral",
    8: "Edge-on w/o Bulge",
    9: "Edge-on w/ Bulge",
}

CLASS_NAMES_SDSS: dict[int, str] = {
    0: "Disturbed",
    1: "Merging",
    2: "Round Smooth",
    3: "In-between Smooth",
    4: "Cigar Shaped",
    5: "Edge-on Boxy Bulge",  # only 17 samples; abandoned in DECaLS
    6: "Barred Spiral",
    7: "Unbarred Tight Spiral",
    8: "Unbarred Loose Spiral",
    9: "Edge-on w/ Bulge",
}

_SKIP_KEYS = {"images"}  # never auto-loaded as metadata


# ── generic loader ───────────────────────────────────────────────────────────

def load_h5(
    path: str | Path,
    name: str,
    label_key: str = "ans",
    metadata_keys: list[str] | None = None,
    class_names: dict[int, str] | None = None,
) -> GalaxyDataset:
    """Load a Galaxy H5 dataset with fault-tolerant metadata reading.

    Args:
        path: Path to the .h5 file.
        name: Human-readable dataset name used in reports and plots.
        label_key: H5 key for the class labels array (required field).
        metadata_keys: List of H5 keys to load as metadata.
            Pass ``None`` to auto-detect all 1-D numeric fields
            (excluding images and the label key).
        class_names: Mapping from class index to display name.
            Missing entries fall back to ``"Class {n}"``.

    Returns:
        GalaxyDataset with loaded data and any accumulated warnings.

    Raises:
        DatasetLoadError: If ``label_key`` is not found in the file.
        OSError: If the file does not exist or cannot be opened.
    """
    path = Path(path)
    dataset_warnings: list[str] = []

    with h5py.File(path, "r") as f:
        available_keys = set(f.keys())

        # ── labels (required) ────────────────────────────────────────────
        if label_key not in available_keys:
            raise DatasetLoadError(
                f"[{name}] Required label key '{label_key}' not found in '{path}'.\n"
                f"  Available keys: {sorted(available_keys)}"
            )
        labels = f[label_key][:]

        # ── images shape/dtype only (no full load) ───────────────────────
        if "images" in available_keys:
            images_shape = f["images"].shape
            images_dtype = f["images"].dtype
        else:
            dataset_warnings.append(
                f"[{name}] 'images' key not found — shape/dtype unavailable."
            )
            images_shape = (len(labels),)
            images_dtype = np.dtype("uint8")

        # ── metadata ─────────────────────────────────────────────────────
        if metadata_keys is None:
            candidate_keys = available_keys - _SKIP_KEYS - {label_key}
        else:
            candidate_keys = set(metadata_keys)

        metadata: dict[str, np.ndarray] = {}
        for key in sorted(candidate_keys):
            if key not in available_keys:
                dataset_warnings.append(
                    f"[{name}] Metadata key '{key}' not found — skipped."
                )
                continue
            arr = f[key][:]
            if arr.ndim != 1:
                dataset_warnings.append(
                    f"[{name}] Metadata key '{key}' has shape {arr.shape} "
                    f"(expected 1-D) — skipped."
                )
                continue
            if not np.issubdtype(arr.dtype, np.number):
                dataset_warnings.append(
                    f"[{name}] Metadata key '{key}' has non-numeric dtype "
                    f"'{arr.dtype}' — skipped."
                )
                continue
            metadata[key] = arr

    resolved_class_names = _resolve_class_names(labels, class_names or {})

    return GalaxyDataset(
        name=name,
        labels=labels,
        images_shape=images_shape,
        images_dtype=images_dtype,
        class_names=resolved_class_names,
        metadata=metadata,
        warnings=dataset_warnings,
    )


# ── dataset-specific wrappers ────────────────────────────────────────────────

def load_decals(path: str | Path) -> GalaxyDataset:
    return load_h5(
        path=path,
        name="Galaxy10 DECaLS",
        label_key="ans",
        metadata_keys=["redshift", "ra", "dec", "pxscale"],
        class_names=CLASS_NAMES_DECALS,
    )


def load_sdss(path: str | Path) -> GalaxyDataset:
    return load_h5(
        path=path,
        name="Galaxy10 SDSS",
        label_key="ans",
        metadata_keys=None,  # SDSS has no extra metadata
        class_names=CLASS_NAMES_SDSS,
    )


# ── helpers ──────────────────────────────────────────────────────────────────

def load_sample_images(
    path: str | Path,
    labels: np.ndarray,
    class_names: dict[int, str],
    n_per_class: int = 3,
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """Load a small sample of images per class without loading the full dataset.

    Args:
        path: Path to the .h5 file.
        labels: Label array already loaded via :func:`load_h5`.
        class_names: Class id → name mapping (only these classes are sampled).
        n_per_class: Number of images to load per class.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping class_id → uint8 array of shape (n, H, W, 3).
        Empty dict if the file has no ``images`` key (with a printed warning).
    """
    rng = np.random.default_rng(seed)
    path = Path(path)

    indices_by_class: dict[int, np.ndarray] = {}
    for class_id in class_names:
        pool = np.where(labels == class_id)[0]
        n = min(n_per_class, len(pool))
        chosen = rng.choice(pool, size=n, replace=False)
        indices_by_class[class_id] = np.sort(chosen)

    samples: dict[int, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        if "images" not in f:
            print(f"  [WARNING] No 'images' key in '{path}' — cannot load samples.")
            return samples
        for class_id, idxs in indices_by_class.items():
            samples[class_id] = f["images"][list(idxs)]  # (n, H, W, 3)

    return samples


def _resolve_class_names(
    labels: np.ndarray, class_names: dict[int, str]
) -> dict[int, str]:
    unique_classes = sorted(int(c) for c in np.unique(labels))
    return {c: class_names.get(c, f"Class {c}") for c in unique_classes}
