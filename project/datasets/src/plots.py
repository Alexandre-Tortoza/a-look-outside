"""Visualization functions for galaxy dataset EDA."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from loader import GalaxyDataset
from stats import class_distribution, numeric_summary

sns.set_theme(style="whitegrid", palette="muted")


def plot_class_distribution(
    datasets: list[GalaxyDataset],
    save_path: Path | None = None,
) -> None:
    """Bar chart comparing class distributions across one or more datasets.

    Each dataset gets its own subplot, sharing the x-axis label space.
    """
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        dist = class_distribution(dataset.labels, dataset.class_names)
        labels_text = [f"{r['class_id']}\n{r['name']}" for r in dist]
        counts = [r["count"] for r in dist]

        bars = sns.barplot(x=labels_text, y=counts, ax=ax, color=sns.color_palette("muted")[0])

        # annotate bars with count
        for bar, count in zip(bars.patches, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ax.set_title(f"{dataset.name}\n(n={len(dataset.labels):,})", fontsize=11)
        ax.set_xlabel("Class", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)

    fig.suptitle("Class Distribution", fontsize=13, y=1.02)
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_metadata_distributions(
    dataset: GalaxyDataset,
    save_path: Path | None = None,
) -> None:
    """Histograms for all numeric metadata fields in a dataset.

    Skips datasets with no metadata silently.
    """
    if not dataset.metadata:
        return

    fields = list(dataset.metadata.items())
    n = len(fields)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten()

    for ax, (field_name, values) in zip(axes_flat, fields):
        clean = values[~np.isnan(values.astype(float))]
        summary = numeric_summary(clean, field_name)

        sns.histplot(clean, ax=ax, kde=True, bins=50, color=sns.color_palette("muted")[1])
        ax.axvline(summary["mean"], color="red", linestyle="--", linewidth=1, label=f"mean={summary['mean']:.3f}")
        ax.axvline(summary["median"], color="orange", linestyle="-.", linewidth=1, label=f"median={summary['median']:.3f}")
        ax.set_title(f"{field_name}", fontsize=10)
        ax.set_xlabel(field_name, fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.legend(fontsize=7)

    # hide unused subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"{dataset.name} — Metadata Distributions", fontsize=12, y=1.02)
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_sample_images(
    samples: dict[int, np.ndarray],
    class_names: dict[int, str],
    dataset_name: str,
    save_path: Path | None = None,
) -> None:
    """Grid of sample images: one row per class, one column per sample."""
    if not samples:
        return

    n_classes = len(samples)
    n_per_class = max(len(imgs) for imgs in samples.values())

    fig, axes = plt.subplots(
        n_classes, n_per_class,
        figsize=(n_per_class * 2.2, n_classes * 2.2),
    )
    axes = np.array(axes).reshape(n_classes, n_per_class)

    for row, (class_id, imgs) in enumerate(sorted(samples.items())):
        for col in range(n_per_class):
            ax = axes[row, col]
            if col < len(imgs):
                ax.imshow(_enhance(imgs[col]))
                ax.axis("off")
            else:
                ax.set_visible(False)
        # class label on the left of the row
        axes[row, 0].set_ylabel(
            f"{class_id}: {class_names.get(class_id, f'Class {class_id}')}",
            fontsize=8,
            rotation=0,
            labelpad=80,
            va="center",
        )

    fig.suptitle(f"{dataset_name} — Sample Images", fontsize=12, y=1.01)
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_dataset_comparison(
    decals_samples: dict[int, np.ndarray],
    sdss_samples: dict[int, np.ndarray],
    decals_class_names: dict[int, str],
    sdss_class_names: dict[int, str],
    n_per_class: int = 2,
    save_path: Path | None = None,
) -> None:
    """Side-by-side comparison: DECaLS (left) vs SDSS (right), one row per class."""
    all_classes = sorted(set(decals_samples) | set(sdss_samples))
    n_classes = len(all_classes)
    n_cols = n_per_class * 2 + 1  # DECaLS cols | gap | SDSS cols

    fig, axes = plt.subplots(
        n_classes, n_cols,
        figsize=(n_cols * 1.8, n_classes * 2.0),
    )
    axes = np.array(axes).reshape(n_classes, n_cols)

    for row, class_id in enumerate(all_classes):
        # DECaLS images — left side
        d_imgs = decals_samples.get(class_id, [])
        for col in range(n_per_class):
            ax = axes[row, col]
            if col < len(d_imgs):
                ax.imshow(_enhance(d_imgs[col]))
            ax.axis("off")
            if row == 0 and col == 0:
                ax.set_title("DECaLS\n(256×256)", fontsize=8, color="#2266aa")

        # spacer column
        axes[row, n_per_class].axis("off")

        # SDSS images — right side
        s_imgs = sdss_samples.get(class_id, [])
        for col in range(n_per_class):
            ax = axes[row, n_per_class + 1 + col]
            if col < len(s_imgs):
                ax.imshow(_enhance(s_imgs[col]))
            ax.axis("off")
            if row == 0 and col == 0:
                ax.set_title("SDSS\n(69×69)", fontsize=8, color="#aa4422")

        # class label
        label = decals_class_names.get(class_id) or sdss_class_names.get(class_id, f"Class {class_id}")
        axes[row, 0].set_ylabel(
            f"{class_id}: {label}",
            fontsize=7.5,
            rotation=0,
            labelpad=90,
            va="center",
        )

    fig.suptitle("Dataset Comparison — DECaLS vs SDSS", fontsize=13, y=1.01)
    fig.tight_layout()
    save_or_show(fig, save_path)


def _enhance(img: np.ndarray) -> np.ndarray:
    """Per-image contrast stretch to [0, 255] for better visibility."""
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo) * 255.0
    return img.clip(0, 255).astype(np.uint8)


def save_or_show(fig: plt.Figure, path: Path | None) -> None:
    """Save figure to path or display interactively."""
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)
    else:
        plt.show()
