from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

CHANNEL_NAMES = ("red", "green", "blue")


def generate_dataset_analysis(
    *,
    images: np.ndarray,
    labels: np.ndarray,
    dataset_label: str,
    source_path: Path | None,
    docs_root: Path,
    class_names: list[str],
    sample_count_per_class: int = 5,
    intensity_histogram_bin_count: int = 64,
    random_seed: int = 42,
) -> Path:
    output_directory = docs_root / dataset_label
    output_directory.mkdir(parents=True, exist_ok=True)

    if images.ndim != 4:
        raise ValueError(
            f"expected images with shape (N, H, W, C), got {images.shape}"
        )

    labels = np.asarray(labels).astype(np.int64)
    num_classes = int(labels.max()) + 1 if len(labels) else 0
    class_label_lookup = _resolve_class_labels(num_classes, class_names)

    summary = _compute_summary(images, labels, source_path, num_classes)
    value_check = _compute_value_check(images)
    class_statistics = _compute_class_statistics(
        images, labels, num_classes, class_label_lookup
    )

    _write_summary_markdown(output_directory, dataset_label, summary)
    _write_value_check_markdown(output_directory, dataset_label, value_check)
    _write_metrics_json(output_directory, summary, value_check, class_statistics)
    _write_class_statistics_csv(output_directory, class_statistics)

    _plot_class_distribution(output_directory, class_statistics, dataset_label)
    _plot_class_balance(output_directory, class_statistics, dataset_label)
    _plot_pixel_intensity_histogram(
        output_directory, images, intensity_histogram_bin_count, dataset_label
    )
    _plot_sample_mosaic(
        output_directory,
        images,
        labels,
        num_classes,
        class_label_lookup,
        sample_count_per_class,
        random_seed,
        dataset_label,
    )

    return output_directory


def _resolve_class_labels(num_classes: int, class_names: list[str]) -> list[str]:
    if class_names and len(class_names) >= num_classes:
        return list(class_names)[:num_classes]
    return [f"class_{i}" for i in range(num_classes)]


def _compute_summary(
    images: np.ndarray,
    labels: np.ndarray,
    source_path: Path | None,
    num_classes: int,
) -> dict[str, Any]:
    counts_per_class = np.bincount(labels, minlength=num_classes).tolist()
    counts_array = np.asarray(counts_per_class)

    image_shape = list(images.shape[1:])
    channel_count = image_shape[2] if len(image_shape) == 3 else 1
    file_size_bytes = source_path.stat().st_size if source_path and source_path.exists() else None

    if np.issubdtype(images.dtype, np.integer):
        normalised = images.astype(np.float32) / float(np.iinfo(images.dtype).max)
    else:
        normalised = images.astype(np.float32)

    per_channel_mean: list[float] = []
    per_channel_std: list[float] = []
    if channel_count == 1 and normalised.ndim == 3:
        per_channel_mean = [float(normalised.mean())]
        per_channel_std = [float(normalised.std())]
    else:
        for channel_index in range(channel_count):
            per_channel_mean.append(float(normalised[..., channel_index].mean()))
            per_channel_std.append(float(normalised[..., channel_index].std()))

    balance_ratio = (
        float(counts_array.max() / counts_array.min())
        if counts_array.min() > 0
        else None
    )

    return {
        "source_path": str(source_path) if source_path else None,
        "sample_count": int(images.shape[0]),
        "num_classes": int(num_classes),
        "image_shape": image_shape,
        "channel_count": int(channel_count),
        "dtype": str(images.dtype),
        "labels_dtype": str(labels.dtype),
        "file_size_bytes": int(file_size_bytes) if file_size_bytes is not None else None,
        "file_size_megabytes": (
            round(file_size_bytes / (1024 ** 2), 2)
            if file_size_bytes is not None
            else None
        ),
        "counts_per_class": [int(value) for value in counts_per_class],
        "minority_class_count": int(counts_array.min()) if num_classes else 0,
        "majority_class_count": int(counts_array.max()) if num_classes else 0,
        "balance_ratio_majority_over_minority": balance_ratio,
        "intensity_min": float(normalised.min()),
        "intensity_max": float(normalised.max()),
        "intensity_mean": float(normalised.mean()),
        "intensity_std": float(normalised.std()),
        "per_channel_mean_normalised": per_channel_mean,
        "per_channel_std_normalised": per_channel_std,
    }


def _compute_value_check(images: np.ndarray) -> dict[str, Any]:
    flat = images.reshape(images.shape[0], -1)
    is_float = np.issubdtype(images.dtype, np.floating)

    nan_count = int(np.isnan(images).sum()) if is_float else 0
    infinity_count = int(np.isinf(images).sum()) if is_float else 0

    if np.issubdtype(images.dtype, np.integer):
        max_value = float(np.iinfo(images.dtype).max)
    else:
        max_value = 1.0 if images.max() <= 1.0 + 1e-6 else float(images.max())

    all_zero_count = int((flat == 0).all(axis=1).sum())
    all_saturated_count = int((flat >= max_value - 1e-6).all(axis=1).sum())
    constant_image_count = int((flat.max(axis=1) == flat.min(axis=1)).sum())

    return {
        "nan_count": nan_count,
        "infinity_count": infinity_count,
        "all_zero_image_count": all_zero_count,
        "all_saturated_image_count": all_saturated_count,
        "constant_value_image_count": constant_image_count,
    }


def _compute_class_statistics(
    images: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    class_label_lookup: list[str],
) -> list[dict[str, Any]]:
    if np.issubdtype(images.dtype, np.integer):
        normalised = images.astype(np.float32) / float(np.iinfo(images.dtype).max)
    else:
        normalised = images.astype(np.float32)
    total = int(images.shape[0])
    channel_count = images.shape[3] if images.ndim == 4 else 1

    rows: list[dict[str, Any]] = []
    for class_index in range(num_classes):
        mask = labels == class_index
        count = int(mask.sum())
        if count == 0:
            continue

        class_images = normalised[mask]
        per_channel_mean = []
        per_channel_std = []
        if channel_count == 1:
            per_channel_mean = [float(class_images.mean())] + [0.0, 0.0]
            per_channel_std = [float(class_images.std())] + [0.0, 0.0]
        else:
            for channel_index in range(min(3, channel_count)):
                per_channel_mean.append(
                    float(class_images[..., channel_index].mean())
                )
                per_channel_std.append(
                    float(class_images[..., channel_index].std())
                )
            while len(per_channel_mean) < 3:
                per_channel_mean.append(0.0)
                per_channel_std.append(0.0)

        rows.append({
            "class_index": class_index,
            "class_name": class_label_lookup[class_index],
            "count": count,
            "percentage": round(100.0 * count / total, 4) if total else 0.0,
            "mean_red": per_channel_mean[0],
            "mean_green": per_channel_mean[1],
            "mean_blue": per_channel_mean[2],
            "std_red": per_channel_std[0],
            "std_green": per_channel_std[1],
            "std_blue": per_channel_std[2],
            "mean_intensity": float(class_images.mean()),
        })
    return rows


def _write_summary_markdown(
    directory: Path, dataset_label: str, summary: dict[str, Any]
) -> None:
    lines = [
        f"# Dataset analysis — {dataset_label}",
        "",
        f"- Source path: `{summary.get('source_path') or 'in-memory'}`",
        f"- Sample count: **{summary['sample_count']}**",
        f"- Number of classes: **{summary['num_classes']}**",
        f"- Image shape (H, W, C): {summary['image_shape']}",
        f"- Channels: {summary['channel_count']}",
        f"- Image dtype: `{summary['dtype']}`",
        f"- Labels dtype: `{summary['labels_dtype']}`",
    ]
    if summary.get("file_size_megabytes") is not None:
        lines.append(f"- File size: {summary['file_size_megabytes']} MB")
    lines.extend([
        "",
        "## Class balance",
        "",
        f"- Minority class count: {summary['minority_class_count']}",
        f"- Majority class count: {summary['majority_class_count']}",
        f"- Imbalance ratio (majority / minority): "
        f"{_format_metric(summary.get('balance_ratio_majority_over_minority'))}",
        "",
        "## Intensity (normalised to [0, 1])",
        "",
        f"- Min: {_format_metric(summary['intensity_min'])}",
        f"- Max: {_format_metric(summary['intensity_max'])}",
        f"- Mean: {_format_metric(summary['intensity_mean'])}",
        f"- Std: {_format_metric(summary['intensity_std'])}",
    ])
    if summary["channel_count"] > 1:
        lines.append("")
        lines.append("## Per-channel statistics (normalised)")
        lines.append("")
        for channel_index, channel_name in enumerate(CHANNEL_NAMES[: summary["channel_count"]]):
            mean_value = summary["per_channel_mean_normalised"][channel_index]
            std_value = summary["per_channel_std_normalised"][channel_index]
            lines.append(
                f"- **{channel_name}**: mean={_format_metric(mean_value)}, "
                f"std={_format_metric(std_value)}"
            )
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `metrics.json`: full machine-readable summary.",
        "- `class_statistics.csv`: per-class counts, percentages, channel statistics.",
        "- `value_check.md`: integrity check report.",
        "- `class_distribution.png`, `class_balance.png`",
        "- `pixel_intensity_histogram.png`",
        "- `sample_mosaic.png`",
    ])
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_value_check_markdown(
    directory: Path, dataset_label: str, value_check: dict[str, Any]
) -> None:
    lines = [
        f"# Integrity check — {dataset_label}",
        "",
        f"- NaN values: **{value_check['nan_count']}**",
        f"- Infinity values: **{value_check['infinity_count']}**",
        f"- Images that are entirely zero: **{value_check['all_zero_image_count']}**",
        f"- Images that are entirely saturated: **{value_check['all_saturated_image_count']}**",
        f"- Images with constant pixel value: **{value_check['constant_value_image_count']}**",
    ]
    if any(value > 0 for value in value_check.values()):
        lines.append("")
        lines.append("> Some samples may be problematic. Investigate before training.")
    (directory / "value_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_metrics_json(
    directory: Path,
    summary: dict[str, Any],
    value_check: dict[str, Any],
    class_statistics: list[dict[str, Any]],
) -> None:
    payload = {
        "summary": summary,
        "value_check": value_check,
        "class_statistics": class_statistics,
    }
    with (directory / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_class_statistics_csv(
    directory: Path, class_statistics: list[dict[str, Any]]
) -> None:
    table = pd.DataFrame(class_statistics)
    table.to_csv(directory / "class_statistics.csv", index=False)


def _plot_class_distribution(
    directory: Path,
    class_statistics: list[dict[str, Any]],
    dataset_label: str,
) -> None:
    if not class_statistics:
        return
    frame = pd.DataFrame(class_statistics)

    figure, axis = plt.subplots(figsize=(max(6, len(frame) * 0.7), 4.5))
    sns.barplot(data=frame, x="class_name", y="count", color="steelblue", ax=axis)
    for index, row in frame.iterrows():
        axis.text(
            index, row["count"], f"{row['percentage']:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )
    axis.set_title(f"Class distribution — {dataset_label}")
    axis.set_xlabel("Class")
    axis.set_ylabel("Count")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(directory / "class_distribution.png", dpi=150)
    plt.close(figure)


def _plot_class_balance(
    directory: Path,
    class_statistics: list[dict[str, Any]],
    dataset_label: str,
) -> None:
    if not class_statistics:
        return
    frame = pd.DataFrame(class_statistics)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.pie(
        frame["count"],
        labels=frame["class_name"],
        autopct="%1.1f%%",
        startangle=90,
    )
    axis.set_title(f"Class balance — {dataset_label}")
    figure.tight_layout()
    figure.savefig(directory / "class_balance.png", dpi=150)
    plt.close(figure)


def _plot_pixel_intensity_histogram(
    directory: Path,
    images: np.ndarray,
    bin_count: int,
    dataset_label: str,
) -> None:
    if np.issubdtype(images.dtype, np.integer):
        normalised = images.astype(np.float32) / float(np.iinfo(images.dtype).max)
    else:
        normalised = images.astype(np.float32)

    figure, axis = plt.subplots(figsize=(7, 4))
    channel_count = images.shape[3] if images.ndim == 4 else 1

    if channel_count == 1:
        axis.hist(normalised.ravel(), bins=bin_count, color="steelblue", alpha=0.8)
    else:
        colors = ("#d62728", "#2ca02c", "#1f77b4")
        for channel_index in range(min(3, channel_count)):
            axis.hist(
                normalised[..., channel_index].ravel(),
                bins=bin_count,
                color=colors[channel_index],
                alpha=0.5,
                label=CHANNEL_NAMES[channel_index],
            )
        axis.legend(loc="upper right")
    axis.set_xlabel("Intensity (normalised)")
    axis.set_ylabel("Count")
    axis.set_title(f"Pixel intensity histogram — {dataset_label}")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(directory / "pixel_intensity_histogram.png", dpi=150)
    plt.close(figure)


def _plot_sample_mosaic(
    directory: Path,
    images: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    class_label_lookup: list[str],
    sample_count_per_class: int,
    random_seed: int,
    dataset_label: str,
) -> None:
    if num_classes == 0 or sample_count_per_class <= 0:
        return
    rng = np.random.default_rng(random_seed)
    figure, axes = plt.subplots(
        num_classes,
        sample_count_per_class,
        figsize=(1.8 * sample_count_per_class, 1.9 * num_classes),
    )
    axes = np.atleast_2d(axes)

    for class_index in range(num_classes):
        class_indices = np.where(labels == class_index)[0]
        sample_count = min(sample_count_per_class, len(class_indices))
        chosen = rng.choice(class_indices, size=sample_count, replace=False) if sample_count else []
        for column in range(sample_count_per_class):
            axis = axes[class_index, column]
            if column >= sample_count:
                axis.set_visible(False)
                continue
            sample = images[int(chosen[column])]
            if np.issubdtype(images.dtype, np.integer):
                axis.imshow(sample)
            else:
                axis.imshow(np.clip(sample, 0.0, 1.0))
            if column == 0:
                axis.set_ylabel(class_label_lookup[class_index], fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle(f"Random samples per class — {dataset_label}")
    figure.tight_layout()
    figure.savefig(directory / "sample_mosaic.png", dpi=150)
    plt.close(figure)


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
