"""Statistical analysis for galaxy datasets."""

import numpy as np

from loader import GalaxyDataset


def class_distribution(
    labels: np.ndarray, class_names: dict[int, str]
) -> list[dict]:
    """Return per-class count and percentage.

    Returns:
        List of dicts with keys: class_id, name, count, pct.
    """
    total = len(labels)
    rows = []
    for class_id, name in sorted(class_names.items()):
        count = int(np.sum(labels == class_id))
        rows.append({
            "class_id": class_id,
            "name": name,
            "count": count,
            "pct": count / total * 100 if total > 0 else 0.0,
        })
    return rows


def numeric_summary(values: np.ndarray, name: str) -> dict:
    """Compute descriptive statistics for a 1-D numeric array.

    NaN values are excluded from all aggregations and reported separately.
    """
    nan_count = int(np.isnan(values).sum())
    clean = values[~np.isnan(values)]
    return {
        "field": name,
        "count": len(values),
        "nan_count": nan_count,
        "mean": float(np.mean(clean)) if len(clean) else float("nan"),
        "median": float(np.median(clean)) if len(clean) else float("nan"),
        "std": float(np.std(clean)) if len(clean) else float("nan"),
        "min": float(np.min(clean)) if len(clean) else float("nan"),
        "max": float(np.max(clean)) if len(clean) else float("nan"),
    }


def print_report(dataset: GalaxyDataset) -> None:
    """Print a full EDA report for a GalaxyDataset to stdout."""
    _section(f"Dataset: {dataset.name}")

    # warnings
    if dataset.warnings:
        print("  ⚠ Warnings:")
        for w in dataset.warnings:
            print(f"    • {w}")
        print()

    # general info
    print(f"  Samples   : {len(dataset.labels):,}")
    print(f"  Images    : shape={dataset.images_shape}, dtype={dataset.images_dtype}")
    print(f"  Metadata  : {list(dataset.metadata.keys()) or '—'}")
    print()

    # class distribution
    _subsection("Class Distribution")
    dist = class_distribution(dataset.labels, dataset.class_names)
    _print_class_table(dist)

    # metadata stats
    if dataset.metadata:
        _subsection("Metadata Statistics")
        for field_name, values in dataset.metadata.items():
            summary = numeric_summary(values.astype(float), field_name)
            _print_summary(summary)


# ── private helpers ──────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _subsection(title: str) -> None:
    print(f"  ── {title} " + "─" * (54 - len(title)))


def _print_class_table(rows: list[dict]) -> None:
    print(f"  {'ID':>3}  {'Name':<28}  {'Count':>7}  {'%':>6}")
    print("  " + "-" * 52)
    for r in rows:
        print(f"  {r['class_id']:>3}  {r['name']:<28}  {r['count']:>7,}  {r['pct']:>5.1f}%")
    total = sum(r["count"] for r in rows)
    print("  " + "-" * 52)
    print(f"  {'Total':<33}  {total:>7,}  100.0%")
    print()


def _print_summary(s: dict) -> None:
    nan_note = f"  ({s['nan_count']} NaN)" if s["nan_count"] > 0 else ""
    print(f"  {s['field']}")
    print(
        f"    mean={s['mean']:.4f}  median={s['median']:.4f}  "
        f"std={s['std']:.4f}  [{s['min']:.4f}, {s['max']:.4f}]{nan_note}"
    )
    print()
