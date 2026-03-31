"""Galaxy10 EDA — entry point.

Usage:
    uv run main.py                # show plots interactively
    uv run main.py --save-plots   # save plots to ../outputs/
"""

import argparse
import sys
from pathlib import Path

from loader import DatasetLoadError, load_decals, load_sdss
from loader import load_sample_images
from plots import (
    plot_class_distribution,
    plot_dataset_comparison,
    plot_metadata_distributions,
    plot_sample_images,
)
from stats import print_report

_SRC_DIR = Path(__file__).parent
_DATASETS_DIR = _SRC_DIR.parent
_OUTPUTS_DIR = _DATASETS_DIR / "outputs"

DECALS_PATH = _DATASETS_DIR / "Galaxy10_DECals_NoDuplicated.h5"
SDSS_PATH = _DATASETS_DIR / "Galaxy10.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Galaxy10 EDA")
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help=f"Save plots to {_OUTPUTS_DIR}/ instead of displaying.",
    )
    return parser.parse_args()


def _load(loader_fn, path: Path):
    """Wraps a loader and handles file-not-found and missing-field errors."""
    try:
        return loader_fn(path)
    except FileNotFoundError:
        print(f"  [ERROR] File not found: {path}", file=sys.stderr)
        return None
    except DatasetLoadError as e:
        print(f"  [ERROR] {e}", file=sys.stderr)
        return None


def main() -> None:
    args = parse_args()
    save = args.save_plots

    print("\nLoading datasets...")
    decals = _load(load_decals, DECALS_PATH)
    sdss = _load(load_sdss, SDSS_PATH)

    loaded = [ds for ds in [decals, sdss] if ds is not None]
    if not loaded:
        print("[ERROR] No datasets could be loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    print()
    for ds in loaded:
        print_report(ds)

    print("\nGenerating plots...")

    plot_class_distribution(
        loaded,
        save_path=_OUTPUTS_DIR / "class_distribution.png" if save else None,
    )

    for ds in loaded:
        if ds.metadata:
            plot_metadata_distributions(
                ds,
                save_path=_OUTPUTS_DIR / f"{ds.name.replace(' ', '_')}_metadata.png" if save else None,
            )

    # image samples
    print("\nLoading image samples (this may take a moment)...")
    N_PER_CLASS = 3

    for ds, path in [(decals, DECALS_PATH), (sdss, SDSS_PATH)]:
        if ds is None:
            continue
        samples = load_sample_images(path, ds.labels, ds.class_names, n_per_class=N_PER_CLASS)
        plot_sample_images(
            samples,
            ds.class_names,
            ds.name,
            save_path=_OUTPUTS_DIR / f"{ds.name.replace(' ', '_')}_samples.png" if save else None,
        )

    # comparison plot (only if both loaded)
    if decals is not None and sdss is not None:
        print("Building comparison plot...")
        decals_samples = load_sample_images(DECALS_PATH, decals.labels, decals.class_names, n_per_class=2)
        sdss_samples = load_sample_images(SDSS_PATH, sdss.labels, sdss.class_names, n_per_class=2)
        plot_dataset_comparison(
            decals_samples,
            sdss_samples,
            decals.class_names,
            sdss.class_names,
            n_per_class=2,
            save_path=_OUTPUTS_DIR / "comparison_decals_vs_sdss.png" if save else None,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
