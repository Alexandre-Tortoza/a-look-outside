from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.analysis import generate_dataset_analysis  # noqa: E402
from dataset.balancing.registry import get_method  # noqa: E402
from dataset.input_output import (  # noqa: E402
    class_distribution,
    read_dataset,
    resolve_class_names,
    write_dataset,
)


@dataclass
class ResolvedDataset:
    dataset_name: str
    h5_path: Path
    dataset_label: str
    methods_applied: list[str]
    was_generated: bool


def resolve_benchmark_datasets(
    benchmark_spec: dict[str, Any],
    config: dict[str, Any],
    project_root: Path,
    logger: logging.Logger,
    generate_analysis: bool,
) -> list[ResolvedDataset]:
    paths = config.get("paths") or {}
    raw_directory = project_root / paths.get("raw_dataset_directory", "dataset/raw")
    processed_directory = project_root / paths.get(
        "processed_dataset_directory", "dataset/processed"
    )
    analysis_section = config.get("dataset_analysis") or {}
    docs_root = project_root / analysis_section.get(
        "output_directory", "docs/dataset"
    )
    sample_count_per_class = int(analysis_section.get("sample_count_per_class", 5))
    histogram_bin_count = int(
        analysis_section.get("intensity_histogram_bin_count", 64)
    )
    random_seed = int(config.get("random_seed", 42))

    datasets_spec = benchmark_spec.get("datasets") or {}
    sources = list(datasets_spec.get("sources") or [])
    include_raw = bool(datasets_spec.get("include_raw", True))
    balancing_variants = list(datasets_spec.get("balancing_variants") or [])

    resolved: list[ResolvedDataset] = []
    cache: dict[str, tuple[Any, Any]] = {}

    for source in sources:
        raw_path = raw_directory / f"{source}.h5"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"raw dataset '{source}' not found at {raw_path}"
            )

        if include_raw:
            entry = ResolvedDataset(
                dataset_name=f"{source}_raw",
                h5_path=raw_path,
                dataset_label=f"{source}-raw",
                methods_applied=[],
                was_generated=False,
            )
            logger.info("dataset ready: %s -> %s", entry.dataset_name, entry.h5_path)
            resolved.append(entry)
            if generate_analysis:
                images, labels = read_dataset(raw_path)
                cache[source] = (images, labels)
                _emit_analysis(
                    images=images,
                    labels=labels,
                    dataset_label=entry.dataset_label,
                    source_path=raw_path,
                    docs_root=docs_root,
                    class_names=resolve_class_names(config.get("class_names"), source),
                    sample_count_per_class=sample_count_per_class,
                    histogram_bin_count=histogram_bin_count,
                    random_seed=random_seed,
                    logger=logger,
                )

        for variant in balancing_variants:
            method_names = list(variant.get("methods") or [])
            if not method_names:
                continue
            suffix = "_".join(method_names)
            output_name = f"{source}_{suffix}"
            output_path = processed_directory / f"{output_name}.h5"

            if output_path.exists():
                logger.info(
                    "dataset already present (skipping balancing): %s",
                    output_path,
                )
                was_generated = False
            else:
                images, labels = cache.get(source) or read_dataset(raw_path)
                cache[source] = (images, labels)
                logger.info(
                    "balancing %s with %s -> %s",
                    source, " -> ".join(method_names), output_path,
                )
                for method_name in method_names:
                    method = get_method(method_name)
                    images, labels = method.apply(images, labels, random_seed)
                    logger.info(
                        "  after %s: shape=%s distribution=%s",
                        method_name, images.shape, class_distribution(labels),
                    )
                write_dataset(output_path, images, labels)
                was_generated = True

            entry = ResolvedDataset(
                dataset_name=output_name,
                h5_path=output_path,
                dataset_label=output_name,
                methods_applied=method_names,
                was_generated=was_generated,
            )
            resolved.append(entry)

            if generate_analysis:
                images, labels = read_dataset(output_path)
                _emit_analysis(
                    images=images,
                    labels=labels,
                    dataset_label=entry.dataset_label,
                    source_path=output_path,
                    docs_root=docs_root,
                    class_names=resolve_class_names(config.get("class_names"), source),
                    sample_count_per_class=sample_count_per_class,
                    histogram_bin_count=histogram_bin_count,
                    random_seed=random_seed,
                    logger=logger,
                )

    return resolved


def _emit_analysis(
    *,
    images: Any,
    labels: Any,
    dataset_label: str,
    source_path: Path,
    docs_root: Path,
    class_names: list[str],
    sample_count_per_class: int,
    histogram_bin_count: int,
    random_seed: int,
    logger: logging.Logger,
) -> None:
    output_directory = generate_dataset_analysis(
        images=images,
        labels=labels,
        dataset_label=dataset_label,
        source_path=source_path,
        docs_root=docs_root,
        class_names=class_names,
        sample_count_per_class=sample_count_per_class,
        intensity_histogram_bin_count=histogram_bin_count,
        random_seed=random_seed,
    )
    logger.info("dataset analysis written to %s", output_directory)
