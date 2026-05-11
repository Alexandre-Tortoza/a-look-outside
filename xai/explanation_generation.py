from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MACHINE_LEARNING_ROOT = PROJECT_ROOT / "machine-learning"
for path in (str(PROJECT_ROOT), str(MACHINE_LEARNING_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset.input_output import resolve_class_names  # noqa: E402
from models.registry import get_model_info  # noqa: E402

from xai.artifact_storage import RunArtifact, output_directories  # noqa: E402
from xai.methods.registry import XaiMethod  # noqa: E402
from xai.sample_extraction import extract_test_samples  # noqa: E402


@dataclass
class ExplanationResult:
    run_artifact: RunArtifact
    samples_directory: Path
    explanations_directory: Path
    sample_count: int
    method_artifact_counts: dict[str, int]


def generate_for_run(
    run_artifact: RunArtifact,
    methods: list[XaiMethod],
    config: dict[str, Any],
    computer_configuration: dict[str, Any],
    logger: logging.Logger,
) -> ExplanationResult:
    paths = config.get("paths") or {}
    xai_root = PROJECT_ROOT / paths.get("xai_output_directory", "docs/xai")

    info = get_model_info(run_artifact.model_name)
    applicable_methods = [method for method in methods if method.applies_to(info)]

    samples_directory, explanations_directory = output_directories(
        xai_root=xai_root,
        model_name=run_artifact.model_name,
        dataset_name=run_artifact.dataset_name,
    )

    class_names = resolve_class_names(config.get("class_names"), run_artifact.dataset_name)
    logger.info(
        "extracting test samples for %s | %s",
        run_artifact.model_name, run_artifact.dataset_name,
    )
    samples = extract_test_samples(
        run_config=run_artifact.run_config,
        samples_root=samples_directory,
        class_names=class_names,
        project_root=PROJECT_ROOT,
    )
    logger.info("extracted %d test samples", len(samples.records))

    method_counts: dict[str, int] = {}
    for method in applicable_methods:
        logger.info("running XAI method '%s'", method.name)
        count = method.apply(
            samples=samples,
            explanations_directory=explanations_directory,
            run_config=run_artifact.run_config,
            factory_kwargs=run_artifact.factory_kwargs,
            checkpoint_path=run_artifact.checkpoint_path,
            computer_configuration=computer_configuration,
            logger=logger,
        )
        method_counts[method.name] = int(count or 0)

    return ExplanationResult(
        run_artifact=run_artifact,
        samples_directory=samples_directory,
        explanations_directory=explanations_directory,
        sample_count=len(samples.records),
        method_artifact_counts=method_counts,
    )
