from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MACHINE_LEARNING_ROOT = PROJECT_ROOT / "machine-learning"
for path in (str(PROJECT_ROOT), str(MACHINE_LEARNING_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from pipeline import ModelSpec, RunResult, run_pipeline  # noqa: E402

from benchmark.dataset_resolution import (  # noqa: E402
    ResolvedDataset,
    resolve_benchmark_datasets,
)
from benchmark.recommendations import Insight, generate_recommendations  # noqa: E402
from xai.artifact_storage import RunArtifact, discover_runs  # noqa: E402
from xai.explanation_generation import generate_for_run  # noqa: E402
from xai.methods.registry import list_methods as list_xai_methods  # noqa: E402


@dataclass
class BenchmarkResult:
    benchmark_name: str
    resolved_datasets: list[ResolvedDataset]
    run_results: list[RunResult]
    insights: list[Insight] = field(default_factory=list)
    stage_durations_seconds: dict[str, float] = field(default_factory=dict)


def run_benchmark(
    benchmark_name: str,
    config: dict[str, Any],
    computer_configuration: dict[str, Any],
    project_root: Path,
    console: Console,
) -> BenchmarkResult:
    benchmarks = config.get("benchmarks") or {}
    if benchmark_name not in benchmarks:
        raise KeyError(
            f"benchmark '{benchmark_name}' not found in config.yaml. "
            f"available: {', '.join(sorted(benchmarks)) or '(none)'}"
        )
    benchmark_spec = benchmarks[benchmark_name]
    logger = _build_logger(console)

    paths = config.get("paths") or {}
    docs_root = project_root / paths.get("documentation_directory", "docs")
    runs_root = project_root / paths.get(
        "machine_learning_run_directory", "machine-learning/runs"
    )
    leaderboard_section = config.get("leaderboard") or {}
    jsonl_filename = leaderboard_section.get("jsonl_filename", "runs.jsonl")
    jsonl_path = docs_root / jsonl_filename

    logger.info("starting benchmark '%s'", benchmark_name)

    stage_durations: dict[str, float] = {}

    # Stage 1 — Datasets
    console.rule("[bold]Stage 1 — Datasets")
    stage_started = time.monotonic()
    resolved_datasets = resolve_benchmark_datasets(
        benchmark_spec=benchmark_spec,
        config=config,
        project_root=project_root,
        logger=logger,
        generate_analysis=bool(
            benchmark_spec.get("generate_dataset_analysis", True)
        ),
    )
    stage_durations["datasets"] = time.monotonic() - stage_started
    logger.info(
        "stage 1 complete: %d datasets ready in %.1fs",
        len(resolved_datasets), stage_durations["datasets"],
    )

    # Stage 2 — Machine learning
    console.rule("[bold]Stage 2 — Machine learning")
    stage_started = time.monotonic()
    model_specs = _build_model_specs(benchmark_spec, config)
    dataset_names = [resolved.dataset_name for resolved in resolved_datasets]
    if not model_specs or not dataset_names:
        raise ValueError(
            "benchmark must define at least one model and one dataset"
        )
    logger.info(
        "training plan: %d models x %d datasets = %d runs",
        len(model_specs), len(dataset_names),
        len(model_specs) * len(dataset_names),
    )
    pre_run_artifact_set = _existing_run_directory_set(runs_root)
    run_results = run_pipeline(
        model_specs=model_specs,
        dataset_names=dataset_names,
        configuration=config,
        computer_configuration=computer_configuration,
    )
    stage_durations["machine_learning"] = time.monotonic() - stage_started
    logger.info(
        "stage 2 complete: %d runs in %.1fs",
        len(run_results), stage_durations["machine_learning"],
    )

    # Stage 3 — XAI (optional)
    if bool(benchmark_spec.get("generate_xai", False)):
        console.rule("[bold]Stage 3 — XAI")
        stage_started = time.monotonic()
        new_artifacts = _new_xai_artifacts(runs_root, pre_run_artifact_set)
        xai_methods = list_xai_methods()
        for artifact in new_artifacts:
            try:
                generate_for_run(
                    run_artifact=artifact,
                    methods=xai_methods,
                    config=config,
                    computer_configuration=computer_configuration,
                    logger=logger,
                )
            except Exception as error:  # noqa: BLE001
                logger.warning(
                    "xai failed for %s: %s", artifact.run_directory.name, error
                )
        stage_durations["xai"] = time.monotonic() - stage_started
        logger.info(
            "stage 3 complete: %d artifacts in %.1fs",
            len(new_artifacts), stage_durations["xai"],
        )
    else:
        logger.info("stage 3 (xai) skipped by benchmark spec")

    # Stage 4 — Recommendations (optional)
    insights: list[Insight] = []
    if bool(benchmark_spec.get("generate_recommendations", True)):
        console.rule("[bold]Stage 4 — Recommendations")
        stage_started = time.monotonic()
        insights = generate_recommendations(
            jsonl_path=jsonl_path,
            runs_root=runs_root,
            docs_root=docs_root,
        )
        stage_durations["recommendations"] = time.monotonic() - stage_started
        logger.info(
            "stage 4 complete: %d insights in %.1fs",
            len(insights), stage_durations["recommendations"],
        )
    else:
        logger.info("stage 4 (recommendations) skipped by benchmark spec")

    return BenchmarkResult(
        benchmark_name=benchmark_name,
        resolved_datasets=resolved_datasets,
        run_results=run_results,
        insights=insights,
        stage_durations_seconds=stage_durations,
    )


def _build_model_specs(
    benchmark_spec: dict[str, Any],
    config: dict[str, Any],
) -> list[ModelSpec]:
    knn_modes = list(benchmark_spec.get("knn_modes") or ["pixels"])
    knn_defaults = (config.get("models") or {}).get("k_nearest_neighbors") or {}
    n_neighbors = int(knn_defaults.get("n_neighbors", 5))

    specs: list[ModelSpec] = []
    for model_name in benchmark_spec.get("models") or []:
        if model_name == "k_nearest_neighbors":
            for mode in knn_modes:
                specs.append(ModelSpec(
                    name=model_name,
                    factory_kwargs={"mode": mode, "n_neighbors": n_neighbors},
                ))
        else:
            specs.append(ModelSpec(name=model_name, factory_kwargs={}))
    return specs


def _existing_run_directory_set(runs_root: Path) -> set[str]:
    if not runs_root.exists():
        return set()
    return {entry.name for entry in runs_root.iterdir() if entry.is_dir()}


def _new_xai_artifacts(
    runs_root: Path,
    pre_run_directory_names: set[str],
) -> list[RunArtifact]:
    artifacts = discover_runs(runs_root)
    return [
        artifact for artifact in artifacts
        if artifact.run_directory.name not in pre_run_directory_names
    ]


def _build_logger(console: Console) -> logging.Logger:
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.addHandler(RichHandler(console=console, show_path=False, markup=True))
    return logger
