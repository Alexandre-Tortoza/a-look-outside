from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from text_formatting import format_metric  # type: ignore[import-not-found]  # noqa: E402

logger = logging.getLogger("leaderboard")

DEFAULT_HEATMAP_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "cohen_kappa",
    "matthews_correlation_coefficient",
    "roc_auc_macro",
    "log_loss",
)

LOWER_IS_BETTER = {"log_loss"}

LEADERBOARD_DISPLAY_COLUMNS = (
    "rank",
    "model_name",
    "dataset_name",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "cohen_kappa",
    "matthews_correlation_coefficient",
    "roc_auc_macro",
    "log_loss",
    "misclassified_count",
    "duration_seconds",
    "documentation_directory",
)


def regenerate_all(
    *,
    jsonl_path: Path,
    docs_root: Path,
    primary_metric: str = "balanced_accuracy",
    secondary_metric: str = "accuracy",
    heatmap_metrics: list[str] | None = None,
) -> dict[str, Any]:
    records = load_records(jsonl_path)
    if not records:
        logger.warning("no records found at %s", jsonl_path)
        return {"records": 0, "leaderboard_path": None}

    docs_root.mkdir(parents=True, exist_ok=True)

    best_records = select_best_per_pair(records, primary_metric)
    leaderboard_frame = build_leaderboard(best_records, primary_metric, secondary_metric)

    leaderboard_markdown_path = docs_root / "leaderboard.md"
    leaderboard_csv_path = docs_root / "leaderboard.csv"
    write_leaderboard_markdown(
        frame=leaderboard_frame,
        output_path=leaderboard_markdown_path,
        all_records=records,
        primary_metric=primary_metric,
        secondary_metric=secondary_metric,
    )
    write_leaderboard_csv(leaderboard_frame, leaderboard_csv_path)

    metrics_to_plot = list(heatmap_metrics or DEFAULT_HEATMAP_METRICS)
    for metric_name in metrics_to_plot:
        plot_metric_heatmap(
            best_records=best_records,
            metric=metric_name,
            output_path=docs_root / f"leaderboard_{metric_name}.png",
        )

    write_per_model_comparisons(records, docs_root / "by_model", primary_metric)
    write_per_dataset_comparisons(records, docs_root / "by_dataset", primary_metric)

    return {
        "records": len(records),
        "best_records": len(best_records),
        "leaderboard_path": leaderboard_markdown_path,
    }


def load_records(jsonl_path: Path) -> list[dict[str, Any]]:
    if not jsonl_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as error:
                logger.warning(
                    "skipping malformed line %d in %s: %s",
                    line_number, jsonl_path, error,
                )
    return records


def select_best_per_pair(
    records: list[dict[str, Any]],
    primary_metric: str,
) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (record.get("model_name"), record.get("dataset_name"))
        if not key[0] or not key[1]:
            continue
        if key not in best:
            best[key] = record
            continue
        candidate_value = record.get(primary_metric)
        incumbent_value = best[key].get(primary_metric)
        if _is_better(candidate_value, incumbent_value, primary_metric):
            best[key] = record
    return list(best.values())


def build_leaderboard(
    best_records: list[dict[str, Any]],
    primary_metric: str,
    secondary_metric: str,
) -> pd.DataFrame:
    if not best_records:
        return pd.DataFrame()

    frame = pd.DataFrame(best_records)
    ascending = primary_metric in LOWER_IS_BETTER
    frame = frame.sort_values(
        by=[primary_metric, secondary_metric],
        ascending=[ascending, secondary_metric in LOWER_IS_BETTER],
        na_position="last",
    ).reset_index(drop=True)
    frame.insert(0, "rank", frame.index + 1)
    return frame


def write_leaderboard_markdown(
    *,
    frame: pd.DataFrame,
    output_path: Path,
    all_records: list[dict[str, Any]],
    primary_metric: str,
    secondary_metric: str,
) -> Path:
    sections: list[str] = [
        f"# Leaderboard — atualizado {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n",
        f"- Primary metric: **{primary_metric}**",
        f"- Secondary metric: **{secondary_metric}**",
        f"- Total runs registered: **{len(all_records)}**",
        f"- Unique (model, dataset) pairs: **{len(frame)}**",
        "",
    ]

    if frame.empty:
        sections.append("_No runs available yet._\n")
        output_path.write_text("\n".join(sections), encoding="utf-8")
        return output_path

    sections.append(f"## Top results (sorted by {primary_metric})\n")
    sections.append(_render_top_table(frame))

    sections.append(f"\n## Best model per dataset (by {primary_metric})\n")
    sections.append(_render_best_model_per_dataset(frame, primary_metric))

    sections.append(f"\n## Best dataset per model (by {primary_metric})\n")
    sections.append(_render_best_dataset_per_model(frame, primary_metric))

    sections.append("\n## Model rankings by metric\n")
    sections.append(_render_metric_rankings(frame))

    sections.append("\n## Recent runs (chronological, last 20)\n")
    sections.append(_render_recent_runs(all_records))

    output_path.write_text("\n".join(sections), encoding="utf-8")
    return output_path


def write_leaderboard_csv(frame: pd.DataFrame, output_path: Path) -> Path:
    if frame.empty:
        output_path.write_text("", encoding="utf-8")
        return output_path
    frame.to_csv(output_path, index=False)
    return output_path


def plot_metric_heatmap(
    *,
    best_records: list[dict[str, Any]],
    metric: str,
    output_path: Path,
) -> Path | None:
    if not best_records:
        return None
    frame = pd.DataFrame(best_records)
    if metric not in frame.columns:
        return None
    pivot = frame.pivot_table(
        index="model_name",
        columns="dataset_name",
        values=metric,
        aggfunc="max" if metric not in LOWER_IS_BETTER else "min",
    ).sort_index()
    if pivot.empty:
        return None

    cmap = "magma_r" if metric in LOWER_IS_BETTER else "magma"
    figure_height = max(3, 0.6 * len(pivot.index) + 1.5)
    figure_width = max(5, 1.1 * len(pivot.columns) + 1.5)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        ax=axis,
        cbar=True,
        linewidths=0.4,
    )
    direction = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
    axis.set_title(f"{metric} ({direction})")
    axis.set_xlabel("dataset")
    axis.set_ylabel("model")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def write_per_model_comparisons(
    records: list[dict[str, Any]],
    output_root: Path,
    primary_metric: str,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    if frame.empty:
        return
    for model_name, group in frame.groupby("model_name"):
        directory = output_root / str(model_name)
        directory.mkdir(parents=True, exist_ok=True)
        ordered = group.sort_values(
            by=primary_metric,
            ascending=primary_metric in LOWER_IS_BETTER,
            na_position="last",
        )
        _write_comparison_markdown(
            directory / "comparison.md",
            title=f"{model_name} — comparison across datasets",
            primary_metric=primary_metric,
            frame=ordered,
            grouping_column="dataset_name",
        )
        _plot_comparison_bar(
            directory / "comparison.png",
            frame=ordered,
            label_column="dataset_name",
            metric=primary_metric,
            title=f"{model_name} — {primary_metric} by dataset",
        )


def write_per_dataset_comparisons(
    records: list[dict[str, Any]],
    output_root: Path,
    primary_metric: str,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    if frame.empty:
        return
    for dataset_name, group in frame.groupby("dataset_name"):
        directory = output_root / str(dataset_name)
        directory.mkdir(parents=True, exist_ok=True)
        ordered = group.sort_values(
            by=primary_metric,
            ascending=primary_metric in LOWER_IS_BETTER,
            na_position="last",
        )
        _write_comparison_markdown(
            directory / "comparison.md",
            title=f"{dataset_name} — comparison across models",
            primary_metric=primary_metric,
            frame=ordered,
            grouping_column="model_name",
        )
        _plot_comparison_bar(
            directory / "comparison.png",
            frame=ordered,
            label_column="model_name",
            metric=primary_metric,
            title=f"{dataset_name} — {primary_metric} by model",
        )


def _is_better(candidate: Any, incumbent: Any, metric: str) -> bool:
    if candidate is None:
        return False
    if incumbent is None:
        return True
    try:
        candidate_value = float(candidate)
        incumbent_value = float(incumbent)
    except (TypeError, ValueError):
        return False
    if metric in LOWER_IS_BETTER:
        return candidate_value < incumbent_value
    return candidate_value > incumbent_value


def _render_top_table(frame: pd.DataFrame) -> str:
    columns = [column for column in LEADERBOARD_DISPLAY_COLUMNS if column in frame.columns]
    rendered = frame[columns].copy()
    for column in rendered.columns:
        if column in {"model_name", "dataset_name", "documentation_directory"}:
            continue
        if column == "rank":
            continue
        rendered[column] = rendered[column].map(format_metric)
    return rendered.to_markdown(index=False)


def _render_best_model_per_dataset(
    frame: pd.DataFrame, primary_metric: str
) -> str:
    rows: list[dict[str, Any]] = []
    for dataset_name, group in frame.groupby("dataset_name"):
        ordered = group.sort_values(
            by=primary_metric,
            ascending=primary_metric in LOWER_IS_BETTER,
            na_position="last",
        )
        if ordered.empty:
            continue
        best = ordered.iloc[0]
        runner_up = ordered.iloc[1] if len(ordered) > 1 else None
        gap = (
            float(best[primary_metric]) - float(runner_up[primary_metric])
            if runner_up is not None
            and runner_up[primary_metric] is not None
            and best[primary_metric] is not None
            else None
        )
        rows.append({
            "dataset_name": dataset_name,
            "best_model": best.get("model_name"),
            primary_metric: format_metric(best.get(primary_metric)),
            "runner_up": runner_up.get("model_name") if runner_up is not None else "",
            "gap": format_metric(gap),
        })
    return pd.DataFrame(rows).to_markdown(index=False) if rows else "_no data_"


def _render_best_dataset_per_model(
    frame: pd.DataFrame, primary_metric: str
) -> str:
    rows: list[dict[str, Any]] = []
    for model_name, group in frame.groupby("model_name"):
        ordered = group.sort_values(
            by=primary_metric,
            ascending=primary_metric in LOWER_IS_BETTER,
            na_position="last",
        )
        if ordered.empty:
            continue
        best = ordered.iloc[0]
        rows.append({
            "model_name": model_name,
            "best_dataset": best.get("dataset_name"),
            primary_metric: format_metric(best.get(primary_metric)),
        })
    return pd.DataFrame(rows).to_markdown(index=False) if rows else "_no data_"


def _render_metric_rankings(frame: pd.DataFrame) -> str:
    sections: list[str] = []
    metric_columns = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "cohen_kappa",
        "matthews_correlation_coefficient",
        "roc_auc_macro",
        "log_loss",
    )
    for metric in metric_columns:
        if metric not in frame.columns:
            continue
        ordered = frame.sort_values(
            by=metric,
            ascending=metric in LOWER_IS_BETTER,
            na_position="last",
        )
        if ordered.empty:
            continue
        rows = []
        for position, (_, row) in enumerate(ordered.head(5).iterrows(), start=1):
            rows.append({
                "rank": position,
                "model_name": row.get("model_name"),
                "dataset_name": row.get("dataset_name"),
                metric: format_metric(row.get(metric)),
            })
        if not rows:
            continue
        sections.append(f"### {metric}\n")
        sections.append(pd.DataFrame(rows).to_markdown(index=False))
        sections.append("")
    return "\n".join(sections) if sections else "_no metrics available_"


def _render_recent_runs(records: list[dict[str, Any]]) -> str:
    sortable = [record for record in records if record.get("completed_at")]
    sortable.sort(key=lambda record: record.get("completed_at"), reverse=True)
    rows = []
    for record in sortable[:20]:
        rows.append({
            "completed_at": record.get("completed_at"),
            "model_name": record.get("model_name"),
            "dataset_name": record.get("dataset_name"),
            "accuracy": format_metric(record.get("accuracy")),
            "balanced_accuracy": format_metric(record.get("balanced_accuracy")),
            "duration_seconds": format_metric(record.get("duration_seconds"), decimals=2),
        })
    return pd.DataFrame(rows).to_markdown(index=False) if rows else "_no runs_"


def _write_comparison_markdown(
    output_path: Path,
    *,
    title: str,
    primary_metric: str,
    frame: pd.DataFrame,
    grouping_column: str,
) -> None:
    columns = [
        grouping_column,
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "cohen_kappa",
        "matthews_correlation_coefficient",
        "roc_auc_macro",
        "log_loss",
        "misclassified_count",
    ]
    columns = [column for column in columns if column in frame.columns]
    rendered = frame[columns].copy()
    for column in rendered.columns:
        if column == grouping_column:
            continue
        rendered[column] = rendered[column].map(format_metric)
    sort_direction = "asc" if primary_metric in LOWER_IS_BETTER else "desc"
    sections = [
        f"# {title}\n",
        f"- Primary metric: **{primary_metric}**",
        f"- Rows ordered by {primary_metric} ({sort_direction}).",
        "",
        rendered.to_markdown(index=False),
        "",
        "See `comparison.png` for the visual comparison.",
        "",
    ]
    output_path.write_text("\n".join(sections), encoding="utf-8")


def _plot_comparison_bar(
    output_path: Path,
    *,
    frame: pd.DataFrame,
    label_column: str,
    metric: str,
    title: str,
) -> None:
    plot_frame = frame[[label_column, metric]].dropna()
    if plot_frame.empty:
        return
    figure_height = max(3, 0.45 * len(plot_frame) + 1.5)
    figure, axis = plt.subplots(figsize=(7, figure_height))
    sns.barplot(
        data=plot_frame,
        y=label_column,
        x=metric,
        color="steelblue",
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel(metric)
    axis.set_ylabel(label_column)
    axis.grid(True, axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
