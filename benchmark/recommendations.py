from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("recommendations")

SEVERITY_ORDER = ("action", "warning", "info")


@dataclass
class Insight:
    severity: str
    scope: str
    title: str
    detail: str
    suggested_action: str | None = None
    affected_runs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_recommendations(
    jsonl_path: Path,
    runs_root: Path,
    docs_root: Path,
) -> list[Insight]:
    if not jsonl_path.exists():
        return []

    records = _load_jsonl(jsonl_path)
    if not records:
        return []

    insights: list[Insight] = []
    insights.extend(_per_run_rules(records, runs_root))
    insights.extend(_cross_run_rules(records, runs_root))

    insights.sort(key=lambda insight: SEVERITY_ORDER.index(insight.severity))

    docs_root.mkdir(parents=True, exist_ok=True)
    _write_markdown(insights, docs_root / "recommendations.md", len(records))
    _write_json(insights, docs_root / "recommendations.json")
    return insights


def _per_run_rules(
    records: list[dict[str, Any]],
    runs_root: Path,
) -> list[Insight]:
    insights: list[Insight] = []
    for record in records:
        insights.extend(_rule_imbalance_dominance(record))
        insights.extend(_rule_overconfident(record))
        insights.extend(_rule_low_kappa_high_acc(record))
        insights.extend(_rule_threshold_issue(record))
        insights.extend(_rule_per_class_failure(record, runs_root))
    return insights


def _cross_run_rules(
    records: list[dict[str, Any]],
    runs_root: Path,
) -> list[Insight]:
    insights: list[Insight] = []
    insights.extend(_rule_smote_helps(records))
    insights.extend(_rule_dataset_intrinsic_difficulty(records))
    insights.extend(_rule_model_dominance(records))
    insights.extend(_rule_systematic_confusion(records, runs_root))
    insights.extend(_rule_hard_examples(records, runs_root))
    return insights


def _rule_imbalance_dominance(record: dict[str, Any]) -> list[Insight]:
    accuracy = _safe_float(record.get("accuracy"))
    balanced = _safe_float(record.get("balanced_accuracy"))
    if accuracy is None or balanced is None:
        return []
    gap = accuracy - balanced
    if gap <= 0.15:
        return []
    return [Insight(
        severity="warning",
        scope="run",
        title=f"{record['model_name']} on {record['dataset_name']}: imbalance dominance",
        detail=(
            f"accuracy ({accuracy:.3f}) is {gap:.2f} above balanced_accuracy "
            f"({balanced:.3f}). The model is benefiting from class imbalance — "
            "balanced_accuracy is the honest signal."
        ),
        suggested_action=(
            "Use balanced_accuracy / macro_f1 / kappa as primary metric, "
            "and consider class-weighted loss or oversampling."
        ),
        affected_runs=[record.get("run_directory_name", "")],
        metadata={"accuracy": accuracy, "balanced_accuracy": balanced},
    )]


def _rule_overconfident(record: dict[str, Any]) -> list[Insight]:
    log_loss = _safe_float(record.get("log_loss"))
    accuracy = _safe_float(record.get("accuracy"))
    if log_loss is None or accuracy is None:
        return []
    if log_loss <= 1.5 or accuracy <= 0.7:
        return []
    return [Insight(
        severity="warning",
        scope="run",
        title=f"{record['model_name']} on {record['dataset_name']}: overconfident errors",
        detail=(
            f"log_loss ({log_loss:.3f}) is high despite accuracy ({accuracy:.3f}). "
            "The model is very confident when it is wrong."
        ),
        suggested_action=(
            "Apply temperature scaling, label smoothing, "
            "or evaluate the calibration_plot.png for the run."
        ),
        affected_runs=[record.get("run_directory_name", "")],
        metadata={"log_loss": log_loss, "accuracy": accuracy},
    )]


def _rule_low_kappa_high_acc(record: dict[str, Any]) -> list[Insight]:
    kappa = _safe_float(record.get("cohen_kappa"))
    accuracy = _safe_float(record.get("accuracy"))
    if kappa is None or accuracy is None:
        return []
    if kappa >= 0.2 or accuracy <= 0.7:
        return []
    return [Insight(
        severity="action",
        scope="run",
        title=f"{record['model_name']} on {record['dataset_name']}: low kappa, high accuracy",
        detail=(
            f"cohen_kappa ({kappa:.3f}) is near random while accuracy ({accuracy:.3f}) "
            "is high. The model is essentially predicting the majority class."
        ),
        suggested_action=(
            "Treat this run as failed beyond majority-class baseline. "
            "Use class-balanced sampling or rebalance the dataset."
        ),
        affected_runs=[record.get("run_directory_name", "")],
        metadata={"cohen_kappa": kappa, "accuracy": accuracy},
    )]


def _rule_threshold_issue(record: dict[str, Any]) -> list[Insight]:
    roc_auc = _safe_float(record.get("roc_auc_macro"))
    accuracy = _safe_float(record.get("accuracy"))
    if roc_auc is None or accuracy is None:
        return []
    if roc_auc <= 0.80 or accuracy >= 0.65:
        return []
    title_text = (
        f"{record['model_name']} on {record['dataset_name']}: "
        "discriminative but bad threshold"
    )
    return [Insight(
        severity="action",
        scope="run",
        title=title_text,
        detail=(
            f"ROC-AUC macro ({roc_auc:.3f}) is high but accuracy ({accuracy:.3f}) is low. "
            "The model separates classes well, but the decision threshold is poor."
        ),
        suggested_action=(
            "Try class-weighted loss, calibrate per-class thresholds, "
            "or revisit the softmax temperature."
        ),
        affected_runs=[record.get("run_directory_name", "")],
        metadata={"roc_auc_macro": roc_auc, "accuracy": accuracy},
    )]


def _rule_per_class_failure(
    record: dict[str, Any],
    runs_root: Path,
) -> list[Insight]:
    metrics = _load_run_metrics(runs_root, record)
    if metrics is None:
        return []
    per_class = metrics.get("per_class_accuracy") or {}
    classification = metrics.get("classification_report") or {}
    insights: list[Insight] = []
    for class_index_raw, accuracy_value in per_class.items():
        accuracy_value = _safe_float(accuracy_value)
        if accuracy_value is None or accuracy_value >= 0.3:
            continue
        recall = _safe_float(
            (classification.get(str(class_index_raw)) or {}).get("recall")
        )
        insights.append(Insight(
            severity="warning",
            scope="run",
            title=(
                f"{record['model_name']} on {record['dataset_name']}: "
                f"class_{class_index_raw} undetected"
            ),
            detail=(
                f"per-class accuracy = {accuracy_value:.3f}"
                + (f", recall = {recall:.3f}" if recall is not None else "")
                + ". The class is essentially never recovered."
            ),
            suggested_action=(
                f"Target class_{class_index_raw} with oversampling, focal loss, "
                "or augmentation."
            ),
            affected_runs=[record.get("run_directory_name", "")],
            metadata={"class_index": class_index_raw, "accuracy": accuracy_value},
        ))
    return insights


def _rule_smote_helps(records: list[dict[str, Any]]) -> list[Insight]:
    by_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        dataset_name = record.get("dataset_name") or ""
        model_name = record.get("model_name") or ""
        if not dataset_name or not model_name:
            continue
        if dataset_name.endswith("_raw"):
            source = dataset_name[: -len("_raw")]
            by_pair[(model_name, source)]["raw"] = record
        elif "_smote" in dataset_name:
            source = dataset_name.split("_smote", 1)[0]
            by_pair[(model_name, source)]["smote"] = record

    insights: list[Insight] = []
    for (model_name, source), variants in by_pair.items():
        if "raw" not in variants or "smote" not in variants:
            continue
        raw_balanced = _safe_float(variants["raw"].get("balanced_accuracy"))
        smote_balanced = _safe_float(variants["smote"].get("balanced_accuracy"))
        if raw_balanced is None or smote_balanced is None:
            continue
        delta = smote_balanced - raw_balanced
        if delta < 0.10:
            continue
        insights.append(Insight(
            severity="info",
            scope="model",
            title=f"SMOTE benefits {model_name} on {source}",
            detail=(
                f"balanced_accuracy improves by {delta:.2f} when applying SMOTE "
                f"({raw_balanced:.3f} -> {smote_balanced:.3f})."
            ),
            suggested_action="Use the SMOTE variant for production runs of this model.",
            affected_runs=[
                variants["raw"].get("run_directory_name", ""),
                variants["smote"].get("run_directory_name", ""),
            ],
            metadata={"delta_balanced_accuracy": delta},
        ))
    return insights


def _rule_dataset_intrinsic_difficulty(
    records: list[dict[str, Any]],
) -> list[Insight]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("dataset_name"):
            by_dataset[record["dataset_name"]].append(record)

    insights: list[Insight] = []
    for dataset_name, dataset_records in by_dataset.items():
        if len(dataset_records) < 2:
            continue
        balanced_values = [
            _safe_float(record.get("balanced_accuracy"))
            for record in dataset_records
        ]
        balanced_values = [value for value in balanced_values if value is not None]
        if not balanced_values:
            continue
        if max(balanced_values) >= 0.5:
            continue
        insights.append(Insight(
            severity="action",
            scope="dataset",
            title=f"Dataset '{dataset_name}' is intrinsically hard",
            detail=(
                f"All {len(dataset_records)} models on this dataset score below 0.5 "
                "balanced_accuracy."
            ),
            suggested_action=(
                "Audit label quality, increase image resolution, or revisit the data "
                "collection. Try semi-supervised pretraining."
            ),
            affected_runs=[r.get("run_directory_name", "") for r in dataset_records],
            metadata={"max_balanced_accuracy": max(balanced_values)},
        ))
    return insights


def _rule_model_dominance(records: list[dict[str, Any]]) -> list[Insight]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("dataset_name"):
            by_dataset[record["dataset_name"]].append(record)

    insights: list[Insight] = []
    for dataset_name, dataset_records in by_dataset.items():
        scored = []
        for record in dataset_records:
            value = _safe_float(record.get("balanced_accuracy"))
            if value is not None:
                scored.append((value, record))
        if len(scored) < 2:
            continue
        scored.sort(key=lambda pair: pair[0], reverse=True)
        winner_value, winner = scored[0]
        runner_up_value, runner_up = scored[1]
        gap = winner_value - runner_up_value
        if gap < 0.10:
            continue
        insights.append(Insight(
            severity="info",
            scope="dataset",
            title=f"{winner['model_name']} dominates on {dataset_name}",
            detail=(
                f"{winner['model_name']} ({winner_value:.3f}) is {gap:.2f} ahead "
                f"of {runner_up['model_name']} ({runner_up_value:.3f})."
            ),
            suggested_action=(
                "Investigate whether this advantage transfers to similar datasets. "
                "If it does, that model family is the right choice."
            ),
            affected_runs=[
                winner.get("run_directory_name", ""),
                runner_up.get("run_directory_name", ""),
            ],
            metadata={"gap": gap},
        ))
    return insights


def _rule_systematic_confusion(
    records: list[dict[str, Any]],
    runs_root: Path,
) -> list[Insight]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("dataset_name"):
            by_dataset[record["dataset_name"]].append(record)

    insights: list[Insight] = []
    for dataset_name, dataset_records in by_dataset.items():
        if len(dataset_records) < 3:
            continue
        pair_counter: Counter[tuple[int, int]] = Counter()
        for record in dataset_records:
            metrics = _load_run_metrics(runs_root, record)
            if metrics is None:
                continue
            for pair in metrics.get("most_confused_pairs") or []:
                if len(pair) >= 2:
                    pair_counter[(int(pair[0]), int(pair[1]))] += 1
        threshold = max(2, int(0.8 * len(dataset_records)))
        for pair, count in pair_counter.items():
            if count < threshold:
                continue
            insights.append(Insight(
                severity="warning",
                scope="dataset",
                title=(
                    f"Systematic confusion class_{pair[0]} -> class_{pair[1]} "
                    f"on {dataset_name}"
                ),
                detail=(
                    f"{count}/{len(dataset_records)} models confuse this pair. "
                    "Likely label ambiguity rather than model failure."
                ),
                suggested_action=(
                    "Inspect labelled samples for these classes; they may share "
                    "morphological features or be inherently ambiguous."
                ),
                affected_runs=[
                    r.get("run_directory_name", "") for r in dataset_records
                ],
                metadata={"pair": list(pair), "frequency": count},
            ))
    return insights


def _rule_hard_examples(
    records: list[dict[str, Any]],
    runs_root: Path,
) -> list[Insight]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("dataset_name"):
            by_dataset[record["dataset_name"]].append(record)

    insights: list[Insight] = []
    for dataset_name, dataset_records in by_dataset.items():
        if len(dataset_records) < 3:
            continue
        index_counter: Counter[int] = Counter()
        for record in dataset_records:
            misclassified_indices = _load_misclassified_indices(runs_root, record)
            for index in misclassified_indices:
                index_counter[index] += 1
        threshold = max(2, int(0.8 * len(dataset_records)))
        hard_indices = [
            index for index, count in index_counter.items() if count >= threshold
        ]
        if not hard_indices:
            continue
        hard_indices.sort()
        insights.append(Insight(
            severity="info",
            scope="dataset",
            title=f"{len(hard_indices)} hard examples in {dataset_name}",
            detail=(
                f"These test samples are misclassified by {threshold}+ of "
                f"{len(dataset_records)} models. They likely sit in genuinely "
                "ambiguous regions of the input space."
            ),
            suggested_action=(
                "Review predictions.csv for these test_index values across runs. "
                "Consider relabelling or excluding."
            ),
            affected_runs=[r.get("run_directory_name", "") for r in dataset_records],
            metadata={
                "hard_test_indices": hard_indices[:50],
                "total_hard": len(hard_indices),
            },
        ))
    return insights


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    return records


def _load_run_metrics(
    runs_root: Path,
    record: dict[str, Any],
) -> dict[str, Any] | None:
    metrics_path = _resolve_run_path(runs_root, record) / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _load_misclassified_indices(
    runs_root: Path,
    record: dict[str, Any],
) -> list[int]:
    csv_path = _resolve_run_path(runs_root, record) / "misclassified.csv"
    if not csv_path.exists():
        return []
    indices: list[int] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                indices.append(int(row["test_index"]))
            except (KeyError, ValueError):
                continue
    return indices


def _resolve_run_path(runs_root: Path, record: dict[str, Any]) -> Path:
    explicit = record.get("run_directory_path")
    if explicit:
        candidate = Path(explicit)
        if candidate.exists():
            return candidate
    name = record.get("run_directory_name")
    if name:
        return runs_root / name
    return runs_root


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def _write_markdown(
    insights: list[Insight],
    output_path: Path,
    total_records: int,
) -> None:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    sections: list[str] = [
        f"# Recommendations — atualizado {timestamp}\n",
        f"- Total runs analysed: **{total_records}**",
        f"- Total insights: **{len(insights)}**",
        "",
    ]
    if not insights:
        sections.append("_No insights triggered yet. Run more experiments._\n")
        output_path.write_text("\n".join(sections), encoding="utf-8")
        return

    grouped: dict[str, list[Insight]] = defaultdict(list)
    for insight in insights:
        grouped[insight.severity].append(insight)

    severity_titles = {
        "action": "## Action items",
        "warning": "## Warnings",
        "info": "## Info",
    }
    for severity in SEVERITY_ORDER:
        bucket = grouped.get(severity) or []
        if not bucket:
            continue
        sections.append(severity_titles[severity])
        sections.append("")
        for insight in bucket:
            sections.append(f"### {insight.title}")
            sections.append("")
            sections.append(f"- **Scope**: {insight.scope}")
            sections.append(f"- **Detail**: {insight.detail}")
            if insight.suggested_action:
                sections.append(f"- **Action**: {insight.suggested_action}")
            if insight.affected_runs:
                runs_text = ", ".join(filter(None, insight.affected_runs))
                if runs_text:
                    sections.append(f"- **Runs**: `{runs_text}`")
            sections.append("")
    output_path.write_text("\n".join(sections), encoding="utf-8")


def _write_json(insights: list[Insight], output_path: Path) -> None:
    payload = [insight.to_dict() for insight in insights]
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
