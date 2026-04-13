"""Markdown content generation from model results."""

from typing import Dict, List
from utils.schema import ModelResults


def _format_config_section(config: Dict[str, any]) -> str:
    """
    Format model configuration as markdown.

    Args:
        config: Configuration dictionary.

    Returns:
        Formatted markdown string for configuration section.
    """
    if not config:
        return "No configuration available.\n"

    lines = []
    for key, value in config.items():
        formatted_key = key.replace("_", " ").title()
        lines.append(f"- **{formatted_key}**: `{value}`")

    return "\n".join(lines) + "\n"


def _format_metrics_section(metrics: Dict[str, float]) -> str:
    """
    Format evaluation metrics as markdown table.

    Args:
        metrics: Metrics dictionary with metric names as keys and floats as values.

    Returns:
        Formatted markdown table string.
    """
    if not metrics:
        return "No metrics available.\n"

    lines = ["| Metric | Value |", "|--------|-------|"]

    for metric_name, value in metrics.items():
        formatted_name = metric_name.replace("_", " ").title()
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        lines.append(f"| {formatted_name} | {formatted_value} |")

    return "\n".join(lines) + "\n"


def _format_images_section(images: List[str], doc_dir_name: str = "") -> str:
    """
    Format images/visualizations as markdown embeds.

    Each image is embedded as a markdown image reference with a relative path.

    Args:
        images: List of image filenames (relative to the documentation folder).
        doc_dir_name: Name of the documentation directory (for context in alt text).

    Returns:
        Formatted markdown string with embedded images.
    """
    if not images:
        return "No visualizations available.\n"

    lines = []
    for idx, image_path in enumerate(images, 1):
        alt_text = f"{doc_dir_name} visualization {idx}"
        lines.append(f"![{alt_text}]({image_path})")
        lines.append("")

    return "\n".join(lines) + "\n"


def build_markdown_content(results: ModelResults) -> str:
    """
    Build complete markdown documentation from model results.

    The markdown structure includes:
    1. Header with model name, variant, and XAI method
    2. Configuration section
    3. Metrics/Results section
    4. Visualizations/Images section
    5. Metadata (timestamp)

    Args:
        results: ModelResults instance with all pipeline outputs.

    Returns:
        Complete markdown document as string.
    """
    model_id = f"{results.model_name}_{results.variant}"

    content = []

    # Header
    content.append(f"# {results.model_name} - {results.variant.capitalize()} Variant\n")
    content.append(
        f"**XAI Method:** {results.xai_method} | "
        f"**Generated:** {results.timestamp}\n\n"
    )

    # Configuration Section
    content.append("## Configuration\n")
    content.append(_format_config_section(results.config))
    content.append("\n")

    # Results Section
    content.append("## Results\n")
    content.append(_format_metrics_section(results.metrics))
    content.append("\n")

    # Visualizations Section
    if results.images:
        content.append("## Visualizations\n")
        content.append(_format_images_section(results.images, model_id))
        content.append("\n")

    # Metadata
    content.append("---\n")
    content.append(f"*Documentation generated for {model_id}*")

    return "".join(content)
