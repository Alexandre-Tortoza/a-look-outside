"""
Example: Integrating DocGenerator into model training pipeline.

This example shows how to use DocGenerator after a model completes training.
It can be adapted for use in main.py or run_benchmark().
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from utils import DocGenerator


logger = logging.getLogger(__name__)


def save_model_results(
    model_name: str,
    variant: str,
    xai_method: str,
    config: Dict,
    metrics: Dict[str, float],
    images: list = None,
    results_dir: Optional[Path] = None,
) -> Path:
    """
    Save model training results to JSON file.

    Convenience function to create properly-formatted results JSON.

    Args:
        model_name: Name of the model (e.g., "CNN")
        variant: Model variant (e.g., "light", "robust")
        xai_method: XAI technique (e.g., "grad-cam")
        config: Configuration dictionary
        metrics: Dictionary of evaluation metrics
        images: List of image filenames (default: [])
        results_dir: Where to save JSON (default: ./results)

    Returns:
        Path to saved JSON file
    """
    if results_dir is None:
        results_dir = Path("results")

    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model_name": model_name,
        "variant": variant,
        "xai_method": xai_method,
        "config": config,
        "metrics": metrics,
        "images": images or [],
    }

    filename = f"{model_name.lower()}_{variant.lower()}_results.json"
    filepath = results_dir / filename

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {filepath}")
    return filepath


def generate_model_documentation(
    json_results_path: Path, docs_root: Optional[Path] = None
) -> Path:
    """
    Generate markdown documentation from saved results.

    Args:
        json_results_path: Path to JSON results file
        docs_root: Root directory for docs (default: ../docs)

    Returns:
        Path to generated markdown file
    """
    gen = DocGenerator(docs_root=docs_root)
    doc_path = gen.generate_from_json(json_results_path)
    logger.info(f"Generated documentation: {doc_path}")
    return doc_path


# Example usage in a training pipeline:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # After model.train() completes:
    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "architecture": "3 conv blocks (32→64→128)",
        "optimizer": "Adam",
        "augmentation": "RandomCrop, HorizontalFlip",
    }

    metrics = {
        "accuracy": 0.924,
        "loss": 0.1835,
        "precision": 0.915,
        "recall": 0.933,
        "f1": 0.924,
    }

    # Step 1: Save results to JSON
    json_path = save_model_results(
        model_name="CNN",
        variant="light",
        xai_method="grad-cam",
        config=config,
        metrics=metrics,
        images=["gradcam_sample1.png", "gradcam_sample2.png"],
    )

    # Step 2: Generate documentation
    doc_path = generate_model_documentation(json_path)

    print(f"\n✅ Results saved: {json_path}")
    print(f"✅ Documentation generated: {doc_path}")
