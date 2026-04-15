"""Data structures and validation for model documentation generation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class ModelResults:
    """
    Encapsulates model pipeline results for documentation generation.

    This schema defines the expected structure of model results that will be
    converted into markdown documentation. It provides type safety and validation
    for the documentation pipeline.

    Attributes:
        model_name: Name of the model (e.g., 'CNN', 'ViT', 'MobileNet').
        variant: Model variant (e.g., 'light', 'robust').
        xai_method: Explainable AI method used (e.g., 'grad-cam', 'lime').
        config: Dictionary of model configuration parameters (hyperparameters,
                architecture details, etc.).
        metrics: Dictionary of evaluation metrics (accuracy, loss, precision,
                 recall, f1, etc.).
        images: List of relative paths to generated images/visualizations
                (e.g., ['gradcam_sample1.png', 'confusion_matrix.png']).
        timestamp: ISO 8601 timestamp when results were generated. If None,
                  current time is used.
    """

    model_name: str
    variant: str
    xai_method: str
    config: Dict[str, any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    images: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None

    def __post_init__(self):
        """Validate schema after initialization."""
        if not self.model_name or not isinstance(self.model_name, str):
            raise ValueError(
                f"model_name must be a non-empty string, got {self.model_name!r}"
            )
        if not self.variant or not isinstance(self.variant, str):
            raise ValueError(
                f"variant must be a non-empty string, got {self.variant!r}"
            )
        if not self.xai_method or not isinstance(self.xai_method, str):
            raise ValueError(
                f"xai_method must be a non-empty string, got {self.xai_method!r}"
            )
        if not isinstance(self.config, dict):
            raise ValueError(f"config must be a dict, got {type(self.config)}")
        if not isinstance(self.metrics, dict):
            raise ValueError(f"metrics must be a dict, got {type(self.metrics)}")
        if not isinstance(self.images, list):
            raise ValueError(f"images must be a list, got {type(self.images)}")
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    @classmethod
    def from_dict(cls, data: dict) -> "ModelResults":
        """
        Create ModelResults from a dictionary (e.g., loaded from JSON).

        Args:
            data: Dictionary with keys matching ModelResults attributes.

        Returns:
            ModelResults instance.

        Raises:
            ValueError: If required fields are missing or have invalid types.
            KeyError: If required fields are missing.
        """
        required_fields = {"model_name", "variant", "xai_method"}
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            raise KeyError(f"Missing required fields: {missing_fields}")

        return cls(
            model_name=data["model_name"],
            variant=data["variant"],
            xai_method=data["xai_method"],
            config=data.get("config", {}),
            metrics=data.get("metrics", {}),
            images=data.get("images", []),
            timestamp=data.get("timestamp"),
        )
