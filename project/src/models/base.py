"""Base class for galaxy morphology classifiers."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class GalaxyClassifier(ABC):
    """Abstract base class for all galaxy morphology classifiers."""

    def __init__(self):
        self._model = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name (e.g., 'CNN', 'ViT', 'MobileNet')."""
        pass

    @property
    @abstractmethod
    def variant(self) -> str:
        """Model variant: 'light' (weak) or 'robust' (strong)."""
        pass

    @property
    @abstractmethod
    def xai_method(self) -> str:
        """XAI technique used by this model (e.g., 'grad-cam', 'lime', 'attention-rollout')."""
        pass

    @abstractmethod
    def build(self, num_classes: int, img_size: int) -> Optional[object]:
        """
        Build and return the neural network model.

        Args:
            num_classes: Number of output classes (galaxy morphologies)
            img_size: Input image size (assumes square images)

        Returns:
            torch.nn.Module: The built neural network model
        """
        pass

    @abstractmethod
    def explain(self, model, input_tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate an XAI explanation for a given input image.

        Args:
            model: Built nn.Module (output of build())
            input_tensor: Input image tensor of shape (1, C, H, W)
            target_class: Target class index for the explanation.
                          If None, uses the predicted class.

        Returns:
            np.ndarray: Attribution/saliency map of shape (H, W),
                        values normalized to [0, 1].
        """
        pass

    def __repr__(self) -> str:
        return f"{self.name}({self.variant})[{self.xai_method}]"
