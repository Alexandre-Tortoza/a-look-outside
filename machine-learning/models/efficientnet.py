from __future__ import annotations

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from models._deep_learning_base import DeepLearningAdapter


class RedeEfficientNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, batch):
        return self.backbone(batch)


class EfficientNetAdapter(DeepLearningAdapter):
    name = "efficientnet"
    display_name = "EfficientNet-B0 (torchvision ImageNet)"

    def build_model(self, num_classes: int, image_size: int) -> nn.Module:
        return RedeEfficientNet(num_classes)
