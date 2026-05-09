from __future__ import annotations

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from models._deep_learning_base import DeepLearningAdapter


class RedeResNet50(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, batch):
        return self.backbone(batch)


class ResNet50Adapter(DeepLearningAdapter):
    name = "resnet50"
    display_name = "ResNet50 (torchvision ImageNet)"

    def build_model(self, num_classes: int, image_size: int) -> nn.Module:
        return RedeResNet50(num_classes)
