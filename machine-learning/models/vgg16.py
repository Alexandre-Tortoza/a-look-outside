from __future__ import annotations

from torch import nn
from torchvision.models import VGG16_Weights, vgg16

from models._deep_learning_base import DeepLearningAdapter


class RedeVGG16(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, batch):
        return self.backbone(batch)


class VGG16Adapter(DeepLearningAdapter):
    name = "vgg16"
    display_name = "VGG16 (torchvision ImageNet)"

    def build_model(self, num_classes: int, image_size: int) -> nn.Module:
        return RedeVGG16(num_classes)
