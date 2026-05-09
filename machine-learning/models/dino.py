from __future__ import annotations

import timm
from torch import nn

from models._deep_learning_base import DeepLearningAdapter


class RedeDino(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch16_224.dino",
            pretrained=True,
            num_classes=num_classes,
        )

    def forward(self, batch):
        return self.backbone(batch)


class DinoAdapter(DeepLearningAdapter):
    name = "dino"
    display_name = "DINO ViT-S/16 (timm self-supervised pretraining)"

    def build_model(self, num_classes: int, image_size: int) -> nn.Module:
        return RedeDino(num_classes)
