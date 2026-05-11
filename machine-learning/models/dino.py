from __future__ import annotations

from typing import Any

import timm
import torch
from torch import nn

from data_loading import DatasetSplits
from models._deep_learning_base import DeepLearningAdapter

DINO_V2_MODEL_NAME = "vit_small_patch14_dinov2"


class RedeDino(nn.Module):
    def __init__(
        self,
        num_classes: int,
        image_size: int,
        model_name: str = DINO_V2_MODEL_NAME,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=image_size,
        )

    def forward(self, batch):
        return self.backbone(batch)


class DinoAdapter(DeepLearningAdapter):
    name = "dino"
    display_name = "DINOv2 ViT-S/14 (self-supervised, Meta 2023)"

    def __init__(
        self,
        model_name: str = DINO_V2_MODEL_NAME,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained

    def build_model(self, num_classes: int, image_size: int) -> nn.Module:
        if "dinov2" not in self.model_name:
            raise ValueError(
                f"DinoAdapter must use a DINOv2 timm model, got '{self.model_name}'"
            )
        return RedeDino(
            num_classes=num_classes,
            image_size=image_size,
            model_name=self.model_name,
            pretrained=self.pretrained,
        )

    def build_criterion(
        self,
        splits: DatasetSplits,
        configuration: dict[str, Any],
        device: torch.device,
    ) -> nn.Module:
        fine_tuning = self._fine_tuning_configuration(configuration)
        label_smoothing = float(fine_tuning.get("label_smoothing", 0.0))
        class_weighting = str(fine_tuning.get("class_weighting", "none"))
        class_weights = None

        if class_weighting != "none":
            counts = torch.bincount(
                torch.as_tensor(splits.train_labels, dtype=torch.long),
                minlength=splits.num_classes,
            ).float()
            counts = counts.clamp_min(1.0)
            if class_weighting == "inverse_frequency":
                weights = counts.sum() / counts
            elif class_weighting == "sqrt_inverse_frequency":
                weights = torch.sqrt(counts.sum() / counts)
            else:
                raise ValueError(
                    "DINO class_weighting must be one of: "
                    "none, inverse_frequency, sqrt_inverse_frequency"
                )
            class_weights = (weights / weights.mean()).to(device)

        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    def build_optimizer(
        self,
        model: nn.Module,
        configuration: dict[str, Any],
        learning_rate: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        fine_tuning = self._fine_tuning_configuration(configuration)
        backbone_lr = float(fine_tuning.get("backbone_learning_rate", learning_rate))
        head_lr = float(fine_tuning.get("head_learning_rate", learning_rate))
        dino_weight_decay = float(fine_tuning.get("weight_decay", weight_decay))

        parameter_groups: dict[tuple[str, bool], list[nn.Parameter]] = {
            ("backbone", True): [],
            ("backbone", False): [],
            ("head", True): [],
            ("head", False): [],
        }

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            is_head = self._is_classifier_parameter(name)
            should_decay = self._should_decay_parameter(name, parameter)
            group_key = ("head" if is_head else "backbone", should_decay)
            parameter_groups[group_key].append(parameter)

        groups = []
        for (scope, should_decay), parameters in parameter_groups.items():
            if not parameters:
                continue
            groups.append(
                {
                    "params": parameters,
                    "lr": head_lr if scope == "head" else backbone_lr,
                    "weight_decay": dino_weight_decay if should_decay else 0.0,
                }
            )

        return torch.optim.AdamW(groups)

    def build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        configuration: dict[str, Any],
    ) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        fine_tuning = self._fine_tuning_configuration(configuration)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(fine_tuning.get("scheduler_factor", 0.5)),
            patience=int(fine_tuning.get("scheduler_patience", 2)),
            min_lr=float(fine_tuning.get("minimum_learning_rate", 1e-6)),
        )

    def _fine_tuning_configuration(
        self,
        configuration: dict[str, Any],
    ) -> dict[str, Any]:
        dino_configuration = (configuration.get("models") or {}).get("dino") or {}
        return dict(dino_configuration.get("fine_tuning") or {})

    @staticmethod
    def _is_classifier_parameter(name: str) -> bool:
        return name.startswith("backbone.head") or ".head." in name

    @staticmethod
    def _should_decay_parameter(name: str, parameter: nn.Parameter) -> bool:
        no_decay_terms = (
            "bias",
            "norm",
            "pos_embed",
            "cls_token",
            "register_tokens",
        )
        return parameter.ndim > 1 and not any(term in name for term in no_decay_terms)
