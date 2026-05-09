from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MACHINE_LEARNING_ROOT = PROJECT_ROOT / "machine-learning"
for path in (str(PROJECT_ROOT), str(MACHINE_LEARNING_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from computer_configuration import resolve_device  # noqa: E402
from data_loading import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from models.registry import build_adapter  # noqa: E402

from xai.sample_extraction import ExtractedSamples  # noqa: E402


def apply(
    *,
    samples: ExtractedSamples,
    explanations_directory: Path,
    run_config: dict[str, Any],
    factory_kwargs: dict[str, Any],
    checkpoint_path: Path,
    computer_configuration: dict[str, Any],
    logger: logging.Logger,
) -> int:
    active_run = run_config.get("active_run") or {}
    model_name = active_run["model_name"]

    training = run_config.get("training") or {}
    image_size = int(training.get("image_size", 224))
    num_classes = int(samples.split.test_labels.max()) + 1

    device = resolve_device(computer_configuration)
    adapter = build_adapter(model_name, **factory_kwargs)
    model: nn.Module = adapter.build_model(num_classes=num_classes, image_size=image_size)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()

    target_layers, reshape_transform = _grad_cam_targets(model_name, model)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    )

    written = 0
    for record in samples.records:
        input_tensor = preprocess(record.image_array).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
        predicted_class = int(logits.argmax(dim=1).item())

        targets = [ClassifierOutputTarget(predicted_class)]
        heatmap = cam(input_tensor=input_tensor, targets=targets)[0]

        rgb_image = _resized_rgb_float(record.image_array, image_size)
        overlay = show_cam_on_image(rgb_image, heatmap, use_rgb=True)
        overlay_image = Image.fromarray(overlay)

        class_directory = explanations_directory / record.class_label
        class_directory.mkdir(parents=True, exist_ok=True)
        output_path = class_directory / f"{record.sample_id}.png"
        overlay_image.save(output_path)
        written += 1

    logger.info("Grad-CAM: wrote %d explanation(s)", written)
    return written


def _grad_cam_targets(
    model_name: str,
    model: nn.Module,
) -> tuple[list[nn.Module], Any]:
    backbone = getattr(model, "backbone", model)

    if model_name == "vgg16":
        return [backbone.features[-1]], None
    if model_name == "resnet50":
        return [backbone.layer4[-1]], None
    if model_name == "efficientnet":
        return [backbone.features[-1]], None
    if model_name == "dino":
        target_layer = backbone.blocks[-1].norm1
        return [target_layer], _vit_reshape_transform

    raise NotImplementedError(f"Grad-CAM target layers not configured for '{model_name}'")


def _vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    tokens = tensor[:, 1:, :]
    side = int(tokens.shape[1] ** 0.5)
    reshaped = tokens.reshape(tensor.size(0), side, side, tensor.size(2))
    return reshaped.permute(0, 3, 1, 2)


def _resized_rgb_float(image_array: np.ndarray, image_size: int) -> np.ndarray:
    pil_image = Image.fromarray(image_array).resize(
        (image_size, image_size), Image.BILINEAR
    )
    return np.asarray(pil_image, dtype=np.float32) / 255.0
