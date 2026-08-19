from __future__ import annotations

import torch
from torch import nn
from torchvision import models

try:
    import timm
except ImportError:  # pragma: no cover - only hit when optional dependency is missing
    timm = None


class TinyImageModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_torchvision_model(
    model_name: str,
    *,
    num_classes: int,
    pretrained: bool,
) -> nn.Module:
    if pretrained:
        model = models.get_model(model_name, weights="DEFAULT")
        return _replace_classifier(model, num_classes)

    return models.get_model(model_name, weights=None, num_classes=num_classes)


def _replace_classifier(module: nn.Module, num_classes: int) -> nn.Module:
    if hasattr(module, "heads") and hasattr(module.heads, "head"):
        head = module.heads.head
        if isinstance(head, nn.Linear):
            module.heads.head = nn.Linear(head.in_features, num_classes)
            return module

    if hasattr(module, "fc") and isinstance(module.fc, nn.Linear):
        module.fc = nn.Linear(module.fc.in_features, num_classes)
        return module

    if hasattr(module, "classifier"):
        classifier = module.classifier

        if isinstance(classifier, nn.Linear):
            module.classifier = nn.Linear(classifier.in_features, num_classes)
            return module

        if isinstance(classifier, nn.Sequential):
            layers = list(classifier)
            for index in range(len(layers) - 1, -1, -1):
                if isinstance(layers[index], nn.Linear):
                    layers[index] = nn.Linear(layers[index].in_features, num_classes)
                    module.classifier = nn.Sequential(*layers)
                    return module

    raise ValueError(
        f"do not know how to replace classifier for model type {type(module).__name__}"
    )


def build_model(
    model_name: str,
    *,
    num_classes: int,
    pretrained: bool = False,
) -> nn.Module:
    """Build one of the image-classification models used by the experiments."""
    if num_classes <= 0:
        raise ValueError(f"num_classes must be > 0, got {num_classes}")

    if model_name == "tiny":
        return TinyImageModel(num_classes=num_classes)

    if model_name in models.list_models():
        return _build_torchvision_model(
            model_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )

    if timm is not None and model_name in timm.list_models():
        return timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    if timm is None:
        raise ValueError(
            f"unknown model_name={model_name!r}. The model is not provided by "
            "torchvision and timm is not installed"
        )

    raise ValueError(f"unknown model_name={model_name!r}")


def prepare_model_images(images: torch.Tensor) -> torch.Tensor:
    """Return floating-point image tensors, preserving already-normalized data."""
    if images.dtype == torch.uint8:
        return images.float().div_(255.0)
    return images.float()
