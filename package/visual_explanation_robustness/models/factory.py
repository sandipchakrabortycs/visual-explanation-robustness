from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import timm
import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet50_Weights,
    Swin_T_Weights,
    convnext_tiny,
    resnet50,
    swin_t,
)


SUPPORTED_MODELS = (
    "resnet50",
    "convnext_tiny",
    "deit_tiny",
    "swin_tiny",
)


@dataclass(frozen=True)
class ModelInformation:
    name: str
    family: str
    input_size: int
    explanation_target_name: str


MODEL_INFORMATION = {
    "resnet50": ModelInformation(
        name="resnet50",
        family="classical_cnn",
        input_size=224,
        explanation_target_name="layer4[-1]",
    ),
    "convnext_tiny": ModelInformation(
        name="convnext_tiny",
        family="modern_cnn",
        input_size=224,
        explanation_target_name="features[-1]",
    ),
    "deit_tiny": ModelInformation(
        name="deit_tiny",
        family="global_transformer",
        input_size=224,
        explanation_target_name="blocks[-1].norm1",
    ),
    "swin_tiny": ModelInformation(
        name="swin_tiny",
        family="hierarchical_transformer",
        input_size=224,
        explanation_target_name="features[-1]",
    ),
}


def create_resnet50(
    number_of_classes: int,
    pretrained: bool,
    dropout: float,
) -> nn.Module:
    weights = (
        ResNet50_Weights.DEFAULT
        if pretrained
        else None
    )

    model = resnet50(weights=weights)

    input_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(
            input_features,
            number_of_classes,
        ),
    )

    return model


def create_convnext_tiny(
    number_of_classes: int,
    pretrained: bool,
    dropout: float,
) -> nn.Module:
    weights = (
        ConvNeXt_Tiny_Weights.DEFAULT
        if pretrained
        else None
    )

    model = convnext_tiny(weights=weights)

    input_features = model.classifier[2].in_features

    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(
            input_features,
            number_of_classes,
        ),
    )

    return model


def create_deit_tiny(
    number_of_classes: int,
    pretrained: bool,
    dropout: float,
) -> nn.Module:
    model = timm.create_model(
        "deit_tiny_patch16_224.fb_in1k",
        pretrained=pretrained,
        num_classes=number_of_classes,
        drop_rate=dropout,
    )

    return model


def create_swin_tiny(
    number_of_classes: int,
    pretrained: bool,
    dropout: float,
) -> nn.Module:
    weights = (
        Swin_T_Weights.DEFAULT
        if pretrained
        else None
    )

    model = swin_t(
        weights=weights,
        dropout=dropout,
    )

    input_features = model.head.in_features

    model.head = nn.Linear(
        input_features,
        number_of_classes,
    )

    return model


MODEL_BUILDERS: dict[
    str,
    Callable[
        [int, bool, float],
        nn.Module,
    ],
] = {
    "resnet50": create_resnet50,
    "convnext_tiny": create_convnext_tiny,
    "deit_tiny": create_deit_tiny,
    "swin_tiny": create_swin_tiny,
}


def create_model(
    model_name: str,
    number_of_classes: int = 200,
    pretrained: bool = True,
    dropout: float = 0.30,
) -> nn.Module:
    normalized_name = (
        model_name.strip().lower()
    )

    if normalized_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported models: "
            f"{list(SUPPORTED_MODELS)}"
        )

    if number_of_classes <= 1:
        raise ValueError(
            "number_of_classes must be greater "
            "than one."
        )

    if not 0.0 <= dropout < 1.0:
        raise ValueError(
            "dropout must be in the range "
            "[0.0, 1.0)."
        )

    builder = MODEL_BUILDERS[
        normalized_name
    ]

    model = builder(
        number_of_classes,
        pretrained,
        dropout,
    )

    return model


def get_model_information(
    model_name: str,
) -> ModelInformation:
    normalized_name = (
        model_name.strip().lower()
    )

    if normalized_name not in MODEL_INFORMATION:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return MODEL_INFORMATION[
        normalized_name
    ]


def get_explanation_target_layer(
    model: nn.Module,
    model_name: str,
) -> nn.Module:
    normalized_name = (
        model_name.strip().lower()
    )

    if normalized_name == "resnet50":
        return model.layer4[-1]

    if normalized_name == "convnext_tiny":
        return model.features[-1]

    if normalized_name == "deit_tiny":
        return model.blocks[-1].norm1

    if normalized_name == "swin_tiny":
        return model.features[-1]

    raise ValueError(
        f"Unsupported model: {model_name}"
    )


def count_parameters(
    model: nn.Module,
) -> tuple[int, int]:
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return (
        total_parameters,
        trainable_parameters,
    )


def validate_model_output(
    model: nn.Module,
    image_size: int = 224,
    number_of_classes: int = 200,
    device: torch.device | None = None,
) -> tuple[int, ...]:
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(
        2,
        3,
        image_size,
        image_size,
        device=device,
    )

    with torch.no_grad():
        output = model(dummy_input)

    expected_shape = (
        2,
        number_of_classes,
    )

    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected model output shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(output.shape)}."
        )

    return tuple(output.shape)