from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "package"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PACKAGE_ROOT),
    )


from visual_explanation_robustness.models import (
    SUPPORTED_MODELS,
    count_parameters,
    create_model,
    get_explanation_target_layer,
    get_model_information,
    validate_model_output,
)


NUMBER_OF_CLASSES = 200
IMAGE_SIZE = 224


def main() -> None:
    print("Unified model-factory test")
    print("-" * 70)

    for model_name in SUPPORTED_MODELS:
        print(f"\nTesting: {model_name}")
        print("-" * 70)

        # Do not download pretrained weights
        # during this structural test.
        model = create_model(
            model_name=model_name,
            number_of_classes=(
                NUMBER_OF_CLASSES
            ),
            pretrained=False,
            dropout=0.30,
        )

        information = get_model_information(
            model_name
        )

        target_layer = (
            get_explanation_target_layer(
                model=model,
                model_name=model_name,
            )
        )

        (
            total_parameters,
            trainable_parameters,
        ) = count_parameters(model)

        output_shape = validate_model_output(
            model=model,
            image_size=IMAGE_SIZE,
            number_of_classes=(
                NUMBER_OF_CLASSES
            ),
            device=torch.device("cpu"),
        )

        print(
            f"Family: {information.family}"
        )
        print(
            f"Input size: "
            f"{information.input_size}"
        )
        print(
            "Explanation target: "
            f"{information.explanation_target_name}"
        )
        print(
            "Target-layer class: "
            f"{target_layer.__class__.__name__}"
        )
        print(
            f"Total parameters: "
            f"{total_parameters:,}"
        )
        print(
            f"Trainable parameters: "
            f"{trainable_parameters:,}"
        )
        print(
            f"Output shape: "
            f"{output_shape}"
        )

        del model
        gc.collect()

    print(
        "\nAll four model architectures "
        "verified successfully."
    )


if __name__ == "__main__":
    main()