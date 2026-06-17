from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "package"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PACKAGE_ROOT),
    )


from visual_explanation_robustness.data.cub_dataset import (
    CUBClassificationDataset,
)
from visual_explanation_robustness.data.metadata import (
    load_metadata,
)
from visual_explanation_robustness.data.splits import (
    create_train_validation_test_splits,
    validate_splits,
)
from visual_explanation_robustness.data.transforms import (
    create_evaluation_transform,
    create_train_transform,
)
from visual_explanation_robustness.models import (
    SUPPORTED_MODELS,
    count_parameters,
    create_model,
    train_model,
)
from visual_explanation_robustness.utils.config import (
    get_model_config,
    load_config,
)
from visual_explanation_robustness.utils.paths import (
    get_model_directory,
)
from visual_explanation_robustness.utils.reproducibility import (
    create_torch_generator,
    seed_data_loader_worker,
    set_seed,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a configured CUB-200 model."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=SUPPORTED_MODELS,
        help="Model architecture to train.",
    )

    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional path to the experiment "
            "configuration YAML."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Override the configured maximum "
            "number of epochs."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the configured batch size.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Override the configured number "
            "of DataLoader workers."
        ),
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override the configured learning rate.",
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help=(
            "Limit training batches per epoch. "
            "Useful for smoke tests."
        ),
    )

    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=None,
        help=(
            "Limit validation batches per epoch. "
            "Useful for smoke tests."
        ),
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help=(
            "Disable pretrained initialization. "
            "Useful for an offline smoke test."
        ),
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run one epoch with two training "
            "and two validation batches."
        ),
    )

    return parser.parse_args()


def apply_overrides(
    training_config: dict[str, Any],
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    config = dict(training_config)

    if arguments.epochs is not None:
        config["maximum_epochs"] = (
            arguments.epochs
        )

    if arguments.batch_size is not None:
        config["batch_size"] = (
            arguments.batch_size
        )

    if arguments.workers is not None:
        config["number_of_workers"] = (
            arguments.workers
        )

    if arguments.learning_rate is not None:
        config["learning_rate"] = (
            arguments.learning_rate
        )

    if arguments.no_pretrained:
        config["pretrained"] = False

    if arguments.smoke_test:
        config["maximum_epochs"] = 1
        config["pretrained"] = False

    return config


def create_data_loaders(
    config: dict[str, Any],
    seed: int,
    image_size: int,
) -> tuple[
    DataLoader,
    DataLoader,
    pd.DataFrame,
    pd.DataFrame,
]:
    metadata = load_metadata()

    (
        train_metadata,
        validation_metadata,
        test_metadata,
    ) = create_train_validation_test_splits(
        metadata=metadata,
        validation_fraction=float(
            config["validation_fraction"]
        ),
        seed=seed,
    )

    validate_splits(
        train_metadata=train_metadata,
        validation_metadata=(
            validation_metadata
        ),
        test_metadata=test_metadata,
    )

    train_dataset = CUBClassificationDataset(
        metadata=train_metadata,
        transform=create_train_transform(
            image_size=image_size
        ),
    )

    validation_dataset = (
        CUBClassificationDataset(
            metadata=validation_metadata,
            transform=(
                create_evaluation_transform(
                    image_size=image_size
                )
            ),
        )
    )

    generator = create_torch_generator(
        seed
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(
            config["batch_size"]
        ),
        shuffle=True,
        num_workers=int(
            config["number_of_workers"]
        ),
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=(
            seed_data_loader_worker
        ),
        generator=generator,
        persistent_workers=(
            int(
                config["number_of_workers"]
            )
            > 0
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(
            config["batch_size"]
        ),
        shuffle=False,
        num_workers=int(
            config["number_of_workers"]
        ),
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=(
            seed_data_loader_worker
        ),
        persistent_workers=(
            int(
                config["number_of_workers"]
            )
            > 0
        ),
    )

    return (
        train_loader,
        validation_loader,
        train_metadata,
        validation_metadata,
    )


def main() -> None:
    arguments = parse_arguments()

    full_config = load_config(
        arguments.config
    )

    seed = int(
        full_config["experiment"]["seed"]
    )

    image_size = int(
        full_config["experiment"][
            "image_size"
        ]
    )

    training_config = get_model_config(
        config=full_config,
        model_name=arguments.model,
    )

    training_config[
        "validation_fraction"
    ] = float(
        full_config["dataset"][
            "validation_fraction"
        ]
    )

    training_config = apply_overrides(
        training_config=training_config,
        arguments=arguments,
    )

    max_train_batches = (
        arguments.max_train_batches
    )

    max_validation_batches = (
        arguments.max_validation_batches
    )

    if arguments.smoke_test:
        max_train_batches = 2
        max_validation_batches = 2

    set_seed(
        seed=seed,
        deterministic=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_directory = (
        get_model_directory(
            arguments.model
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Unified model training")
    print("-" * 70)
    print(
        f"Model: {arguments.model}"
    )
    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        "Pretrained: "
        f"{training_config['pretrained']}"
    )
    print(
        "Maximum epochs: "
        f"{training_config['maximum_epochs']}"
    )
    print(
        "Batch size: "
        f"{training_config['batch_size']}"
    )
    print(
        "Workers: "
        f"{training_config['number_of_workers']}"
    )

    (
        train_loader,
        validation_loader,
        train_metadata,
        validation_metadata,
    ) = create_data_loaders(
        config=training_config,
        seed=seed,
        image_size=image_size,
    )

    train_metadata.to_csv(
        output_directory
        / "train_split.csv",
        index=False,
    )

    validation_metadata.to_csv(
        output_directory
        / "validation_split.csv",
        index=False,
    )

    model = create_model(
        model_name=arguments.model,
        number_of_classes=200,
        pretrained=bool(
            training_config["pretrained"]
        ),
        dropout=0.30,
    )

    (
        total_parameters,
        trainable_parameters,
    ) = count_parameters(model)

    print(
        f"Training samples: "
        f"{len(train_loader.dataset)}"
    )
    print(
        f"Validation samples: "
        f"{len(validation_loader.dataset)}"
    )
    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )
    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    with (
        output_directory
        / "resolved_training_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            training_config,
            file,
            indent=4,
        )

    train_model(
        model=model,
        model_name=arguments.model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        device=device,
        output_directory=(
            output_directory
        ),
        training_config=(
            training_config
        ),
        maximum_epochs=int(
            training_config[
                "maximum_epochs"
            ]
        ),
        learning_rate=float(
            training_config[
                "learning_rate"
            ]
        ),
        weight_decay=float(
            training_config[
                "weight_decay"
            ]
        ),
        label_smoothing=float(
            training_config[
                "label_smoothing"
            ]
        ),
        early_stopping_patience=int(
            training_config[
                "early_stopping_patience"
            ]
        ),
        scheduler_patience=int(
            training_config[
                "scheduler_patience"
            ]
        ),
        scheduler_factor=float(
            training_config[
                "scheduler_factor"
            ]
        ),
        use_mixed_precision=bool(
            training_config[
                "mixed_precision"
            ]
        ),
        max_train_batches=(
            max_train_batches
        ),
        max_validation_batches=(
            max_validation_batches
        ),
    )


if __name__ == "__main__":
    main()