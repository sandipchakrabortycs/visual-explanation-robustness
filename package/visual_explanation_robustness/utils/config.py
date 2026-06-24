from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIG_ROOT


DEFAULT_CONFIG_PATH = (
    CONFIG_ROOT
    / "full_experiment.yaml"
)


def load_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    if config_path is None:
        path = DEFAULT_CONFIG_PATH
    else:
        path = Path(config_path)

        if not path.is_absolute():
            path = Path.cwd() / path

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "The configuration file must contain "
            "a YAML mapping at its root."
        )

    validate_config(config)

    return config


def validate_config(
    config: dict[str, Any],
) -> None:
    required_sections = {
        "experiment",
        "dataset",
        "models",
        "training",
        "corruptions",
        "explanations",
        "stability",
        "localization",
        "faithfulness",
        "statistics",
    }

    missing_sections = (
        required_sections
        - set(config.keys())
    )

    if missing_sections:
        raise ValueError(
            "Missing configuration sections: "
            f"{sorted(missing_sections)}"
        )

    model_names = config[
        "models"
    ].get("names", [])

    expected_models = {
        "resnet50",
        "convnext_tiny",
        "deit_tiny",
        "swin_tiny",
    }

    if set(model_names) != expected_models:
        raise ValueError(
            "The full experiment must contain "
            "exactly these models: "
            f"{sorted(expected_models)}"
        )

    model_settings = (
        config["models"]
        .get("settings", {})
    )

    missing_model_settings = (
        expected_models
        - set(model_settings.keys())
    )

    if missing_model_settings:
        raise ValueError(
            "Missing model-specific settings for: "
            f"{sorted(missing_model_settings)}"
        )

    for model_name in expected_models:
        settings = model_settings[
            model_name
        ]

        required_model_fields = {
            "batch_size",
            "number_of_workers",
            "learning_rate",
            "dropout",
        }

        missing_fields = (
            required_model_fields
            - set(settings.keys())
        )

        if missing_fields:
            raise ValueError(
                f"Model {model_name} is missing "
                f"settings: {sorted(missing_fields)}"
            )

        if int(settings["batch_size"]) <= 0:
            raise ValueError(
                f"Invalid batch size for "
                f"{model_name}."
            )

        if int(settings["number_of_workers"]) < 0:
            raise ValueError(
                f"Invalid number of workers for "
                f"{model_name}."
            )

        if float(settings["learning_rate"]) <= 0:
            raise ValueError(
                f"Invalid learning rate for "
                f"{model_name}."
            )

        dropout = float(
            settings["dropout"]
        )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"Invalid dropout for "
                f"{model_name}."
            )

    severities = config[
        "corruptions"
    ].get("severities", [])

    if severities != [1, 2, 3, 4, 5]:
        raise ValueError(
            "The full experiment must use "
            "severity levels 1, 2, 3, 4 and 5."
        )

    corruption_types = config[
        "corruptions"
    ].get("types", [])

    if len(corruption_types) != 8:
        raise ValueError(
            "The full experiment must contain "
            "eight corruption types."
        )


def get_model_config(
    config: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    available_models = config[
        "models"
    ]["names"]

    if model_name not in available_models:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {available_models}"
        )

    model_config = deepcopy(
        config["training"]
    )

    model_settings = (
        config["models"]
        .get("settings", {})
        .get(model_name, {})
    )

    model_config.update(
        model_settings
    )

    model_config["model_name"] = (
        model_name
    )

    return model_config