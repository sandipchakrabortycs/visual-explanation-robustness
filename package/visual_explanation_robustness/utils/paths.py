from __future__ import annotations

from pathlib import Path


# package/visual_explanation_robustness/utils/paths.py
#
# parents:
# 0 = utils
# 1 = visual_explanation_robustness
# 2 = package
# 3 = project root

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_ROOT = PROJECT_ROOT / "configs"

DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw"
PROCESSED_DATA_ROOT = DATA_ROOT / "processed"
CORRUPTED_DATA_ROOT = DATA_ROOT / "corrupted"

CUB_ROOT = RAW_DATA_ROOT / "CUB_200_2011"
IMAGE_ROOT = CUB_ROOT / "images"
SEGMENTATION_ROOT = RAW_DATA_ROOT / "segmentations"

METADATA_PATH = (
    PROCESSED_DATA_ROOT
    / "cub_metadata.csv"
)

MODEL_ROOT = PROJECT_ROOT / "models"

RESULT_ROOT = PROJECT_ROOT / "results"
FULL_RESULT_ROOT = (
    RESULT_ROOT
    / "full_experiment"
)

FIGURE_ROOT = PROJECT_ROOT / "figures"
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"
DOCUMENTATION_ROOT = PROJECT_ROOT / "docs"


def get_model_directory(
    model_name: str,
) -> Path:
    return MODEL_ROOT / model_name


def get_model_checkpoint_path(
    model_name: str,
    checkpoint_name: str = "best_model.pth",
) -> Path:
    return (
        get_model_directory(model_name)
        / checkpoint_name
    )


def get_prediction_directory(
    model_name: str,
) -> Path:
    return (
        FULL_RESULT_ROOT
        / "predictions"
        / model_name
    )


def get_explanation_directory(
    model_name: str,
    explanation_method: str,
) -> Path:
    return (
        FULL_RESULT_ROOT
        / "explanations"
        / model_name
        / explanation_method
    )


def get_corrupted_directory(
    corruption_type: str,
    severity: int,
) -> Path:
    return (
        CORRUPTED_DATA_ROOT
        / "full"
        / corruption_type
        / f"severity_{severity}"
    )


def create_required_directories() -> None:
    directories = [
        PROCESSED_DATA_ROOT,
        CORRUPTED_DATA_ROOT / "full",
        MODEL_ROOT,
        FULL_RESULT_ROOT,
        FULL_RESULT_ROOT / "predictions",
        FULL_RESULT_ROOT / "explanations",
        FULL_RESULT_ROOT / "stability",
        FULL_RESULT_ROOT / "localization",
        FULL_RESULT_ROOT / "faithfulness",
        FULL_RESULT_ROOT / "statistics",
        FULL_RESULT_ROOT / "tables",
        FULL_RESULT_ROOT / "figures",
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )