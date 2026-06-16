from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DRIVE_PROJECT_ROOT = Path(
    "/content/drive/MyDrive/visual_explanation_robustness"
)

GRADCAM_ROOT = (
    PROJECT_ROOT
    / "results"
    / "pilot_gradcam"
)

GRADCAM_MANIFEST_PATH = (
    GRADCAM_ROOT
    / "pilot_gradcam_manifest.csv"
)

HEATMAP_ROOT = GRADCAM_ROOT / "heatmaps"

LOCAL_SEGMENTATION_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "segmentations"
)

DRIVE_SEGMENTATION_ROOT = (
    DRIVE_PROJECT_ROOT
    / "data"
    / "raw"
    / "segmentations"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "pilot_gradcam_evaluation"
)

PAIR_METRICS_PATH = (
    OUTPUT_ROOT
    / "pilot_gradcam_pair_metrics.csv"
)

CONDITION_METRICS_PATH = (
    OUTPUT_ROOT
    / "pilot_gradcam_condition_metrics.csv"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "pilot_gradcam_evaluation_summary.json"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

IMAGE_SIZE = 224
RESIZE_SIZE = 256
TOP_FRACTION = 0.20
EPSILON = 1e-8


# ---------------------------------------------------------------------
# File resolution
# ---------------------------------------------------------------------

def resolve_heatmap_path(
    relative_heatmap_path: str,
) -> Path:
    path = (
        HEATMAP_ROOT
        / Path(relative_heatmap_path)
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Grad-CAM heatmap not found: {path}"
        )

    return path


def resolve_segmentation_path(
    relative_image_path: str,
) -> Path:
    relative_mask_path = Path(
        relative_image_path
    ).with_suffix(".png")

    local_path = (
        LOCAL_SEGMENTATION_ROOT
        / relative_mask_path
    )

    if local_path.exists():
        return local_path

    drive_path = (
        DRIVE_SEGMENTATION_ROOT
        / relative_mask_path
    )

    if drive_path.exists():
        return drive_path

    raise FileNotFoundError(
        "Segmentation mask was not found:\n"
        f"Local: {local_path}\n"
        f"Drive: {drive_path}"
    )


# ---------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------

def load_heatmap(
    path: Path,
) -> np.ndarray:
    with Image.open(path) as image:
        heatmap = np.asarray(
            image,
            dtype=np.float32,
        )

    maximum = float(heatmap.max())

    if maximum > 1.0:
        heatmap /= 65535.0

    return np.clip(
        heatmap,
        0.0,
        1.0,
    )


@lru_cache(maxsize=1000)
def load_processed_mask(
    relative_image_path: str,
) -> np.ndarray:
    mask_path = resolve_segmentation_path(
        relative_image_path
    )

    with Image.open(mask_path) as mask:
        mask = mask.convert("L")

        mask = TF.resize(
            mask,
            RESIZE_SIZE,
            interpolation=InterpolationMode.NEAREST,
        )

        width, height = mask.size

        top = max(
            0,
            (height - IMAGE_SIZE) // 2,
        )

        left = max(
            0,
            (width - IMAGE_SIZE) // 2,
        )

        mask = TF.crop(
            mask,
            top=top,
            left=left,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
        )

        mask_array = np.asarray(
            mask,
            dtype=np.float32,
        )

    return (
        mask_array > 127
    ).astype(np.uint8)


# ---------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------

def safe_pearson(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_flat = first.ravel()
    second_flat = second.ravel()

    if (
        np.std(first_flat) < EPSILON
        or np.std(second_flat) < EPSILON
    ):
        return 0.0

    value = pearsonr(
        first_flat,
        second_flat,
    ).statistic

    return float(value)


def safe_spearman(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_flat = first.ravel()
    second_flat = second.ravel()

    if (
        np.std(first_flat) < EPSILON
        or np.std(second_flat) < EPSILON
    ):
        return 0.0

    value = spearmanr(
        first_flat,
        second_flat,
    ).statistic

    if np.isnan(value):
        return 0.0

    return float(value)


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_flat = first.ravel()
    second_flat = second.ravel()

    denominator = (
        np.linalg.norm(first_flat)
        * np.linalg.norm(second_flat)
    )

    if denominator < EPSILON:
        return 0.0

    return float(
        np.dot(
            first_flat,
            second_flat,
        )
        / denominator
    )


def calculate_ssim(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    return float(
        structural_similarity(
            first,
            second,
            data_range=1.0,
        )
    )


# ---------------------------------------------------------------------
# Salient-region metrics
# ---------------------------------------------------------------------

def top_fraction_mask(
    heatmap: np.ndarray,
    fraction: float,
) -> np.ndarray:
    flat = heatmap.ravel()

    number_of_pixels = max(
        1,
        int(
            np.ceil(
                fraction * flat.size
            )
        ),
    )

    selected_indices = np.argpartition(
        flat,
        -number_of_pixels,
    )[-number_of_pixels:]

    output = np.zeros(
        flat.shape,
        dtype=np.uint8,
    )

    output[selected_indices] = 1

    return output.reshape(
        heatmap.shape
    )


def binary_iou(
    first_mask: np.ndarray,
    second_mask: np.ndarray,
) -> float:
    intersection = np.logical_and(
        first_mask,
        second_mask,
    ).sum()

    union = np.logical_or(
        first_mask,
        second_mask,
    ).sum()

    if union == 0:
        return 1.0

    return float(
        intersection / union
    )


def weighted_centroid(
    heatmap: np.ndarray,
) -> tuple[float, float]:
    total_mass = float(
        heatmap.sum()
    )

    height, width = heatmap.shape

    if total_mass < EPSILON:
        return (
            (width - 1) / 2.0,
            (height - 1) / 2.0,
        )

    y_coordinates, x_coordinates = np.indices(
        heatmap.shape
    )

    centroid_x = float(
        (
            x_coordinates
            * heatmap
        ).sum()
        / total_mass
    )

    centroid_y = float(
        (
            y_coordinates
            * heatmap
        ).sum()
        / total_mass
    )

    return centroid_x, centroid_y


def normalized_centroid_displacement(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_x, first_y = weighted_centroid(
        first
    )

    second_x, second_y = weighted_centroid(
        second
    )

    distance = np.sqrt(
        (first_x - second_x) ** 2
        + (first_y - second_y) ** 2
    )

    height, width = first.shape

    diagonal = np.sqrt(
        (width - 1) ** 2
        + (height - 1) ** 2
    )

    return float(
        distance / diagonal
    )


# ---------------------------------------------------------------------
# Localization metrics
# ---------------------------------------------------------------------

def attribution_mass_inside_object(
    heatmap: np.ndarray,
    object_mask: np.ndarray,
) -> float:
    total_mass = float(
        heatmap.sum()
    )

    if total_mass < EPSILON:
        return 0.0

    inside_mass = float(
        heatmap[
            object_mask == 1
        ].sum()
    )

    return inside_mass / total_mass


def pointing_game(
    heatmap: np.ndarray,
    object_mask: np.ndarray,
) -> int:
    maximum_index = np.unravel_index(
        np.argmax(heatmap),
        heatmap.shape,
    )

    return int(
        object_mask[maximum_index] == 1
    )


def salient_object_iou(
    heatmap: np.ndarray,
    object_mask: np.ndarray,
) -> float:
    salient_mask = top_fraction_mask(
        heatmap,
        TOP_FRACTION,
    )

    return binary_iou(
        salient_mask,
        object_mask,
    )


# ---------------------------------------------------------------------
# Manifest preparation
# ---------------------------------------------------------------------

def load_manifest() -> pd.DataFrame:
    if not GRADCAM_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Grad-CAM manifest not found: "
            f"{GRADCAM_MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        GRADCAM_MANIFEST_PATH
    )

    required_columns = {
        "image_id",
        "class_name",
        "true_label",
        "true_class_id",
        "source_type",
        "corruption_type",
        "severity",
        "relative_image_path",
        "relative_heatmap_path",
    }

    missing_columns = required_columns.difference(
        manifest.columns
    )

    if missing_columns:
        raise ValueError(
            "Grad-CAM manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    clean_count = (
        manifest["source_type"]
        == "clean"
    ).sum()

    if clean_count != 500:
        raise ValueError(
            f"Expected 500 clean heatmaps, "
            f"but found {clean_count}."
        )

    return manifest


# ---------------------------------------------------------------------
# Pairwise evaluation
# ---------------------------------------------------------------------

def evaluate_pairs(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    clean_rows = manifest[
        manifest["source_type"] == "clean"
    ].copy()

    corrupted_rows = manifest[
        manifest["source_type"]
        == "corrupted"
    ].copy()

    clean_lookup = {
        int(row["image_id"]): row
        for _, row in clean_rows.iterrows()
    }

    metric_rows: list[dict[str, Any]] = []

    for _, corrupted_row in tqdm(
        corrupted_rows.iterrows(),
        total=len(corrupted_rows),
        desc="Evaluating Grad-CAM pairs",
    ):
        image_id = int(
            corrupted_row["image_id"]
        )

        if image_id not in clean_lookup:
            raise ValueError(
                "Clean Grad-CAM map is missing for "
                f"image ID {image_id}."
            )

        clean_row = clean_lookup[image_id]

        clean_heatmap_path = resolve_heatmap_path(
            clean_row[
                "relative_heatmap_path"
            ]
        )

        corrupted_heatmap_path = resolve_heatmap_path(
            corrupted_row[
                "relative_heatmap_path"
            ]
        )

        clean_heatmap = load_heatmap(
            clean_heatmap_path
        )

        corrupted_heatmap = load_heatmap(
            corrupted_heatmap_path
        )

        if (
            clean_heatmap.shape
            != corrupted_heatmap.shape
        ):
            raise ValueError(
                "Heatmap shape mismatch for image "
                f"ID {image_id}."
            )

        object_mask = load_processed_mask(
            corrupted_row[
                "relative_image_path"
            ]
        )

        clean_top_mask = top_fraction_mask(
            clean_heatmap,
            TOP_FRACTION,
        )

        corrupted_top_mask = top_fraction_mask(
            corrupted_heatmap,
            TOP_FRACTION,
        )

        clean_object_mass = (
            attribution_mass_inside_object(
                clean_heatmap,
                object_mask,
            )
        )

        corrupted_object_mass = (
            attribution_mass_inside_object(
                corrupted_heatmap,
                object_mask,
            )
        )

        clean_pointing = pointing_game(
            clean_heatmap,
            object_mask,
        )

        corrupted_pointing = pointing_game(
            corrupted_heatmap,
            object_mask,
        )

        clean_salient_object_iou = (
            salient_object_iou(
                clean_heatmap,
                object_mask,
            )
        )

        corrupted_salient_object_iou = (
            salient_object_iou(
                corrupted_heatmap,
                object_mask,
            )
        )

        metric_rows.append(
            {
                "pilot_index": int(
                    corrupted_row[
                        "pilot_index"
                    ]
                ),
                "image_id": image_id,
                "class_name": corrupted_row[
                    "class_name"
                ],
                "true_label": int(
                    corrupted_row[
                        "true_label"
                    ]
                ),
                "true_class_id": int(
                    corrupted_row[
                        "true_class_id"
                    ]
                ),
                "corruption_type": (
                    corrupted_row[
                        "corruption_type"
                    ]
                ),
                "severity": int(
                    corrupted_row[
                        "severity"
                    ]
                ),
                "pearson_correlation": (
                    safe_pearson(
                        clean_heatmap,
                        corrupted_heatmap,
                    )
                ),
                "spearman_correlation": (
                    safe_spearman(
                        clean_heatmap,
                        corrupted_heatmap,
                    )
                ),
                "cosine_similarity": (
                    cosine_similarity(
                        clean_heatmap,
                        corrupted_heatmap,
                    )
                ),
                "ssim": calculate_ssim(
                    clean_heatmap,
                    corrupted_heatmap,
                ),
                "top20_iou": binary_iou(
                    clean_top_mask,
                    corrupted_top_mask,
                ),
                "centroid_displacement": (
                    normalized_centroid_displacement(
                        clean_heatmap,
                        corrupted_heatmap,
                    )
                ),
                "clean_object_mass": (
                    clean_object_mass
                ),
                "corrupted_object_mass": (
                    corrupted_object_mass
                ),
                "object_mass_change": (
                    corrupted_object_mass
                    - clean_object_mass
                ),
                "object_mass_drop": (
                    clean_object_mass
                    - corrupted_object_mass
                ),
                "clean_pointing_game": (
                    clean_pointing
                ),
                "corrupted_pointing_game": (
                    corrupted_pointing
                ),
                "pointing_game_preserved": int(
                    clean_pointing == 1
                    and corrupted_pointing == 1
                ),
                "clean_salient_object_iou": (
                    clean_salient_object_iou
                ),
                "corrupted_salient_object_iou": (
                    corrupted_salient_object_iou
                ),
                "salient_object_iou_change": (
                    corrupted_salient_object_iou
                    - clean_salient_object_iou
                ),
                "clean_heatmap_path": str(
                    clean_heatmap_path
                ),
                "corrupted_heatmap_path": str(
                    corrupted_heatmap_path
                ),
            }
        )

    return pd.DataFrame(
        metric_rows
    )


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def summarize_conditions(
    pair_metrics: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        pair_metrics
        .groupby(
            [
                "corruption_type",
                "severity",
            ],
            as_index=False,
        )
        .agg(
            image_count=(
                "image_id",
                "count",
            ),
            mean_pearson=(
                "pearson_correlation",
                "mean",
            ),
            mean_spearman=(
                "spearman_correlation",
                "mean",
            ),
            mean_cosine=(
                "cosine_similarity",
                "mean",
            ),
            mean_ssim=(
                "ssim",
                "mean",
            ),
            mean_top20_iou=(
                "top20_iou",
                "mean",
            ),
            mean_centroid_displacement=(
                "centroid_displacement",
                "mean",
            ),
            mean_clean_object_mass=(
                "clean_object_mass",
                "mean",
            ),
            mean_corrupted_object_mass=(
                "corrupted_object_mass",
                "mean",
            ),
            mean_object_mass_drop=(
                "object_mass_drop",
                "mean",
            ),
            clean_pointing_accuracy=(
                "clean_pointing_game",
                "mean",
            ),
            corrupted_pointing_accuracy=(
                "corrupted_pointing_game",
                "mean",
            ),
            pointing_preservation_rate=(
                "pointing_game_preserved",
                "mean",
            ),
            mean_clean_salient_object_iou=(
                "clean_salient_object_iou",
                "mean",
            ),
            mean_corrupted_salient_object_iou=(
                "corrupted_salient_object_iou",
                "mean",
            ),
        )
        .sort_values(
            [
                "corruption_type",
                "severity",
            ]
        )
        .reset_index(drop=True)
    )

    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Pilot Grad-CAM evaluation")
    print("-" * 70)
    print(f"Manifest: {GRADCAM_MANIFEST_PATH}")
    print(f"Heatmaps: {HEATMAP_ROOT}")
    print(
        f"Top salient fraction: "
        f"{TOP_FRACTION:.2f}"
    )

    manifest = load_manifest()

    corrupted_count = int(
        (
            manifest["source_type"]
            == "corrupted"
        ).sum()
    )

    print(
        "Prediction-preserved corrupted maps: "
        f"{corrupted_count}"
    )

    pair_metrics = evaluate_pairs(
        manifest
    )

    condition_metrics = summarize_conditions(
        pair_metrics
    )

    pair_metrics.to_csv(
        PAIR_METRICS_PATH,
        index=False,
    )

    condition_metrics.to_csv(
        CONDITION_METRICS_PATH,
        index=False,
    )

    summary = {
        "clean_images": 500,
        "evaluated_pairs": int(
            len(pair_metrics)
        ),
        "top_fraction": TOP_FRACTION,
        "corruption_types": sorted(
            pair_metrics[
                "corruption_type"
            ].unique().tolist()
        ),
        "severity_levels": sorted(
            pair_metrics[
                "severity"
            ].unique().tolist()
        ),
        "overall_mean_spearman": float(
            pair_metrics[
                "spearman_correlation"
            ].mean()
        ),
        "overall_mean_ssim": float(
            pair_metrics["ssim"].mean()
        ),
        "overall_mean_top20_iou": float(
            pair_metrics[
                "top20_iou"
            ].mean()
        ),
        "overall_mean_centroid_displacement": float(
            pair_metrics[
                "centroid_displacement"
            ].mean()
        ),
        "overall_clean_object_mass": float(
            pair_metrics[
                "clean_object_mass"
            ].mean()
        ),
        "overall_corrupted_object_mass": float(
            pair_metrics[
                "corrupted_object_mass"
            ].mean()
        ),
        "overall_clean_pointing_accuracy": float(
            pair_metrics[
                "clean_pointing_game"
            ].mean()
        ),
        "overall_corrupted_pointing_accuracy": float(
            pair_metrics[
                "corrupted_pointing_game"
            ].mean()
        ),
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    display_columns = [
        "corruption_type",
        "severity",
        "image_count",
        "mean_spearman",
        "mean_ssim",
        "mean_top20_iou",
        "mean_centroid_displacement",
        "mean_corrupted_object_mass",
        "corrupted_pointing_accuracy",
    ]

    print("\nCondition-level explanation results")
    print("-" * 70)

    print(
        condition_metrics[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print("\nOverall explanation results")
    print("-" * 70)
    print(
        "Mean Spearman correlation: "
        f"{summary['overall_mean_spearman']:.4f}"
    )
    print(
        "Mean SSIM: "
        f"{summary['overall_mean_ssim']:.4f}"
    )
    print(
        "Mean top-20% IoU: "
        f"{summary['overall_mean_top20_iou']:.4f}"
    )
    print(
        "Mean centroid displacement: "
        f"{summary['overall_mean_centroid_displacement']:.4f}"
    )
    print(
        "Clean object attribution mass: "
        f"{summary['overall_clean_object_mass']:.4f}"
    )
    print(
        "Corrupted object attribution mass: "
        f"{summary['overall_corrupted_object_mass']:.4f}"
    )
    print(
        "Clean pointing accuracy: "
        f"{summary['overall_clean_pointing_accuracy']:.4f}"
    )
    print(
        "Corrupted pointing accuracy: "
        f"{summary['overall_corrupted_pointing_accuracy']:.4f}"
    )

    print("\nSaved files")
    print("-" * 70)
    print(PAIR_METRICS_PATH)
    print(CONDITION_METRICS_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()