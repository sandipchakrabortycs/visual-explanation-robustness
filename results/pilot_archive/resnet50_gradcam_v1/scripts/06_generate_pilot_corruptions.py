from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from tqdm import tqdm


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CUB_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CUB_200_2011"
)

IMAGE_ROOT = CUB_ROOT / "images"

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cub_metadata.csv"
)

PILOT_CSV = (
    PROJECT_ROOT
    / "results"
    / "pilot_subset"
    / "pilot_500_correctly_classified.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "corrupted"
    / "pilot"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "pilot_subset"
    / "pilot_corruption_manifest.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "pilot_subset"
    / "pilot_corruption_summary.json"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42

CORRUPTION_TYPES = [
    "gaussian_noise",
    "motion_blur",
    "brightness",
]

SEVERITY_LEVELS = [1, 3, 5]


GAUSSIAN_NOISE_SIGMA = {
    1: 0.04,
    3: 0.10,
    5: 0.18,
}

MOTION_BLUR_KERNEL_SIZE = {
    1: 5,
    3: 9,
    5: 15,
}

BRIGHTNESS_FACTOR = {
    1: 0.85,
    3: 0.60,
    5: 0.35,
}


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def create_deterministic_seed(
    image_id: int,
    corruption_type: str,
    severity: int,
) -> int:
    key = (
        f"{SEED}_"
        f"{image_id}_"
        f"{corruption_type}_"
        f"{severity}"
    )

    digest = hashlib.sha256(
        key.encode("utf-8")
    ).hexdigest()

    return int(digest[:8], 16)


# ---------------------------------------------------------------------
# Corruption functions
# ---------------------------------------------------------------------

def apply_gaussian_noise(
    image: Image.Image,
    severity: int,
    random_seed: int,
) -> Image.Image:
    sigma = GAUSSIAN_NOISE_SIGMA[severity]

    image_array = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    random_generator = np.random.default_rng(
        random_seed
    )

    noise = random_generator.normal(
        loc=0.0,
        scale=sigma,
        size=image_array.shape,
    )

    corrupted = np.clip(
        image_array + noise,
        0.0,
        1.0,
    )

    corrupted = (
        corrupted * 255.0
    ).astype(np.uint8)

    return Image.fromarray(corrupted)


def apply_motion_blur(
    image: Image.Image,
    severity: int,
) -> Image.Image:
    kernel_size = MOTION_BLUR_KERNEL_SIZE[severity]

    kernel = np.zeros(
        (kernel_size, kernel_size),
        dtype=np.float32,
    )

    center = kernel_size // 2

    # Horizontal motion-blur kernel.
    kernel[center, :] = 1.0
    kernel /= kernel_size

    image_array = np.asarray(
        image,
        dtype=np.uint8,
    )

    blurred = cv2.filter2D(
        image_array,
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT,
    )

    return Image.fromarray(blurred)


def apply_brightness(
    image: Image.Image,
    severity: int,
) -> Image.Image:
    factor = BRIGHTNESS_FACTOR[severity]

    enhancer = ImageEnhance.Brightness(image)

    return enhancer.enhance(factor)


def apply_corruption(
    image: Image.Image,
    corruption_type: str,
    severity: int,
    random_seed: int,
) -> Image.Image:
    if corruption_type == "gaussian_noise":
        return apply_gaussian_noise(
            image=image,
            severity=severity,
            random_seed=random_seed,
        )

    if corruption_type == "motion_blur":
        return apply_motion_blur(
            image=image,
            severity=severity,
        )

    if corruption_type == "brightness":
        return apply_brightness(
            image=image,
            severity=severity,
        )

    raise ValueError(
        f"Unknown corruption type: {corruption_type}"
    )


# ---------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------

def get_output_path(
    original_path: Path,
    image_id: int,
    corruption_type: str,
    severity: int,
) -> Path:
    class_name = original_path.parent.name

    output_directory = (
        OUTPUT_ROOT
        / corruption_type
        / f"severity_{severity}"
        / class_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_filename = (
        f"{image_id}_{original_path.stem}.jpg"
    )

    return output_directory / output_filename


# ---------------------------------------------------------------------
# Validation and path reconstruction
# ---------------------------------------------------------------------

def validate_pilot_dataframe(
    pilot_data: pd.DataFrame,
) -> None:
    required_columns = {
        "pilot_index",
        "image_id",
        "image_path",
        "true_label",
        "true_class_id",
        "predicted_label",
        "predicted_class_id",
        "confidence",
        "correct",
        "class_name",
    }

    missing_columns = required_columns.difference(
        pilot_data.columns
    )

    if missing_columns:
        raise ValueError(
            "Pilot file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(pilot_data) != 500:
        raise ValueError(
            f"Expected 500 pilot images, "
            f"but found {len(pilot_data)}."
        )

    if not (pilot_data["correct"] == 1).all():
        raise ValueError(
            "Pilot subset contains incorrectly "
            "classified images."
        )

    if pilot_data["image_id"].duplicated().any():
        duplicate_count = (
            pilot_data["image_id"]
            .duplicated()
            .sum()
        )

        raise ValueError(
            f"Found {duplicate_count} duplicate image IDs "
            "in the pilot subset."
        )


def attach_local_image_paths(
    pilot_data: pd.DataFrame,
) -> pd.DataFrame:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    metadata = pd.read_csv(
        METADATA_PATH,
        usecols=[
            "image_id",
            "image_path",
        ],
    )

    metadata = metadata.rename(
        columns={
            "image_path": "relative_image_path"
        }
    )

    pilot_data = pilot_data.drop(
        columns=["image_path"],
        errors="ignore",
    )

    pilot_data = pilot_data.merge(
        metadata,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    if pilot_data["relative_image_path"].isna().any():
        missing_ids = pilot_data.loc[
            pilot_data["relative_image_path"].isna(),
            "image_id",
        ].tolist()

        raise ValueError(
            "Metadata paths were not found for image IDs: "
            f"{missing_ids[:10]}"
        )

    pilot_data["local_image_path"] = (
        pilot_data["relative_image_path"]
        .map(
            lambda path: str(
                IMAGE_ROOT / Path(path)
            )
        )
    )

    return pilot_data


# ---------------------------------------------------------------------
# Main generation procedure
# ---------------------------------------------------------------------

def generate_corruptions() -> None:
    if not PILOT_CSV.exists():
        raise FileNotFoundError(
            f"Pilot subset file not found: {PILOT_CSV}"
        )

    pilot_data = pd.read_csv(PILOT_CSV)

    validate_pilot_dataframe(pilot_data)

    pilot_data = attach_local_image_paths(
        pilot_data
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows: list[dict] = []

    total_conditions = (
        len(pilot_data)
        * len(CORRUPTION_TYPES)
        * len(SEVERITY_LEVELS)
    )

    progress_bar = tqdm(
        total=total_conditions,
        desc="Generating corruptions",
    )

    try:
        for _, row in pilot_data.iterrows():
            image_id = int(row["image_id"])

            original_path = Path(
                row["local_image_path"]
            )

            if not original_path.exists():
                raise FileNotFoundError(
                    "Original image not found: "
                    f"{original_path}"
                )

            with Image.open(original_path) as source_image:
                source_image = source_image.convert("RGB")

                original_width, original_height = (
                    source_image.size
                )

                for corruption_type in CORRUPTION_TYPES:
                    for severity in SEVERITY_LEVELS:
                        random_seed = (
                            create_deterministic_seed(
                                image_id=image_id,
                                corruption_type=corruption_type,
                                severity=severity,
                            )
                        )

                        corrupted_image = apply_corruption(
                            image=source_image,
                            corruption_type=corruption_type,
                            severity=severity,
                            random_seed=random_seed,
                        )

                        output_path = get_output_path(
                            original_path=original_path,
                            image_id=image_id,
                            corruption_type=corruption_type,
                            severity=severity,
                        )

                        corrupted_image.save(
                            output_path,
                            format="JPEG",
                            quality=95,
                            subsampling=0,
                        )

                        manifest_rows.append(
                            {
                                "pilot_index": int(
                                    row["pilot_index"]
                                ),
                                "image_id": image_id,
                                "class_name": row["class_name"],
                                "true_label": int(
                                    row["true_label"]
                                ),
                                "true_class_id": int(
                                    row["true_class_id"]
                                ),
                                "clean_predicted_label": int(
                                    row["predicted_label"]
                                ),
                                "clean_predicted_class_id": int(
                                    row["predicted_class_id"]
                                ),
                                "clean_confidence": float(
                                    row["confidence"]
                                ),
                                "corruption_type": (
                                    corruption_type
                                ),
                                "severity": severity,
                                "random_seed": random_seed,
                                "original_width": (
                                    original_width
                                ),
                                "original_height": (
                                    original_height
                                ),
                                "relative_image_path": row[
                                    "relative_image_path"
                                ],
                                "original_path": str(
                                    original_path
                                ),
                                "corrupted_path": str(
                                    output_path
                                ),
                            }
                        )

                        progress_bar.update(1)

    finally:
        progress_bar.close()

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    summary = {
        "seed": SEED,
        "pilot_images": int(len(pilot_data)),
        "corruption_types": CORRUPTION_TYPES,
        "severity_levels": SEVERITY_LEVELS,
        "total_corrupted_images": int(
            len(manifest)
        ),
        "expected_corrupted_images": int(
            total_conditions
        ),
        "output_root": str(OUTPUT_ROOT),
        "manifest_path": str(MANIFEST_PATH),
        "gaussian_noise_sigma": (
            GAUSSIAN_NOISE_SIGMA
        ),
        "motion_blur_kernel_size": (
            MOTION_BLUR_KERNEL_SIZE
        ),
        "brightness_factor": (
            BRIGHTNESS_FACTOR
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

    print("\nPilot corruption generation")
    print("-" * 60)
    print(f"Pilot images: {len(pilot_data)}")
    print(
        f"Corruption types: "
        f"{len(CORRUPTION_TYPES)}"
    )
    print(
        f"Severity levels: "
        f"{SEVERITY_LEVELS}"
    )
    print(
        f"Generated images: "
        f"{len(manifest)}"
    )
    print(
        f"Expected images: "
        f"{total_conditions}"
    )
    print(
        f"Manifest: "
        f"{MANIFEST_PATH}"
    )
    print(
        f"Output root: "
        f"{OUTPUT_ROOT}"
    )

    condition_counts = (
        manifest
        .groupby(
            [
                "corruption_type",
                "severity",
            ]
        )
        .size()
        .reset_index(
            name="image_count"
        )
    )

    print("\nCondition summary")
    print("-" * 60)

    print(
        condition_counts.to_string(
            index=False
        )
    )

    if len(manifest) != total_conditions:
        raise RuntimeError(
            "Generated image count does not "
            "match the expected count."
        )

    missing_files = (
        ~manifest["corrupted_path"]
        .map(
            lambda path: Path(path).exists()
        )
    ).sum()

    if missing_files > 0:
        raise RuntimeError(
            f"{missing_files} generated files "
            "could not be found."
        )

    expected_condition_count = len(pilot_data)

    invalid_conditions = condition_counts[
        condition_counts["image_count"]
        != expected_condition_count
    ]

    if not invalid_conditions.empty:
        raise RuntimeError(
            "One or more corruption conditions "
            "do not contain exactly 500 images."
        )

    print(
        "\nCorruption generation completed "
        "successfully."
    )


if __name__ == "__main__":
    generate_corruptions()