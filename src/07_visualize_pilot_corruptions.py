from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "pilot_subset"
    / "pilot_corruption_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "figures"
    / "pilot_corruptions"
)

CORRUPTION_TYPES = [
    "gaussian_noise",
    "motion_blur",
    "brightness",
]

SEVERITY_LEVELS = [1, 3, 5]

NUMBER_OF_SAMPLES = 5
SEED = 42


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest file not found: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    required_columns = {
        "image_id",
        "class_name",
        "original_path",
        "corrupted_path",
        "corruption_type",
        "severity",
    }

    missing_columns = required_columns.difference(
        manifest.columns
    )

    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing_columns)}"
        )

    return manifest


def visualize_one_image(
    image_id: int,
    corruption_type: str,
    image_rows: pd.DataFrame,
) -> None:
    condition_rows = image_rows[
        image_rows["corruption_type"] == corruption_type
    ].copy()

    condition_rows = condition_rows.sort_values(
        "severity"
    )

    if len(condition_rows) != len(SEVERITY_LEVELS):
        raise ValueError(
            f"Image {image_id} does not have all expected "
            f"severity levels for {corruption_type}."
        )

    original_path = Path(
        condition_rows.iloc[0]["original_path"]
    )

    class_name = condition_rows.iloc[0]["class_name"]

    with Image.open(original_path) as image:
        clean_image = image.convert("RGB")

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(16, 5),
    )

    axes[0].imshow(clean_image)
    axes[0].set_title("Clean")
    axes[0].axis("off")

    for axis_index, severity in enumerate(
        SEVERITY_LEVELS,
        start=1,
    ):
        severity_row = condition_rows[
            condition_rows["severity"] == severity
        ].iloc[0]

        corrupted_path = Path(
            severity_row["corrupted_path"]
        )

        if not corrupted_path.exists():
            raise FileNotFoundError(
                f"Corrupted image not found: {corrupted_path}"
            )

        with Image.open(corrupted_path) as image:
            corrupted_image = image.convert("RGB")

        axes[axis_index].imshow(corrupted_image)
        axes[axis_index].set_title(
            f"Severity {severity}"
        )
        axes[axis_index].axis("off")

    figure.suptitle(
        f"{class_name}\n"
        f"{corruption_type.replace('_', ' ').title()} "
        f"| Image ID: {image_id}",
        fontsize=14,
    )

    figure.tight_layout()

    output_subdirectory = (
        OUTPUT_DIR / corruption_type
    )

    output_subdirectory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_subdirectory
        / f"image_{image_id}_{corruption_type}.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def main() -> None:
    manifest = load_manifest()

    unique_images = (
        manifest[
            [
                "image_id",
                "class_name",
            ]
        ]
        .drop_duplicates()
        .sample(
            n=NUMBER_OF_SAMPLES,
            random_state=SEED,
        )
    )

    print("Pilot corruption visualization")
    print("-" * 60)

    for _, selected_row in unique_images.iterrows():
        image_id = int(selected_row["image_id"])

        image_rows = manifest[
            manifest["image_id"] == image_id
        ]

        for corruption_type in CORRUPTION_TYPES:
            visualize_one_image(
                image_id=image_id,
                corruption_type=corruption_type,
                image_rows=image_rows,
            )

    print("\nVisualization completed successfully.")
    print(f"Check: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()