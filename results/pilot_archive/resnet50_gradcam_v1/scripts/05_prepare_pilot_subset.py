from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_PATH = (
    PROJECT_ROOT
    / "results"
    / "predictions"
    / "resnet50_clean_test_predictions.csv"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cub_metadata.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "pilot_subset"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SEED = 42
PILOT_SIZE = 500
NUMBER_OF_CLASSES = 200


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTION_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    predictions = pd.read_csv(PREDICTION_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    return predictions, metadata


def validate_predictions(predictions: pd.DataFrame) -> None:
    required_columns = {
        "image_id",
        "image_path",
        "true_label",
        "true_class_id",
        "predicted_label",
        "predicted_class_id",
        "confidence",
        "correct",
    }

    missing_columns = required_columns.difference(
        predictions.columns
    )

    if missing_columns:
        raise ValueError(
            "Prediction file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if predictions["image_id"].duplicated().any():
        duplicate_count = (
            predictions["image_id"].duplicated().sum()
        )

        raise ValueError(
            f"Found {duplicate_count} duplicate image IDs."
        )

    if len(predictions) != 5794:
        raise ValueError(
            "Expected 5794 test predictions, "
            f"but found {len(predictions)}."
        )


def create_class_names(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    class_name_table = (
        metadata[
            ["class_id", "image_path"]
        ]
        .copy()
    )

    class_name_table["class_name"] = (
        class_name_table["image_path"]
        .str.split("/")
        .str[0]
    )

    class_name_table = (
        class_name_table[
            ["class_id", "class_name"]
        ]
        .drop_duplicates()
        .sort_values("class_id")
    )

    return class_name_table


def allocate_samples_per_class(
    correctly_classified: pd.DataFrame,
    total_size: int,
) -> dict[int, int]:
    class_counts = (
        correctly_classified["true_class_id"]
        .value_counts()
        .sort_index()
    )

    available_classes = class_counts.index.tolist()

    if total_size < len(available_classes):
        raise ValueError(
            "Pilot size is smaller than the number "
            "of represented classes."
        )

    allocation = {
        int(class_id): 1
        for class_id in available_classes
    }

    remaining = total_size - len(available_classes)

    capacities = {
        int(class_id): int(count - 1)
        for class_id, count in class_counts.items()
    }

    while remaining > 0:
        eligible_classes = [
            class_id
            for class_id, capacity in capacities.items()
            if capacity > 0
        ]

        if not eligible_classes:
            break

        eligible_classes.sort(
            key=lambda class_id: (
                capacities[class_id],
                class_id,
            ),
            reverse=True,
        )

        for class_id in eligible_classes:
            if remaining == 0:
                break

            allocation[class_id] += 1
            capacities[class_id] -= 1
            remaining -= 1

    if remaining > 0:
        raise ValueError(
            "Not enough correctly classified samples "
            "to create the requested pilot subset."
        )

    return allocation


def select_pilot_subset(
    correctly_classified: pd.DataFrame,
) -> pd.DataFrame:
    allocation = allocate_samples_per_class(
        correctly_classified=correctly_classified,
        total_size=PILOT_SIZE,
    )

    selected_groups: list[pd.DataFrame] = []

    for class_id, sample_count in allocation.items():
        class_rows = correctly_classified[
            correctly_classified["true_class_id"]
            == class_id
        ]

        sampled_rows = class_rows.sample(
            n=sample_count,
            random_state=SEED + class_id,
        )

        selected_groups.append(sampled_rows)

    pilot_subset = pd.concat(
        selected_groups,
        ignore_index=True,
    )

    pilot_subset = pilot_subset.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(drop=True)

    pilot_subset.insert(
        0,
        "pilot_index",
        np.arange(len(pilot_subset)),
    )

    return pilot_subset


def main() -> None:
    predictions, metadata = load_data()

    validate_predictions(predictions)

    class_names = create_class_names(metadata)

    predictions = predictions.merge(
        class_names,
        left_on="true_class_id",
        right_on="class_id",
        how="left",
    )

    predictions = predictions.drop(
        columns=["class_id"]
    )

    correctly_classified = predictions[
        predictions["correct"] == 1
    ].copy()

    incorrectly_classified = predictions[
        predictions["correct"] == 0
    ].copy()

    pilot_subset = select_pilot_subset(
        correctly_classified
    )

    correctly_classified.to_csv(
        OUTPUT_DIR
        / "correctly_classified_test_images.csv",
        index=False,
    )

    incorrectly_classified.to_csv(
        OUTPUT_DIR
        / "incorrectly_classified_test_images.csv",
        index=False,
    )

    pilot_subset.to_csv(
        OUTPUT_DIR
        / "pilot_500_correctly_classified.csv",
        index=False,
    )

    class_distribution = (
        pilot_subset
        .groupby(
            ["true_class_id", "class_name"],
            as_index=False,
        )
        .agg(
            selected_images=("image_id", "count"),
            mean_confidence=("confidence", "mean"),
            minimum_confidence=("confidence", "min"),
            maximum_confidence=("confidence", "max"),
        )
    )

    class_distribution.to_csv(
        OUTPUT_DIR
        / "pilot_class_distribution.csv",
        index=False,
    )

    summary = {
        "seed": SEED,
        "requested_pilot_size": PILOT_SIZE,
        "total_test_images": int(len(predictions)),
        "correctly_classified_images": int(
            len(correctly_classified)
        ),
        "incorrectly_classified_images": int(
            len(incorrectly_classified)
        ),
        "clean_test_accuracy": float(
            predictions["correct"].mean()
        ),
        "pilot_subset_size": int(
            len(pilot_subset)
        ),
        "classes_represented": int(
            pilot_subset["true_class_id"].nunique()
        ),
        "mean_clean_confidence": float(
            pilot_subset["confidence"].mean()
        ),
        "minimum_clean_confidence": float(
            pilot_subset["confidence"].min()
        ),
        "maximum_clean_confidence": float(
            pilot_subset["confidence"].max()
        ),
    }

    with open(
        OUTPUT_DIR / "pilot_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print("Pilot-subset preparation")
    print("-" * 60)
    print(f"Total test images: {len(predictions)}")
    print(
        "Correctly classified images: "
        f"{len(correctly_classified)}"
    )
    print(
        "Incorrectly classified images: "
        f"{len(incorrectly_classified)}"
    )
    print(
        "Clean test accuracy: "
        f"{predictions['correct'].mean():.4f}"
    )
    print(f"Pilot subset size: {len(pilot_subset)}")
    print(
        "Classes represented: "
        f"{pilot_subset['true_class_id'].nunique()}"
    )
    print(
        "Mean pilot confidence: "
        f"{pilot_subset['confidence'].mean():.4f}"
    )

    print("\nPilot class-count summary")
    print("-" * 60)
    print(
        pilot_subset["true_class_id"]
        .value_counts()
        .value_counts()
        .sort_index()
        .rename_axis("images_per_class")
        .reset_index(name="number_of_classes")
        .to_string(index=False)
    )

    print("\nSaved files")
    print("-" * 60)

    for file_path in sorted(OUTPUT_DIR.iterdir()):
        print(file_path)


if __name__ == "__main__":
    main()