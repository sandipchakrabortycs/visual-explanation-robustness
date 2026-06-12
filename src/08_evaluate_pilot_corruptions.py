from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet50"
    / "best_resnet50_cub.pth"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "pilot_subset"
    / "pilot_corruption_manifest.csv"
)

CORRUPTED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "corrupted"
    / "pilot"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "pilot_corruption_evaluation"
)

DETAILED_OUTPUT_PATH = (
    OUTPUT_DIR
    / "pilot_corrupted_predictions.csv"
)

CONDITION_OUTPUT_PATH = (
    OUTPUT_DIR
    / "pilot_condition_metrics.csv"
)

SUMMARY_OUTPUT_PATH = (
    OUTPUT_DIR
    / "pilot_prediction_robustness_summary.json"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

IMAGE_SIZE = 224
NUMBER_OF_CLASSES = 200
BATCH_SIZE = 64
NUMBER_OF_WORKERS = 2


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class CorruptedPilotDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def reconstruct_corrupted_path(
        self,
        row: pd.Series,
    ) -> Path:
        relative_image_path = Path(
            row["relative_image_path"]
        )

        class_name = relative_image_path.parent.name
        original_stem = relative_image_path.stem

        filename = (
            f"{int(row['image_id'])}_"
            f"{original_stem}.jpg"
        )

        return (
            CORRUPTED_ROOT
            / row["corruption_type"]
            / f"severity_{int(row['severity'])}"
            / class_name
            / filename
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        row = self.manifest.iloc[index]

        corrupted_path = self.reconstruct_corrupted_path(
            row
        )

        if not corrupted_path.exists():
            raise FileNotFoundError(
                f"Corrupted image not found: {corrupted_path}"
            )

        with Image.open(corrupted_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "manifest_index": torch.tensor(
                index,
                dtype=torch.long,
            ),
            "image_id": torch.tensor(
                int(row["image_id"]),
                dtype=torch.long,
            ),
            "true_label": torch.tensor(
                int(row["true_label"]),
                dtype=torch.long,
            ),
            "clean_predicted_label": torch.tensor(
                int(row["clean_predicted_label"]),
                dtype=torch.long,
            ),
            "clean_confidence": torch.tensor(
                float(row["clean_confidence"]),
                dtype=torch.float32,
            ),
            "severity": torch.tensor(
                int(row["severity"]),
                dtype=torch.long,
            ),
            "corruption_type": row["corruption_type"],
            "corrupted_path": str(corrupted_path),
        }


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def create_model() -> nn.Module:
    model = resnet50(weights=None)

    input_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(
            input_features,
            NUMBER_OF_CLASSES,
        ),
    )

    return model


def load_model(
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model = create_model()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def load_and_validate_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    required_columns = {
        "pilot_index",
        "image_id",
        "class_name",
        "true_label",
        "true_class_id",
        "clean_predicted_label",
        "clean_predicted_class_id",
        "clean_confidence",
        "corruption_type",
        "severity",
        "relative_image_path",
    }

    missing_columns = required_columns.difference(
        manifest.columns
    )

    if missing_columns:
        raise ValueError(
            "Manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(manifest) != 4500:
        raise ValueError(
            f"Expected 4500 manifest rows, "
            f"but found {len(manifest)}."
        )

    duplicate_conditions = manifest.duplicated(
        subset=[
            "image_id",
            "corruption_type",
            "severity",
        ]
    ).sum()

    if duplicate_conditions > 0:
        raise ValueError(
            f"Found {duplicate_conditions} duplicate "
            "image-condition rows."
        )

    return manifest


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: nn.Module,
    data_loader: DataLoader,
    manifest: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    prediction_rows: list[dict[str, Any]] = []

    for batch in tqdm(
        data_loader,
        desc="Evaluating corrupted images",
    ):
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)

        corrupted_confidence, corrupted_predictions = (
            probabilities.max(dim=1)
        )

        manifest_indices = (
            batch["manifest_index"].cpu().tolist()
        )

        corrupted_predictions = (
            corrupted_predictions.cpu().tolist()
        )

        corrupted_confidence = (
            corrupted_confidence.cpu().tolist()
        )

        for (
            manifest_index,
            predicted_label,
            confidence,
        ) in zip(
            manifest_indices,
            corrupted_predictions,
            corrupted_confidence,
        ):
            row = manifest.iloc[manifest_index]

            true_label = int(row["true_label"])
            clean_predicted_label = int(
                row["clean_predicted_label"]
            )
            clean_confidence = float(
                row["clean_confidence"]
            )

            corrupted_correct = int(
                predicted_label == true_label
            )

            prediction_preserved = int(
                predicted_label
                == clean_predicted_label
            )

            confidence_drop = (
                clean_confidence - confidence
            )

            prediction_rows.append(
                {
                    "pilot_index": int(
                        row["pilot_index"]
                    ),
                    "image_id": int(
                        row["image_id"]
                    ),
                    "class_name": row["class_name"],
                    "true_label": true_label,
                    "true_class_id": int(
                        row["true_class_id"]
                    ),
                    "clean_predicted_label": (
                        clean_predicted_label
                    ),
                    "clean_predicted_class_id": int(
                        row[
                            "clean_predicted_class_id"
                        ]
                    ),
                    "clean_confidence": (
                        clean_confidence
                    ),
                    "corruption_type": row[
                        "corruption_type"
                    ],
                    "severity": int(
                        row["severity"]
                    ),
                    "corrupted_predicted_label": (
                        predicted_label
                    ),
                    "corrupted_predicted_class_id": (
                        predicted_label + 1
                    ),
                    "corrupted_confidence": (
                        confidence
                    ),
                    "confidence_drop": (
                        confidence_drop
                    ),
                    "corrupted_correct": (
                        corrupted_correct
                    ),
                    "prediction_preserved": (
                        prediction_preserved
                    ),
                    "correct_and_preserved": int(
                        corrupted_correct == 1
                        and prediction_preserved == 1
                    ),
                    "corrupted_path": (
                        batch["corrupted_path"][
                            manifest_indices.index(
                                manifest_index
                            )
                        ]
                    ),
                }
            )

    return pd.DataFrame(prediction_rows)


# ---------------------------------------------------------------------
# Condition-level metrics
# ---------------------------------------------------------------------

def calculate_condition_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    metric_rows: list[dict[str, Any]] = []

    grouped = predictions.groupby(
        [
            "corruption_type",
            "severity",
        ]
    )

    for (
        corruption_type,
        severity,
    ), group in grouped:
        accuracy = accuracy_score(
            group["true_label"],
            group["corrupted_predicted_label"],
        )

        macro_f1 = f1_score(
            group["true_label"],
            group["corrupted_predicted_label"],
            average="macro",
            zero_division=0,
        )

        metric_rows.append(
            {
                "corruption_type": corruption_type,
                "severity": int(severity),
                "image_count": int(len(group)),
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
                "prediction_preservation_rate": float(
                    group[
                        "prediction_preserved"
                    ].mean()
                ),
                "correct_and_preserved_rate": float(
                    group[
                        "correct_and_preserved"
                    ].mean()
                ),
                "mean_clean_confidence": float(
                    group[
                        "clean_confidence"
                    ].mean()
                ),
                "mean_corrupted_confidence": float(
                    group[
                        "corrupted_confidence"
                    ].mean()
                ),
                "mean_confidence_drop": float(
                    group[
                        "confidence_drop"
                    ].mean()
                ),
                "median_confidence_drop": float(
                    group[
                        "confidence_drop"
                    ].median()
                ),
                "incorrect_images": int(
                    (
                        group[
                            "corrupted_correct"
                        ]
                        == 0
                    ).sum()
                ),
            }
        )

    return (
        pd.DataFrame(metric_rows)
        .sort_values(
            [
                "corruption_type",
                "severity",
            ]
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def create_summary(
    predictions: pd.DataFrame,
    condition_metrics: pd.DataFrame,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    best_condition = condition_metrics.loc[
        condition_metrics["accuracy"].idxmax()
    ]

    worst_condition = condition_metrics.loc[
        condition_metrics["accuracy"].idxmin()
    ]

    return {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "evaluated_images": int(
            len(predictions)
        ),
        "unique_clean_images": int(
            predictions["image_id"].nunique()
        ),
        "corruption_types": sorted(
            predictions[
                "corruption_type"
            ].unique().tolist()
        ),
        "severity_levels": sorted(
            predictions[
                "severity"
            ].unique().tolist()
        ),
        "overall_corrupted_accuracy": float(
            predictions[
                "corrupted_correct"
            ].mean()
        ),
        "overall_prediction_preservation_rate": float(
            predictions[
                "prediction_preserved"
            ].mean()
        ),
        "overall_correct_and_preserved_rate": float(
            predictions[
                "correct_and_preserved"
            ].mean()
        ),
        "overall_mean_confidence_drop": float(
            predictions[
                "confidence_drop"
            ].mean()
        ),
        "best_condition": {
            "corruption_type": best_condition[
                "corruption_type"
            ],
            "severity": int(
                best_condition["severity"]
            ),
            "accuracy": float(
                best_condition["accuracy"]
            ),
        },
        "worst_condition": {
            "corruption_type": worst_condition[
                "corruption_type"
            ],
            "severity": int(
                worst_condition["severity"]
            ),
            "accuracy": float(
                worst_condition["accuracy"]
            ),
        },
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Pilot corruption evaluation")
    print("-" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")

    manifest = load_and_validate_manifest()

    dataset = CorruptedPilotDataset(
        manifest=manifest
    )

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUMBER_OF_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    model, checkpoint = load_model(device)

    print(
        f"Loaded checkpoint from epoch: "
        f"{checkpoint['epoch']}"
    )
    print(
        f"Corrupted images to evaluate: "
        f"{len(dataset)}"
    )

    predictions = run_inference(
        model=model,
        data_loader=data_loader,
        manifest=manifest,
        device=device,
    )

    condition_metrics = (
        calculate_condition_metrics(
            predictions
        )
    )

    predictions.to_csv(
        DETAILED_OUTPUT_PATH,
        index=False,
    )

    condition_metrics.to_csv(
        CONDITION_OUTPUT_PATH,
        index=False,
    )

    summary = create_summary(
        predictions=predictions,
        condition_metrics=condition_metrics,
        checkpoint=checkpoint,
    )

    with open(
        SUMMARY_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\nCondition-level results")
    print("-" * 60)

    display_columns = [
        "corruption_type",
        "severity",
        "accuracy",
        "macro_f1",
        "prediction_preservation_rate",
        "mean_confidence_drop",
    ]

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

    print("\nOverall results")
    print("-" * 60)
    print(
        "Overall corrupted accuracy: "
        f"{summary['overall_corrupted_accuracy']:.4f}"
    )
    print(
        "Prediction preservation rate: "
        f"{summary['overall_prediction_preservation_rate']:.4f}"
    )
    print(
        "Correct-and-preserved rate: "
        f"{summary['overall_correct_and_preserved_rate']:.4f}"
    )
    print(
        "Mean confidence drop: "
        f"{summary['overall_mean_confidence_drop']:.4f}"
    )

    print("\nSaved files")
    print("-" * 60)
    print(DETAILED_OUTPUT_PATH)
    print(CONDITION_OUTPUT_PATH)
    print(SUMMARY_OUTPUT_PATH)


if __name__ == "__main__":
    main()