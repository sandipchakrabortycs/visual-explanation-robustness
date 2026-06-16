from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DRIVE_PROJECT_ROOT = Path(
    "/content/drive/MyDrive/visual_explanation_robustness"
)

LOCAL_IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CUB_200_2011"
    / "images"
)

DRIVE_IMAGE_ROOT = (
    DRIVE_PROJECT_ROOT
    / "data"
    / "raw"
    / "CUB_200_2011"
    / "images"
)

CORRUPTED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "corrupted"
    / "pilot"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet50"
    / "best_resnet50_cub.pth"
)

PILOT_PATH = (
    PROJECT_ROOT
    / "results"
    / "pilot_subset"
    / "pilot_500_correctly_classified.csv"
)

CORRUPTION_MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "pilot_subset"
    / "pilot_corruption_manifest.csv"
)

CORRUPTED_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "pilot_corruption_evaluation"
    / "pilot_corrupted_predictions.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "pilot_gradcam"
)

HEATMAP_ROOT = OUTPUT_ROOT / "heatmaps"

OUTPUT_MANIFEST_PATH = (
    OUTPUT_ROOT
    / "pilot_gradcam_manifest.csv"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "pilot_gradcam_summary.json"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

IMAGE_SIZE = 224
NUMBER_OF_CLASSES = 200
BATCH_SIZE = 32
NUMBER_OF_WORKERS = 2

HEATMAP_MAX_VALUE = 65535


# ---------------------------------------------------------------------
# Path reconstruction
# ---------------------------------------------------------------------

def resolve_clean_image_path(
    relative_image_path: str,
) -> Path:
    relative_path = Path(relative_image_path)

    local_path = LOCAL_IMAGE_ROOT / relative_path

    if local_path.exists():
        return local_path

    drive_path = DRIVE_IMAGE_ROOT / relative_path

    if drive_path.exists():
        return drive_path

    raise FileNotFoundError(
        "Clean image was not found locally or in Drive:\n"
        f"Local: {local_path}\n"
        f"Drive: {drive_path}"
    )


def reconstruct_corrupted_path(
    row: pd.Series,
) -> Path:
    relative_path = Path(
        row["relative_image_path"]
    )

    class_name = relative_path.parent.name

    filename = (
        f"{int(row['image_id'])}_"
        f"{relative_path.stem}.jpg"
    )

    path = (
        CORRUPTED_ROOT
        / row["corruption_type"]
        / f"severity_{int(row['severity'])}"
        / class_name
        / filename
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Corrupted image not found: {path}"
        )

    return path


# ---------------------------------------------------------------------
# Prepare Grad-CAM records
# ---------------------------------------------------------------------

def load_and_prepare_records() -> pd.DataFrame:
    required_files = [
        CHECKPOINT_PATH,
        PILOT_PATH,
        CORRUPTION_MANIFEST_PATH,
        CORRUPTED_PREDICTIONS_PATH,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required files are missing:\n"
            + "\n".join(
                str(path)
                for path in missing_files
            )
        )

    pilot = pd.read_csv(PILOT_PATH)

    corruption_manifest = pd.read_csv(
        CORRUPTION_MANIFEST_PATH
    )

    corrupted_predictions = pd.read_csv(
        CORRUPTED_PREDICTIONS_PATH
    )

    relative_paths = (
        corruption_manifest[
            [
                "image_id",
                "relative_image_path",
            ]
        ]
        .drop_duplicates()
    )

    clean_records = pilot.merge(
        relative_paths,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    if clean_records[
        "relative_image_path"
    ].isna().any():
        raise ValueError(
            "Relative image paths are missing "
            "for some clean pilot images."
        )

    clean_records = clean_records[
        [
            "pilot_index",
            "image_id",
            "class_name",
            "true_label",
            "true_class_id",
            "relative_image_path",
        ]
    ].copy()

    clean_records["source_type"] = "clean"
    clean_records["corruption_type"] = "clean"
    clean_records["severity"] = 0

    preserved_predictions = (
        corrupted_predictions[
            corrupted_predictions[
                "correct_and_preserved"
            ] == 1
        ]
        .copy()
    )

    manifest_columns = (
        corruption_manifest[
            [
                "image_id",
                "corruption_type",
                "severity",
                "relative_image_path",
            ]
        ]
        .drop_duplicates()
    )

    preserved_records = preserved_predictions.merge(
        manifest_columns,
        on=[
            "image_id",
            "corruption_type",
            "severity",
        ],
        how="left",
        validate="one_to_one",
    )

    if preserved_records[
        "relative_image_path"
    ].isna().any():
        raise ValueError(
            "Relative image paths are missing for "
            "some corrupted records."
        )

    preserved_records = preserved_records[
        [
            "pilot_index",
            "image_id",
            "class_name",
            "true_label",
            "true_class_id",
            "corruption_type",
            "severity",
            "relative_image_path",
        ]
    ].copy()

    preserved_records["source_type"] = "corrupted"

    records = pd.concat(
        [
            clean_records,
            preserved_records,
        ],
        ignore_index=True,
    )

    records.insert(
        0,
        "gradcam_index",
        np.arange(len(records)),
    )

    records = records.sort_values(
        [
            "source_type",
            "image_id",
            "corruption_type",
            "severity",
        ]
    ).reset_index(drop=True)

    records["gradcam_index"] = np.arange(
        len(records)
    )

    return records


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class GradCAMDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
    ) -> None:
        self.records = records.reset_index(
            drop=True
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(
                    IMAGE_SIZE
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def resolve_source_path(
        self,
        row: pd.Series,
    ) -> Path:
        if row["source_type"] == "clean":
            return resolve_clean_image_path(
                row["relative_image_path"]
            )

        return reconstruct_corrupted_path(row)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        row = self.records.iloc[index]

        source_path = self.resolve_source_path(
            row
        )

        with Image.open(source_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "record_index": torch.tensor(
                index,
                dtype=torch.long,
            ),
            "target_label": torch.tensor(
                int(row["true_label"]),
                dtype=torch.long,
            ),
            "source_path": str(source_path),
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
# Grad-CAM
# ---------------------------------------------------------------------

class GradCAM:
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ) -> None:
        self.model = model
        self.target_layer = target_layer

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self._forward_hook
            )
        )

    def _save_gradient(
        self,
        gradient: torch.Tensor,
    ) -> None:
        self.gradients = gradient

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output

        output.register_hook(
            self._save_gradient
        )

    def generate(
        self,
        images: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        self.model.zero_grad(
            set_to_none=True
        )

        logits = self.model(images)

        selected_scores = logits.gather(
            1,
            target_labels.view(-1, 1),
        ).sum()

        selected_scores.backward()

        if (
            self.activations is None
            or self.gradients is None
        ):
            raise RuntimeError(
                "Grad-CAM activations or gradients "
                "were not captured."
            )

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights
            * self.activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            mode="bilinear",
            align_corners=False,
        )

        batch_size = cam.shape[0]

        flat_cam = cam.view(
            batch_size,
            -1,
        )

        minimum = flat_cam.min(
            dim=1,
            keepdim=True,
        ).values

        maximum = flat_cam.max(
            dim=1,
            keepdim=True,
        ).values

        normalized = (
            flat_cam - minimum
        ) / (
            maximum - minimum + 1e-8
        )

        normalized = normalized.view(
            batch_size,
            1,
            IMAGE_SIZE,
            IMAGE_SIZE,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidence, predictions = (
            probabilities.max(dim=1)
        )

        return (
            normalized.detach(),
            predictions.detach(),
            confidence.detach(),
        )

    def close(self) -> None:
        self.forward_handle.remove()


# ---------------------------------------------------------------------
# Heatmap saving
# ---------------------------------------------------------------------

def get_heatmap_relative_path(
    row: pd.Series,
) -> Path:
    image_id = int(row["image_id"])

    if row["source_type"] == "clean":
        return (
            Path("clean")
            / f"image_{image_id}.png"
        )

    return (
        Path("corrupted")
        / row["corruption_type"]
        / f"severity_{int(row['severity'])}"
        / f"image_{image_id}.png"
    )


def save_heatmap(
    heatmap: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    heatmap_uint16 = np.round(
        np.clip(
            heatmap,
            0.0,
            1.0,
        )
        * HEATMAP_MAX_VALUE
    ).astype(np.uint16)

    image = Image.fromarray(
        heatmap_uint16,
        mode="I;16",
    )

    image.save(
        output_path,
        format="PNG",
    )


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

def generate_gradcam_maps() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    HEATMAP_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Pilot Grad-CAM generation")
    print("-" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    records = load_and_prepare_records()

    clean_count = int(
        (
            records["source_type"]
            == "clean"
        ).sum()
    )

    corrupted_count = int(
        (
            records["source_type"]
            == "corrupted"
        ).sum()
    )

    print(
        f"Clean images: {clean_count}"
    )
    print(
        "Prediction-preserved corrupted images: "
        f"{corrupted_count}"
    )
    print(
        f"Total Grad-CAM maps: {len(records)}"
    )

    dataset = GradCAMDataset(records)

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

    model, checkpoint = load_model(
        device
    )

    gradcam = GradCAM(
        model=model,
        target_layer=model.layer4[-1],
    )

    output_rows: list[dict[str, Any]] = []

    mismatch_count = 0

    try:
        for batch in tqdm(
            data_loader,
            desc="Generating Grad-CAM",
        ):
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            target_labels = (
                batch["target_label"].to(
                    device,
                    non_blocking=True,
                )
            )

            heatmaps, predictions, confidence = (
                gradcam.generate(
                    images=images,
                    target_labels=target_labels,
                )
            )

            heatmaps = (
                heatmaps
                .squeeze(1)
                .cpu()
                .numpy()
            )

            predictions = (
                predictions.cpu().tolist()
            )

            confidence = (
                confidence.cpu().tolist()
            )

            record_indices = (
                batch["record_index"]
                .cpu()
                .tolist()
            )

            source_paths = batch[
                "source_path"
            ]

            for (
                heatmap,
                predicted_label,
                predicted_confidence,
                record_index,
                source_path,
            ) in zip(
                heatmaps,
                predictions,
                confidence,
                record_indices,
                source_paths,
            ):
                row = records.iloc[
                    record_index
                ]

                target_label = int(
                    row["true_label"]
                )

                prediction_matches_target = int(
                    predicted_label
                    == target_label
                )

                if not prediction_matches_target:
                    mismatch_count += 1

                relative_heatmap_path = (
                    get_heatmap_relative_path(
                        row
                    )
                )

                heatmap_path = (
                    HEATMAP_ROOT
                    / relative_heatmap_path
                )

                save_heatmap(
                    heatmap=heatmap,
                    output_path=heatmap_path,
                )

                output_rows.append(
                    {
                        "gradcam_index": int(
                            row["gradcam_index"]
                        ),
                        "pilot_index": int(
                            row["pilot_index"]
                        ),
                        "image_id": int(
                            row["image_id"]
                        ),
                        "class_name": row[
                            "class_name"
                        ],
                        "true_label": target_label,
                        "true_class_id": int(
                            row["true_class_id"]
                        ),
                        "source_type": row[
                            "source_type"
                        ],
                        "corruption_type": row[
                            "corruption_type"
                        ],
                        "severity": int(
                            row["severity"]
                        ),
                        "predicted_label": int(
                            predicted_label
                        ),
                        "predicted_class_id": int(
                            predicted_label + 1
                        ),
                        "prediction_confidence": float(
                            predicted_confidence
                        ),
                        "prediction_matches_target": (
                            prediction_matches_target
                        ),
                        "relative_image_path": row[
                            "relative_image_path"
                        ],
                        "source_path": str(
                            source_path
                        ),
                        "relative_heatmap_path": str(
                            relative_heatmap_path
                        ),
                        "heatmap_path": str(
                            heatmap_path
                        ),
                    }
                )

    finally:
        gradcam.close()

    output_manifest = pd.DataFrame(
        output_rows
    )

    output_manifest.to_csv(
        OUTPUT_MANIFEST_PATH,
        index=False,
    )

    condition_counts = (
        output_manifest
        .groupby(
            [
                "source_type",
                "corruption_type",
                "severity",
            ],
            as_index=False,
        )
        .agg(
            heatmap_count=(
                "image_id",
                "count",
            ),
            prediction_match_rate=(
                "prediction_matches_target",
                "mean",
            ),
        )
    )

    condition_counts.to_csv(
        OUTPUT_ROOT
        / "pilot_gradcam_condition_counts.csv",
        index=False,
    )

    summary = {
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "target_layer": (
            "model.layer4[-1]"
        ),
        "target_class": (
            "ground-truth class"
        ),
        "image_size": IMAGE_SIZE,
        "heatmap_format": (
            "16-bit normalized PNG"
        ),
        "clean_heatmaps": clean_count,
        "corrupted_heatmaps": (
            corrupted_count
        ),
        "total_heatmaps": int(
            len(output_manifest)
        ),
        "prediction_mismatches": int(
            mismatch_count
        ),
        "output_manifest": str(
            OUTPUT_MANIFEST_PATH
        ),
        "heatmap_root": str(
            HEATMAP_ROOT
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

    print("\nGrad-CAM generation completed")
    print("-" * 60)
    print(
        f"Clean heatmaps: {clean_count}"
    )
    print(
        "Corrupted heatmaps: "
        f"{corrupted_count}"
    )
    print(
        f"Total heatmaps saved: "
        f"{len(output_manifest)}"
    )
    print(
        "Prediction mismatches during "
        f"re-evaluation: {mismatch_count}"
    )
    print(
        f"Manifest: {OUTPUT_MANIFEST_PATH}"
    )
    print(
        f"Heatmap root: {HEATMAP_ROOT}"
    )

    print("\nCondition counts")
    print("-" * 60)
    print(
        condition_counts.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )


if __name__ == "__main__":
    generate_gradcam_maps()