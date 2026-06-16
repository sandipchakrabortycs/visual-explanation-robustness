import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
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

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet50"
    / "best_resnet50_cub.pth"
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

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "pilot_faithfulness_refined"
)

DETAILED_OUTPUT_PATH = (
    OUTPUT_ROOT
    / "pilot_faithfulness_refined_metrics.csv"
)

CONDITION_OUTPUT_PATH = (
    OUTPUT_ROOT
    / "pilot_faithfulness_refined_condition_metrics.csv"
)

SUMMARY_OUTPUT_PATH = (
    OUTPUT_ROOT
    / "pilot_faithfulness_refined_summary.json"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

IMAGE_SIZE = 224
NUMBER_OF_CLASSES = 200

MASKING_FRACTIONS = [0.10, 0.20, 0.30]
BLUR_RADIUS = 12.0
RANDOM_REPEATS = 5

BATCH_SIZE = 16
NUMBER_OF_WORKERS = 2

SEED = 42
EPSILON = 1e-8

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(3, 1, 1)


# ---------------------------------------------------------------------
# Path resolution
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
        "Clean image not found:\n"
        f"Local: {local_path}\n"
        f"Drive: {drive_path}"
    )


def reconstruct_corrupted_path(
    row: pd.Series,
) -> Path:
    relative_path = Path(row["relative_image_path"])
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


def resolve_heatmap_path(
    relative_heatmap_path: str,
) -> Path:
    path = HEATMAP_ROOT / Path(relative_heatmap_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Heatmap not found: {path}"
        )

    return path


# ---------------------------------------------------------------------
# Deterministic seeds and masks
# ---------------------------------------------------------------------

def create_deterministic_seed(
    image_id: int,
    corruption_type: str,
    severity: int,
    masking_fraction: float,
    baseline_type: str,
    repeat_index: int,
) -> int:
    key = (
        f"{SEED}_{image_id}_{corruption_type}_{severity}_"
        f"{masking_fraction:.4f}_{baseline_type}_{repeat_index}"
    )

    digest = hashlib.sha256(
        key.encode("utf-8")
    ).hexdigest()

    return int(digest[:8], 16)


def create_top_fraction_mask(
    heatmap: np.ndarray,
    fraction: float,
) -> np.ndarray:
    flattened = heatmap.ravel()

    number_of_pixels = max(
        1,
        int(np.ceil(fraction * flattened.size)),
    )

    selected_indices = np.argpartition(
        flattened,
        -number_of_pixels,
    )[-number_of_pixels:]

    mask = np.zeros(
        flattened.shape,
        dtype=np.uint8,
    )
    mask[selected_indices] = 1

    return mask.reshape(heatmap.shape)


def create_compact_random_mask(
    height: int,
    width: int,
    target_pixels: int,
    random_seed: int,
) -> np.ndarray:
    """
    Create an exact-area compact random mask.

    Pixels nearest to a random centre under a randomly sampled anisotropic
    distance are selected. This produces a contiguous/compact region while
    preserving exactly the same number of masked pixels as the explanation.
    """
    target_pixels = int(
        np.clip(
            target_pixels,
            1,
            height * width,
        )
    )

    generator = np.random.default_rng(random_seed)

    center_y = generator.uniform(0, height - 1)
    center_x = generator.uniform(0, width - 1)
    aspect_ratio = generator.uniform(0.5, 2.0)

    y_coordinates, x_coordinates = np.indices(
        (height, width),
        dtype=np.float32,
    )

    distances = (
        ((y_coordinates - center_y) ** 2) * aspect_ratio
        + ((x_coordinates - center_x) ** 2) / aspect_ratio
    )

    selected_indices = np.argpartition(
        distances.ravel(),
        target_pixels - 1,
    )[:target_pixels]

    mask = np.zeros(
        height * width,
        dtype=np.uint8,
    )
    mask[selected_indices] = 1

    return mask.reshape(height, width)


def create_relocated_explanation_mask(
    mask: np.ndarray,
    random_seed: int,
) -> np.ndarray:
    """
    Relocate the tight explanation-mask shape without wrap-around.

    The number of selected pixels and the shape of the explanation region
    are preserved. Only its spatial location changes.
    """
    selected = np.argwhere(mask > 0)

    if selected.size == 0:
        return mask.copy()

    min_y, min_x = selected.min(axis=0)
    max_y, max_x = selected.max(axis=0)

    cropped = mask[
        min_y:max_y + 1,
        min_x:max_x + 1,
    ]

    crop_height, crop_width = cropped.shape
    height, width = mask.shape

    generator = np.random.default_rng(random_seed)

    max_top = height - crop_height
    max_left = width - crop_width

    top = int(
        generator.integers(
            0,
            max_top + 1,
        )
    )
    left = int(
        generator.integers(
            0,
            max_left + 1,
        )
    )

    original_top = int(min_y)
    original_left = int(min_x)

    if (
        max_top > 0 or max_left > 0
    ) and (
        top == original_top
        and left == original_left
    ):
        if max_left > 0:
            left = (left + 1) % (max_left + 1)
        elif max_top > 0:
            top = (top + 1) % (max_top + 1)

    relocated = np.zeros_like(
        mask,
        dtype=np.uint8,
    )

    relocated[
        top:top + crop_height,
        left:left + crop_width,
    ] = cropped

    return relocated


# ---------------------------------------------------------------------
# Heatmap processing
# ---------------------------------------------------------------------

def load_heatmap(
    path: Path,
) -> np.ndarray:
    with Image.open(path) as image:
        heatmap = np.asarray(
            image,
            dtype=np.float32,
        )

    if float(heatmap.max()) > 1.0:
        heatmap /= 65535.0

    return np.clip(
        heatmap,
        0.0,
        1.0,
    )


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class FaithfulnessDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
    ) -> None:
        self.manifest = manifest.reset_index(
            drop=True
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.manifest)

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
        row = self.manifest.iloc[index]

        source_path = self.resolve_source_path(row)

        heatmap_path = resolve_heatmap_path(
            row["relative_heatmap_path"]
        )

        with Image.open(source_path) as image:
            image = image.convert("RGB")

            blurred_image = image.filter(
                ImageFilter.GaussianBlur(
                    radius=BLUR_RADIUS
                )
            )

            image_tensor = self.transform(image)
            blurred_tensor = self.transform(
                blurred_image
            )

        heatmap = load_heatmap(heatmap_path)

        if heatmap.shape != (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise ValueError(
                "Unexpected heatmap shape for "
                f"image ID {row['image_id']}: "
                f"{heatmap.shape}"
            )

        return {
            "image": image_tensor,
            "blurred_image": blurred_tensor,
            "heatmap": torch.from_numpy(
                heatmap
            ).float(),
            "target_label": torch.tensor(
                int(row["true_label"]),
                dtype=torch.long,
            ),
            "record_index": torch.tensor(
                index,
                dtype=torch.long,
            ),
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
# Masking and inference
# ---------------------------------------------------------------------

def normalize_images(
    images: torch.Tensor,
) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)

    return (images - mean) / std


def apply_blurred_replacement(
    images: torch.Tensor,
    blurred_images: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    expanded_masks = masks.unsqueeze(1).expand(
        -1,
        3,
        -1,
        -1,
    )

    return torch.where(
        expanded_masks,
        blurred_images,
        images,
    )


@torch.no_grad()
def get_target_probabilities(
    model: nn.Module,
    images: torch.Tensor,
    target_labels: torch.Tensor,
) -> torch.Tensor:
    normalized_images = normalize_images(images)
    logits = model(normalized_images)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    return probabilities.gather(
        1,
        target_labels.view(-1, 1),
    ).squeeze(1)


# ---------------------------------------------------------------------
# Manifest validation
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
        "gradcam_index",
        "pilot_index",
        "image_id",
        "class_name",
        "true_label",
        "true_class_id",
        "source_type",
        "corruption_type",
        "severity",
        "prediction_confidence",
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

    if len(manifest) != 4263:
        raise ValueError(
            f"Expected 4263 Grad-CAM records, "
            f"but found {len(manifest)}."
        )

    return manifest.reset_index(drop=True)


# ---------------------------------------------------------------------
# Faithfulness evaluation
# ---------------------------------------------------------------------

def build_masks_for_batch(
    heatmaps: np.ndarray,
    batch_rows: list[pd.Series],
    masking_fraction: float,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
]:
    explanation_masks: list[np.ndarray] = []
    compact_random_repeats: list[list[np.ndarray]] = [
        []
        for _ in range(RANDOM_REPEATS)
    ]
    relocated_repeats: list[list[np.ndarray]] = [
        []
        for _ in range(RANDOM_REPEATS)
    ]

    for heatmap, row in zip(
        heatmaps,
        batch_rows,
    ):
        explanation_mask = create_top_fraction_mask(
            heatmap=heatmap,
            fraction=masking_fraction,
        )

        explanation_masks.append(
            explanation_mask
        )

        target_pixels = int(
            explanation_mask.sum()
        )

        for repeat_index in range(
            RANDOM_REPEATS
        ):
            compact_seed = create_deterministic_seed(
                image_id=int(row["image_id"]),
                corruption_type=str(
                    row["corruption_type"]
                ),
                severity=int(row["severity"]),
                masking_fraction=masking_fraction,
                baseline_type="compact_random",
                repeat_index=repeat_index,
            )

            relocated_seed = create_deterministic_seed(
                image_id=int(row["image_id"]),
                corruption_type=str(
                    row["corruption_type"]
                ),
                severity=int(row["severity"]),
                masking_fraction=masking_fraction,
                baseline_type="relocated_explanation",
                repeat_index=repeat_index,
            )

            compact_random_repeats[
                repeat_index
            ].append(
                create_compact_random_mask(
                    height=IMAGE_SIZE,
                    width=IMAGE_SIZE,
                    target_pixels=target_pixels,
                    random_seed=compact_seed,
                )
            )

            relocated_repeats[
                repeat_index
            ].append(
                create_relocated_explanation_mask(
                    mask=explanation_mask,
                    random_seed=relocated_seed,
                )
            )

    explanation_tensor = torch.from_numpy(
        np.stack(explanation_masks)
    ).bool()

    compact_tensors = [
        torch.from_numpy(
            np.stack(repeat_masks)
        ).bool()
        for repeat_masks in compact_random_repeats
    ]

    relocated_tensors = [
        torch.from_numpy(
            np.stack(repeat_masks)
        ).bool()
        for repeat_masks in relocated_repeats
    ]

    return (
        explanation_tensor,
        compact_tensors,
        relocated_tensors,
    )


def evaluate_faithfulness(
    model: nn.Module,
    data_loader: DataLoader,
    manifest: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    output_rows: list[dict[str, Any]] = []

    for batch in tqdm(
        data_loader,
        desc="Evaluating refined faithfulness",
    ):
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        blurred_images = batch[
            "blurred_image"
        ].to(
            device,
            non_blocking=True,
        )

        target_labels = batch[
            "target_label"
        ].to(
            device,
            non_blocking=True,
        )

        record_indices = batch[
            "record_index"
        ].cpu().tolist()

        heatmaps = batch[
            "heatmap"
        ].cpu().numpy()

        batch_rows = [
            manifest.iloc[index]
            for index in record_indices
        ]

        original_probabilities = (
            get_target_probabilities(
                model=model,
                images=images,
                target_labels=target_labels,
            )
        )

        for masking_fraction in MASKING_FRACTIONS:
            (
                explanation_masks,
                compact_random_masks,
                relocated_masks,
            ) = build_masks_for_batch(
                heatmaps=heatmaps,
                batch_rows=batch_rows,
                masking_fraction=masking_fraction,
            )

            explanation_masks = (
                explanation_masks.to(
                    device,
                    non_blocking=True,
                )
            )

            explanation_masked_images = (
                apply_blurred_replacement(
                    images=images,
                    blurred_images=blurred_images,
                    masks=explanation_masks,
                )
            )

            explanation_probabilities = (
                get_target_probabilities(
                    model=model,
                    images=explanation_masked_images,
                    target_labels=target_labels,
                )
            )

            compact_probability_repeats: list[
                torch.Tensor
            ] = []

            relocated_probability_repeats: list[
                torch.Tensor
            ] = []

            for repeat_index in range(
                RANDOM_REPEATS
            ):
                compact_masks = (
                    compact_random_masks[
                        repeat_index
                    ].to(
                        device,
                        non_blocking=True,
                    )
                )

                relocated_repeat_masks = (
                    relocated_masks[
                        repeat_index
                    ].to(
                        device,
                        non_blocking=True,
                    )
                )

                compact_masked_images = (
                    apply_blurred_replacement(
                        images=images,
                        blurred_images=blurred_images,
                        masks=compact_masks,
                    )
                )

                relocated_masked_images = (
                    apply_blurred_replacement(
                        images=images,
                        blurred_images=blurred_images,
                        masks=relocated_repeat_masks,
                    )
                )

                compact_probability_repeats.append(
                    get_target_probabilities(
                        model=model,
                        images=compact_masked_images,
                        target_labels=target_labels,
                    )
                )

                relocated_probability_repeats.append(
                    get_target_probabilities(
                        model=model,
                        images=relocated_masked_images,
                        target_labels=target_labels,
                    )
                )

            compact_probabilities = torch.stack(
                compact_probability_repeats,
                dim=1,
            )

            relocated_probabilities = torch.stack(
                relocated_probability_repeats,
                dim=1,
            )

            explanation_drops = (
                original_probabilities
                - explanation_probabilities
            )

            compact_drops = (
                original_probabilities.unsqueeze(1)
                - compact_probabilities
            )

            relocated_drops = (
                original_probabilities.unsqueeze(1)
                - relocated_probabilities
            )

            mean_compact_drops = compact_drops.mean(
                dim=1
            )
            std_compact_drops = compact_drops.std(
                dim=1,
                unbiased=False,
            )

            mean_relocated_drops = relocated_drops.mean(
                dim=1
            )
            std_relocated_drops = relocated_drops.std(
                dim=1,
                unbiased=False,
            )

            advantage_over_compact = (
                explanation_drops
                - mean_compact_drops
            )

            advantage_over_relocated = (
                explanation_drops
                - mean_relocated_drops
            )

            relative_explanation_drops = (
                explanation_drops
                / (
                    original_probabilities
                    + EPSILON
                )
            )

            relative_compact_drops = (
                mean_compact_drops
                / (
                    original_probabilities
                    + EPSILON
                )
            )

            relative_relocated_drops = (
                mean_relocated_drops
                / (
                    original_probabilities
                    + EPSILON
                )
            )

            for batch_index, row in enumerate(
                batch_rows
            ):
                output_row: dict[str, Any] = {
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
                    "true_label": int(
                        row["true_label"]
                    ),
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
                    "masking_fraction": float(
                        masking_fraction
                    ),
                    "blur_radius": float(
                        BLUR_RADIUS
                    ),
                    "random_repeats": int(
                        RANDOM_REPEATS
                    ),
                    "original_target_probability": float(
                        original_probabilities[
                            batch_index
                        ].cpu()
                    ),
                    "explanation_masked_probability": float(
                        explanation_probabilities[
                            batch_index
                        ].cpu()
                    ),
                    "explanation_confidence_drop": float(
                        explanation_drops[
                            batch_index
                        ].cpu()
                    ),
                    "relative_explanation_drop": float(
                        relative_explanation_drops[
                            batch_index
                        ].cpu()
                    ),
                    "mean_compact_random_drop": float(
                        mean_compact_drops[
                            batch_index
                        ].cpu()
                    ),
                    "compact_random_drop_std": float(
                        std_compact_drops[
                            batch_index
                        ].cpu()
                    ),
                    "relative_compact_random_drop": float(
                        relative_compact_drops[
                            batch_index
                        ].cpu()
                    ),
                    "mean_relocated_explanation_drop": float(
                        mean_relocated_drops[
                            batch_index
                        ].cpu()
                    ),
                    "relocated_explanation_drop_std": float(
                        std_relocated_drops[
                            batch_index
                        ].cpu()
                    ),
                    "relative_relocated_explanation_drop": float(
                        relative_relocated_drops[
                            batch_index
                        ].cpu()
                    ),
                    "advantage_over_compact_random": float(
                        advantage_over_compact[
                            batch_index
                        ].cpu()
                    ),
                    "advantage_over_relocated_explanation": float(
                        advantage_over_relocated[
                            batch_index
                        ].cpu()
                    ),
                    "explanation_beats_compact_random": int(
                        explanation_drops[
                            batch_index
                        ]
                        > mean_compact_drops[
                            batch_index
                        ]
                    ),
                    "explanation_beats_relocated_explanation": int(
                        explanation_drops[
                            batch_index
                        ]
                        > mean_relocated_drops[
                            batch_index
                        ]
                    ),
                }

                compact_values = (
                    compact_drops[
                        batch_index
                    ].cpu().tolist()
                )

                relocated_values = (
                    relocated_drops[
                        batch_index
                    ].cpu().tolist()
                )

                for repeat_index, value in enumerate(
                    compact_values,
                    start=1,
                ):
                    output_row[
                        f"compact_random_drop_{repeat_index}"
                    ] = float(value)

                for repeat_index, value in enumerate(
                    relocated_values,
                    start=1,
                ):
                    output_row[
                        f"relocated_explanation_drop_{repeat_index}"
                    ] = float(value)

                output_rows.append(output_row)

    return pd.DataFrame(output_rows)


# ---------------------------------------------------------------------
# Condition summaries
# ---------------------------------------------------------------------

def summarize_conditions(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    return (
        metrics
        .groupby(
            [
                "source_type",
                "corruption_type",
                "severity",
                "masking_fraction",
            ],
            as_index=False,
        )
        .agg(
            image_count=(
                "image_id",
                "count",
            ),
            mean_original_probability=(
                "original_target_probability",
                "mean",
            ),
            mean_explanation_drop=(
                "explanation_confidence_drop",
                "mean",
            ),
            median_explanation_drop=(
                "explanation_confidence_drop",
                "median",
            ),
            mean_compact_random_drop=(
                "mean_compact_random_drop",
                "mean",
            ),
            mean_relocated_explanation_drop=(
                "mean_relocated_explanation_drop",
                "mean",
            ),
            mean_advantage_over_compact_random=(
                "advantage_over_compact_random",
                "mean",
            ),
            mean_advantage_over_relocated_explanation=(
                "advantage_over_relocated_explanation",
                "mean",
            ),
            explanation_beats_compact_random_rate=(
                "explanation_beats_compact_random",
                "mean",
            ),
            explanation_beats_relocated_rate=(
                "explanation_beats_relocated_explanation",
                "mean",
            ),
            mean_relative_explanation_drop=(
                "relative_explanation_drop",
                "mean",
            ),
            mean_relative_compact_random_drop=(
                "relative_compact_random_drop",
                "mean",
            ),
            mean_relative_relocated_drop=(
                "relative_relocated_explanation_drop",
                "mean",
            ),
        )
        .sort_values(
            [
                "source_type",
                "corruption_type",
                "severity",
                "masking_fraction",
            ]
        )
        .reset_index(drop=True)
    )


def create_summary(
    metrics: pd.DataFrame,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    clean_metrics = metrics[
        metrics["source_type"] == "clean"
    ]

    corrupted_metrics = metrics[
        metrics["source_type"] == "corrupted"
    ]

    fraction_summaries: dict[str, Any] = {}

    for masking_fraction in MASKING_FRACTIONS:
        fraction_key = f"{masking_fraction:.2f}"

        fraction_rows = metrics[
            np.isclose(
                metrics["masking_fraction"],
                masking_fraction,
            )
        ]

        clean_fraction = fraction_rows[
            fraction_rows["source_type"]
            == "clean"
        ]

        corrupted_fraction = fraction_rows[
            fraction_rows["source_type"]
            == "corrupted"
        ]

        fraction_summaries[fraction_key] = {
            "clean_mean_explanation_drop": float(
                clean_fraction[
                    "explanation_confidence_drop"
                ].mean()
            ),
            "clean_mean_compact_random_drop": float(
                clean_fraction[
                    "mean_compact_random_drop"
                ].mean()
            ),
            "clean_mean_relocated_drop": float(
                clean_fraction[
                    "mean_relocated_explanation_drop"
                ].mean()
            ),
            "clean_advantage_over_compact_random": float(
                clean_fraction[
                    "advantage_over_compact_random"
                ].mean()
            ),
            "clean_advantage_over_relocated": float(
                clean_fraction[
                    "advantage_over_relocated_explanation"
                ].mean()
            ),
            "corrupted_mean_explanation_drop": float(
                corrupted_fraction[
                    "explanation_confidence_drop"
                ].mean()
            ),
            "corrupted_mean_compact_random_drop": float(
                corrupted_fraction[
                    "mean_compact_random_drop"
                ].mean()
            ),
            "corrupted_mean_relocated_drop": float(
                corrupted_fraction[
                    "mean_relocated_explanation_drop"
                ].mean()
            ),
            "corrupted_advantage_over_compact_random": float(
                corrupted_fraction[
                    "advantage_over_compact_random"
                ].mean()
            ),
            "corrupted_advantage_over_relocated": float(
                corrupted_fraction[
                    "advantage_over_relocated_explanation"
                ].mean()
            ),
        }

    return {
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "unique_explanations": int(
            metrics["gradcam_index"].nunique()
        ),
        "total_evaluation_rows": int(
            len(metrics)
        ),
        "clean_explanations": int(
            clean_metrics[
                "gradcam_index"
            ].nunique()
        ),
        "corrupted_explanations": int(
            corrupted_metrics[
                "gradcam_index"
            ].nunique()
        ),
        "masking_fractions": (
            MASKING_FRACTIONS
        ),
        "blur_radius": BLUR_RADIUS,
        "random_repeats": RANDOM_REPEATS,
        "replacement_strategy": (
            "Gaussian-blurred image pixels"
        ),
        "compact_random_baseline": (
            "Exact-area compact random region"
        ),
        "relocated_baseline": (
            "Relocated Grad-CAM mask shape"
        ),
        "fraction_summaries": (
            fraction_summaries
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Refined pilot Grad-CAM "
        "faithfulness evaluation"
    )
    print("-" * 76)
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(
        "Masking fractions: "
        f"{MASKING_FRACTIONS}"
    )
    print(
        f"Blur radius: {BLUR_RADIUS}"
    )
    print(
        f"Baseline repeats: {RANDOM_REPEATS}"
    )

    manifest = load_manifest()

    dataset = FaithfulnessDataset(
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

    metrics = evaluate_faithfulness(
        model=model,
        data_loader=data_loader,
        manifest=manifest,
        device=device,
    )

    expected_rows = (
        len(manifest)
        * len(MASKING_FRACTIONS)
    )

    if len(metrics) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} result rows, "
            f"but produced {len(metrics)}."
        )

    condition_metrics = summarize_conditions(
        metrics
    )

    metrics.to_csv(
        DETAILED_OUTPUT_PATH,
        index=False,
    )

    condition_metrics.to_csv(
        CONDITION_OUTPUT_PATH,
        index=False,
    )

    summary = create_summary(
        metrics=metrics,
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

    display_columns = [
        "source_type",
        "corruption_type",
        "severity",
        "masking_fraction",
        "image_count",
        "mean_explanation_drop",
        "mean_compact_random_drop",
        "mean_relocated_explanation_drop",
        "mean_advantage_over_compact_random",
        "mean_advantage_over_relocated_explanation",
        "explanation_beats_compact_random_rate",
        "explanation_beats_relocated_rate",
    ]

    print(
        "\nCondition-level refined "
        "faithfulness results"
    )
    print("-" * 76)

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

    print("\nSaved files")
    print("-" * 76)
    print(DETAILED_OUTPUT_PATH)
    print(CONDITION_OUTPUT_PATH)
    print(SUMMARY_OUTPUT_PATH)


if __name__ == "__main__":
    main()