from __future__ import annotations

import csv
import json
import random
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

cub_module = import_module("02_cub_dataset")
CUBDataset = cub_module.CUBDataset

CUB_ROOT = PROJECT_ROOT / "data" / "raw" / "CUB_200_2011"
IMAGE_ROOT = CUB_ROOT / "images"

MODEL_DIR = PROJECT_ROOT / "models" / "resnet50"
RESULT_DIR = PROJECT_ROOT / "results" / "resnet50"
PREDICTION_DIR = PROJECT_ROOT / "results" / "predictions"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CONFIG: dict[str, Any] = {
    "seed": 42,
    "image_size": 224,
    "number_of_classes": 200,
    "validation_fraction": 0.20,
    "batch_size": 32,
    "number_of_workers": 2,
    "epochs": 20,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "early_stopping_patience": 5,
    "scheduler_patience": 2,
    "scheduler_factor": 0.5,
    "minimum_learning_rate": 1e-7,
    "use_pretrained_weights": True,
}


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # These settings improve reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Training-only dataset
# ---------------------------------------------------------------------

class CUBTrainingDataset(Dataset):
    """
    Classification-only CUB dataset with image augmentation.

    Segmentation masks are not required during classifier training.
    The original CUBDataset class remains responsible for aligned
    image-mask loading during explanation evaluation.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        transform: transforms.Compose,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.metadata.iloc[index]

        image_path = IMAGE_ROOT / row["image_path"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        class_id = int(row["class_id"])
        label = class_id - 1

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "image_id": torch.tensor(
                int(row["image_id"]),
                dtype=torch.long,
            ),
            "image_path": str(image_path),
        }


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def get_transforms(
    image_size: int,
) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.70, 1.0),
                ratio=(0.80, 1.25),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.20,
                saturation=0.20,
                hue=0.05,
            ),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, evaluation_transform


def create_datasets() -> tuple[
    Dataset,
    Dataset,
    Dataset,
    pd.DataFrame,
    pd.DataFrame,
]:
    metadata_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "cub_metadata.csv"
    )

    metadata = pd.read_csv(metadata_path)

    official_train = metadata[
        metadata["is_train"] == 1
    ].copy()

    official_test = metadata[
        metadata["is_train"] == 0
    ].copy()

    train_indices, validation_indices = train_test_split(
        np.arange(len(official_train)),
        test_size=CONFIG["validation_fraction"],
        random_state=CONFIG["seed"],
        shuffle=True,
        stratify=official_train["class_id"],
    )

    train_metadata = official_train.iloc[
        train_indices
    ].reset_index(drop=True)

    validation_metadata = official_train.iloc[
        validation_indices
    ].reset_index(drop=True)

    test_metadata = official_test.reset_index(drop=True)

    train_transform, evaluation_transform = get_transforms(
        CONFIG["image_size"]
    )

    train_dataset = CUBTrainingDataset(
        metadata=train_metadata,
        transform=train_transform,
    )

    validation_dataset = CUBTrainingDataset(
        metadata=validation_metadata,
        transform=evaluation_transform,
    )

    test_dataset = CUBTrainingDataset(
        metadata=test_metadata,
        transform=evaluation_transform,
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_metadata,
        validation_metadata,
    )


def create_data_loaders() -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    pd.DataFrame,
    pd.DataFrame,
]:
    (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_metadata,
        validation_metadata,
    ) = create_datasets()

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["number_of_workers"],
        pin_memory=pin_memory,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["number_of_workers"],
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["number_of_workers"],
        pin_memory=pin_memory,
        drop_last=False,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        train_metadata,
        validation_metadata,
    )


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def create_model() -> nn.Module:
    if CONFIG["use_pretrained_weights"]:
        weights = ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = resnet50(weights=weights)

    input_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(
            input_features,
            CONFIG["number_of_classes"],
        ),
    )

    return model


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    labels: list[int],
    predictions: list[int],
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


# ---------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    description: str = "",
) -> dict[str, Any]:
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    progress_bar = tqdm(
        data_loader,
        desc=description,
        leave=False,
    )

    for batch in progress_bar:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

        predictions = logits.argmax(dim=1)

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size

        all_labels.extend(
            labels.detach().cpu().tolist()
        )

        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = running_loss / len(data_loader.dataset)

    metrics = calculate_metrics(
        labels=all_labels,
        predictions=all_predictions,
    )

    return {
        "loss": average_loss,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }


# ---------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_metrics: dict[str, Any],
    path: Path,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": validation_metrics,
        "config": CONFIG,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return checkpoint


# ---------------------------------------------------------------------
# Prediction export
# ---------------------------------------------------------------------

@torch.no_grad()
def export_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    output_path: Path,
) -> dict[str, float]:
    model.eval()

    rows: list[dict[str, Any]] = []
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for batch in tqdm(
        data_loader,
        desc="Final test evaluation",
    ):
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)

        confidence, predictions = probabilities.max(dim=1)

        image_ids = batch["image_id"].cpu().tolist()
        image_paths = batch["image_path"]

        labels_cpu = labels.cpu().tolist()
        predictions_cpu = predictions.cpu().tolist()
        confidence_cpu = confidence.cpu().tolist()

        all_labels.extend(labels_cpu)
        all_predictions.extend(predictions_cpu)

        for (
            image_id,
            image_path,
            true_label,
            predicted_label,
            predicted_confidence,
        ) in zip(
            image_ids,
            image_paths,
            labels_cpu,
            predictions_cpu,
            confidence_cpu,
        ):
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "true_label": true_label,
                    "true_class_id": true_label + 1,
                    "predicted_label": predicted_label,
                    "predicted_class_id": predicted_label + 1,
                    "confidence": predicted_confidence,
                    "correct": int(
                        true_label == predicted_label
                    ),
                }
            )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
    )

    return calculate_metrics(
        labels=all_labels,
        predictions=all_predictions,
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_model() -> None:
    set_seed(CONFIG["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Training configuration")
    print("-" * 60)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Maximum epochs: {CONFIG['epochs']}")
    print(
        f"Learning rate: "
        f"{CONFIG['learning_rate']}"
    )

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
    else:
        print(
            "WARNING: CUDA GPU was not detected. "
            "ResNet-50 training on CPU will be slow."
        )

    (
        train_loader,
        validation_loader,
        test_loader,
        train_metadata,
        validation_metadata,
    ) = create_data_loaders()

    print("\nDataset split")
    print("-" * 60)
    print(
        f"Training samples: "
        f"{len(train_loader.dataset)}"
    )
    print(
        f"Validation samples: "
        f"{len(validation_loader.dataset)}"
    )
    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    train_metadata.to_csv(
        RESULT_DIR / "train_split.csv",
        index=False,
    )

    validation_metadata.to_csv(
        RESULT_DIR / "validation_split.csv",
        index=False,
    )

    with open(
        RESULT_DIR / "training_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            CONFIG,
            file,
            indent=4,
        )

    model = create_model().to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=CONFIG["label_smoothing"]
    )

    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=CONFIG["scheduler_factor"],
        patience=CONFIG["scheduler_patience"],
        min_lr=CONFIG["minimum_learning_rate"],
    )

    scaler = None

    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    best_checkpoint_path = (
        MODEL_DIR / "best_resnet50_cub.pth"
    )

    last_checkpoint_path = (
        MODEL_DIR / "last_resnet50_cub.pth"
    )

    history: list[dict[str, Any]] = []

    best_validation_f1 = -1.0
    epochs_without_improvement = 0

    training_start_time = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        epoch_start_time = time.time()

        print(
            f"\nEpoch {epoch}/{CONFIG['epochs']}"
        )
        print("-" * 60)

        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            description="Training",
        )

        validation_metrics = run_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            description="Validation",
        )

        scheduler.step(
            validation_metrics["macro_f1"]
        )

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        epoch_seconds = time.time() - epoch_start_time

        epoch_record = {
            "epoch": epoch,
            "learning_rate": current_learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "epoch_seconds": epoch_seconds,
        }

        history.append(epoch_record)

        pd.DataFrame(history).to_csv(
            RESULT_DIR / "training_history.csv",
            index=False,
        )

        print(
            f"Train loss: "
            f"{train_metrics['loss']:.4f}"
        )
        print(
            f"Train accuracy: "
            f"{train_metrics['accuracy']:.4f}"
        )
        print(
            f"Train macro-F1: "
            f"{train_metrics['macro_f1']:.4f}"
        )
        print(
            f"Validation loss: "
            f"{validation_metrics['loss']:.4f}"
        )
        print(
            f"Validation accuracy: "
            f"{validation_metrics['accuracy']:.4f}"
        )
        print(
            f"Validation macro-F1: "
            f"{validation_metrics['macro_f1']:.4f}"
        )
        print(
            f"Learning rate: "
            f"{current_learning_rate:.8f}"
        )
        print(
            f"Epoch duration: "
            f"{epoch_seconds / 60:.2f} minutes"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            validation_metrics=validation_metrics,
            path=last_checkpoint_path,
        )

        if (
            validation_metrics["macro_f1"]
            > best_validation_f1
        ):
            best_validation_f1 = (
                validation_metrics["macro_f1"]
            )

            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_metrics=validation_metrics,
                path=best_checkpoint_path,
            )

            print(
                "Best checkpoint updated: "
                f"validation macro-F1 = "
                f"{best_validation_f1:.4f}"
            )
        else:
            epochs_without_improvement += 1

            print(
                "No validation improvement for "
                f"{epochs_without_improvement} epoch(s)."
            )

        if (
            epochs_without_improvement
            >= CONFIG["early_stopping_patience"]
        ):
            print("\nEarly stopping activated.")
            break

    total_training_seconds = (
        time.time() - training_start_time
    )

    print("\nTraining completed")
    print("-" * 60)
    print(
        f"Total duration: "
        f"{total_training_seconds / 60:.2f} minutes"
    )
    print(
        f"Best validation macro-F1: "
        f"{best_validation_f1:.4f}"
    )

    print("\nLoading the best checkpoint...")

    best_checkpoint = load_checkpoint(
        model=model,
        path=best_checkpoint_path,
        device=device,
    )

    print(
        f"Best epoch: "
        f"{best_checkpoint['epoch']}"
    )

    prediction_path = (
        PREDICTION_DIR
        / "resnet50_clean_test_predictions.csv"
    )

    test_metrics = export_predictions(
        model=model,
        data_loader=test_loader,
        device=device,
        output_path=prediction_path,
    )

    print("\nFinal clean test results")
    print("-" * 60)
    print(
        f"Test accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )
    print(
        f"Test macro-F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )
    print(
        f"Predictions saved to: "
        f"{prediction_path}"
    )

    summary = {
        "best_epoch": best_checkpoint["epoch"],
        "best_validation_macro_f1": (
            best_validation_f1
        ),
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "training_minutes": (
            total_training_seconds / 60
        ),
        "best_checkpoint": str(
            best_checkpoint_path
        ),
        "prediction_file": str(prediction_path),
    }

    with open(
        RESULT_DIR / "final_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )


if __name__ == "__main__":
    train_model()