from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    macro_f1: float


@dataclass
class TrainingResult:
    best_epoch: int
    best_validation_macro_f1: float
    epochs_completed: int
    training_seconds: float
    best_checkpoint_path: str
    last_checkpoint_path: str
    history_path: str


def calculate_metrics(
    labels: list[int],
    predictions: list[int],
) -> tuple[float, float]:
    accuracy = accuracy_score(
        labels,
        predictions,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return float(accuracy), float(macro_f1)


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_mixed_precision: bool = True,
    description: str = "",
    max_batches: int | None = None,
) -> EpochMetrics:
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    processed_samples = 0

    all_labels: list[int] = []
    all_predictions: list[int] = []

    progress_bar = tqdm(
        data_loader,
        desc=description,
        leave=False,
    )

    for batch_index, batch in enumerate(
        progress_bar
    ):
        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        amp_enabled = (
            use_mixed_precision
            and device.type == "cuda"
        )

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(
                    logits,
                    labels,
                )

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        predictions = logits.argmax(
            dim=1
        )

        current_batch_size = images.size(0)

        running_loss += (
            loss.item()
            * current_batch_size
        )

        processed_samples += (
            current_batch_size
        )

        all_labels.extend(
            labels.detach().cpu().tolist()
        )

        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    if processed_samples == 0:
        raise RuntimeError(
            "No samples were processed in the epoch."
        )

    average_loss = (
        running_loss
        / processed_samples
    )

    accuracy, macro_f1 = calculate_metrics(
        labels=all_labels,
        predictions=all_predictions,
    )

    return EpochMetrics(
        loss=float(average_loss),
        accuracy=accuracy,
        macro_f1=macro_f1,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    validation_metrics: EpochMetrics,
    model_name: str,
    training_config: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_name": model_name,
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "scheduler_state_dict": (
            scheduler.state_dict()
        ),
        "validation_metrics": asdict(
            validation_metrics
        ),
        "training_config": (
            training_config
        ),
    }

    torch.save(
        checkpoint,
        path,
    )


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    output_directory: Path,
    training_config: dict[str, Any],
    maximum_epochs: int,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    early_stopping_patience: int,
    scheduler_patience: int,
    scheduler_factor: float,
    use_mixed_precision: bool,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
) -> TrainingResult:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint_path = (
        output_directory
        / "best_model.pth"
    )

    last_checkpoint_path = (
        output_directory
        / "last_model.pth"
    )

    history_path = (
        output_directory
        / "training_history.csv"
    )

    summary_path = (
        output_directory
        / "training_summary.json"
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing
    )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=1e-7,
    )

    scaler: torch.amp.GradScaler | None = None

    if (
        use_mixed_precision
        and device.type == "cuda"
    ):
        scaler = torch.amp.GradScaler(
            "cuda"
        )

    history: list[dict[str, Any]] = []

    best_validation_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    training_start_time = time.time()

    for epoch in range(
        1,
        maximum_epochs + 1,
    ):
        epoch_start_time = time.time()

        print(
            f"\nEpoch {epoch}/{maximum_epochs}"
        )
        print("-" * 70)

        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_mixed_precision=(
                use_mixed_precision
            ),
            description="Training",
            max_batches=max_train_batches,
        )

        validation_metrics = run_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            use_mixed_precision=(
                use_mixed_precision
            ),
            description="Validation",
            max_batches=(
                max_validation_batches
            ),
        )

        scheduler.step(
            validation_metrics.macro_f1
        )

        learning_rate_now = float(
            optimizer.param_groups[0]["lr"]
        )

        epoch_seconds = (
            time.time()
            - epoch_start_time
        )

        epoch_record = {
            "epoch": epoch,
            "learning_rate": (
                learning_rate_now
            ),
            "train_loss": (
                train_metrics.loss
            ),
            "train_accuracy": (
                train_metrics.accuracy
            ),
            "train_macro_f1": (
                train_metrics.macro_f1
            ),
            "validation_loss": (
                validation_metrics.loss
            ),
            "validation_accuracy": (
                validation_metrics.accuracy
            ),
            "validation_macro_f1": (
                validation_metrics.macro_f1
            ),
            "epoch_seconds": (
                epoch_seconds
            ),
        }

        history.append(epoch_record)

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

        print(
            f"Train loss: "
            f"{train_metrics.loss:.4f}"
        )
        print(
            f"Train accuracy: "
            f"{train_metrics.accuracy:.4f}"
        )
        print(
            f"Train macro-F1: "
            f"{train_metrics.macro_f1:.4f}"
        )
        print(
            f"Validation loss: "
            f"{validation_metrics.loss:.4f}"
        )
        print(
            f"Validation accuracy: "
            f"{validation_metrics.accuracy:.4f}"
        )
        print(
            f"Validation macro-F1: "
            f"{validation_metrics.macro_f1:.4f}"
        )
        print(
            f"Learning rate: "
            f"{learning_rate_now:.8f}"
        )
        print(
            f"Epoch duration: "
            f"{epoch_seconds / 60:.2f} minutes"
        )

        save_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            validation_metrics=(
                validation_metrics
            ),
            model_name=model_name,
            training_config=(
                training_config
            ),
        )

        if (
            validation_metrics.macro_f1
            > best_validation_macro_f1
        ):
            best_validation_macro_f1 = (
                validation_metrics.macro_f1
            )

            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                validation_metrics=(
                    validation_metrics
                ),
                model_name=model_name,
                training_config=(
                    training_config
                ),
            )

            print(
                "Best checkpoint updated: "
                f"validation macro-F1 = "
                f"{best_validation_macro_f1:.4f}"
            )
        else:
            epochs_without_improvement += 1

            print(
                "No validation improvement for "
                f"{epochs_without_improvement} "
                "epoch(s)."
            )

        if (
            epochs_without_improvement
            >= early_stopping_patience
        ):
            print(
                "\nEarly stopping activated."
            )
            break

    training_seconds = (
        time.time()
        - training_start_time
    )

    result = TrainingResult(
        best_epoch=best_epoch,
        best_validation_macro_f1=(
            best_validation_macro_f1
        ),
        epochs_completed=len(history),
        training_seconds=(
            training_seconds
        ),
        best_checkpoint_path=str(
            best_checkpoint_path
        ),
        last_checkpoint_path=str(
            last_checkpoint_path
        ),
        history_path=str(
            history_path
        ),
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(result),
            file,
            indent=4,
        )

    print("\nTraining completed")
    print("-" * 70)
    print(
        f"Best epoch: {best_epoch}"
    )
    print(
        "Best validation macro-F1: "
        f"{best_validation_macro_f1:.4f}"
    )
    print(
        "Total duration: "
        f"{training_seconds / 60:.2f} minutes"
    )
    print(
        f"Best checkpoint: "
        f"{best_checkpoint_path}"
    )

    return result