from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_train_validation_test_splits(
    metadata: pd.DataFrame,
    validation_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    official_train = metadata[
        metadata["is_train"] == 1
    ].copy()

    official_test = metadata[
        metadata["is_train"] == 0
    ].copy()

    train_indices, validation_indices = (
        train_test_split(
            np.arange(
                len(official_train)
            ),
            test_size=validation_fraction,
            random_state=seed,
            shuffle=True,
            stratify=official_train[
                "class_id"
            ],
        )
    )

    train_metadata = (
        official_train.iloc[
            train_indices
        ]
        .sort_values("image_id")
        .reset_index(drop=True)
    )

    validation_metadata = (
        official_train.iloc[
            validation_indices
        ]
        .sort_values("image_id")
        .reset_index(drop=True)
    )

    test_metadata = (
        official_test
        .sort_values("image_id")
        .reset_index(drop=True)
    )

    return (
        train_metadata,
        validation_metadata,
        test_metadata,
    )


def validate_splits(
    train_metadata: pd.DataFrame,
    validation_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
) -> None:
    train_ids = set(
        train_metadata["image_id"]
    )

    validation_ids = set(
        validation_metadata["image_id"]
    )

    test_ids = set(
        test_metadata["image_id"]
    )

    if train_ids & validation_ids:
        raise ValueError(
            "Train and validation sets overlap."
        )

    if train_ids & test_ids:
        raise ValueError(
            "Train and test sets overlap."
        )

    if validation_ids & test_ids:
        raise ValueError(
            "Validation and test sets overlap."
        )

    if len(train_metadata) != 4795:
        raise ValueError(
            "Expected 4795 training samples, "
            f"found {len(train_metadata)}."
        )

    if len(validation_metadata) != 1199:
        raise ValueError(
            "Expected 1199 validation samples, "
            f"found {len(validation_metadata)}."
        )

    if len(test_metadata) != 5794:
        raise ValueError(
            "Expected 5794 test samples, "
            f"found {len(test_metadata)}."
        )