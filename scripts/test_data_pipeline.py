from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "package"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PACKAGE_ROOT),
    )


from visual_explanation_robustness.data.cub_dataset import (
    CUBClassificationDataset,
    CUBExplanationDataset,
)
from visual_explanation_robustness.data.metadata import (
    load_metadata,
)
from visual_explanation_robustness.data.splits import (
    create_train_validation_test_splits,
    validate_splits,
)
from visual_explanation_robustness.data.transforms import (
    create_evaluation_transform,
    create_train_transform,
)


def main() -> None:
    metadata = load_metadata()

    (
        train_metadata,
        validation_metadata,
        test_metadata,
    ) = create_train_validation_test_splits(
        metadata=metadata,
        validation_fraction=0.20,
        seed=42,
    )

    validate_splits(
        train_metadata=train_metadata,
        validation_metadata=(
            validation_metadata
        ),
        test_metadata=test_metadata,
    )

    train_dataset = (
        CUBClassificationDataset(
            metadata=train_metadata,
            transform=(
                create_train_transform(
                    image_size=224
                )
            ),
        )
    )

    validation_dataset = (
        CUBClassificationDataset(
            metadata=validation_metadata,
            transform=(
                create_evaluation_transform(
                    image_size=224
                )
            ),
        )
    )

    explanation_dataset = (
        CUBExplanationDataset(
            metadata=test_metadata,
            image_transform=(
                create_evaluation_transform(
                    image_size=224
                )
            ),
            image_size=224,
        )
    )

    train_sample = train_dataset[0]

    explanation_sample = (
        explanation_dataset[0]
    )

    loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("Reusable data-pipeline test")
    print("-" * 60)
    print(
        f"Metadata rows: {len(metadata)}"
    )
    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )
    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )
    print(
        f"Test samples: "
        f"{len(explanation_dataset)}"
    )

    print("\nClassification sample")
    print("-" * 60)
    print(
        f"Image shape: "
        f"{tuple(train_sample['image'].shape)}"
    )
    print(
        f"Label: "
        f"{train_sample['label'].item()}"
    )
    print(
        f"Class: "
        f"{train_sample['class_name']}"
    )

    print("\nExplanation sample")
    print("-" * 60)
    print(
        f"Image shape: "
        f"{tuple(explanation_sample['image'].shape)}"
    )
    print(
        f"Mask shape: "
        f"{tuple(explanation_sample['mask'].shape)}"
    )
    print(
        f"Bounding box: "
        f"{explanation_sample['bbox'].tolist()}"
    )
    print(
        "Mask values: "
        f"{torch.unique(explanation_sample['mask'])}"
    )

    print("\nDataLoader batch")
    print("-" * 60)
    print(
        f"Images: "
        f"{tuple(batch['image'].shape)}"
    )
    print(
        f"Labels: "
        f"{tuple(batch['label'].shape)}"
    )

    assert len(metadata) == 11788
    assert len(train_dataset) == 4795
    assert len(validation_dataset) == 1199
    assert len(explanation_dataset) == 5794

    assert (
        train_sample["image"].shape
        == (3, 224, 224)
    )

    assert (
        explanation_sample["mask"].shape
        == (1, 224, 224)
    )

    assert (
        explanation_sample["bbox"].shape
        == (4,)
    )

    print(
        "\nReusable data pipeline "
        "verified successfully."
    )


if __name__ == "__main__":
    main()