from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CUB_ROOT = PROJECT_ROOT / "data" / "raw" / "CUB_200_2011"
IMAGE_ROOT = CUB_ROOT / "images"
SEGMENTATION_ROOT = PROJECT_ROOT / "data" / "raw" / "segmentations"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "cub_metadata.csv"


class CUBDataset(Dataset):
    """
    PyTorch dataset for CUB-200-2011 with segmentation masks
    and bounding-box annotations.
    """

    def __init__(
        self,
        split: str = "train",
        image_size: tuple[int, int] = (224, 224),
        normalize: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        split:
            Either "train", "test", or "all".

        image_size:
            Output image size as (height, width).

        normalize:
            Whether to apply ImageNet normalization.
        """

        if split not in {"train", "test", "all"}:
            raise ValueError(
                "split must be 'train', 'test', or 'all'."
            )

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {METADATA_PATH}\n"
                "Run src/01_verify_dataset.py first."
            )

        self.split = split
        self.image_size = image_size
        self.normalize = normalize

        metadata = pd.read_csv(METADATA_PATH)

        if split == "train":
            metadata = metadata[metadata["is_train"] == 1]
        elif split == "test":
            metadata = metadata[metadata["is_train"] == 0]

        self.metadata = metadata.reset_index(drop=True)

        self.class_names = self._load_class_names()

    def _load_class_names(self) -> dict[int, str]:
        classes = pd.read_csv(
            CUB_ROOT / "classes.txt",
            sep=r"\s+",
            names=["class_id", "class_name"],
        )

        return dict(
            zip(
                classes["class_id"],
                classes["class_name"],
            )
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.metadata.iloc[index]

        image_id = int(row["image_id"])
        class_id = int(row["class_id"])

        # Convert labels from 1-200 to 0-199 for PyTorch.
        label = class_id - 1

        relative_image_path = Path(row["image_path"])
        image_path = IMAGE_ROOT / relative_image_path

        relative_mask_path = relative_image_path.with_suffix(".png")
        mask_path = SEGMENTATION_ROOT / relative_mask_path

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        original_width, original_height = image.size

        x = float(row["x"])
        y = float(row["y"])
        width = float(row["width"])
        height = float(row["height"])

        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(float(original_width), x + width)
        y2 = min(float(original_height), y + height)

        output_height, output_width = self.image_size

        scale_x = output_width / original_width
        scale_y = output_height / original_height

        resized_box = torch.tensor(
            [
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y,
            ],
            dtype=torch.float32,
        )

        image = TF.resize(
            image,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        mask = TF.resize(
            mask,
            self.image_size,
            interpolation=InterpolationMode.NEAREST,
        )

        image_tensor = TF.to_tensor(image)

        mask_tensor = TF.pil_to_tensor(mask).float() / 255.0

        # Convert the segmentation mask into a binary mask.
        mask_tensor = (mask_tensor > 0.5).float()

        if self.normalize:
            image_tensor = TF.normalize(
                image_tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

        sample = {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "class_name": self.class_names[class_id],
            "mask": mask_tensor,
            "bbox": resized_box,
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "original_size": torch.tensor(
                [original_height, original_width],
                dtype=torch.long,
            ),
        }

        return sample


def test_dataset() -> None:
    print("Loading training dataset...")

    train_dataset = CUBDataset(
        split="train",
        image_size=(224, 224),
        normalize=True,
    )

    print("Loading test dataset...")

    test_dataset = CUBDataset(
        split="test",
        image_size=(224, 224),
        normalize=True,
    )

    print("\nDataset summary")
    print("-" * 50)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.class_names)}")

    train_sample = train_dataset[0]
    test_sample = test_dataset[0]

    print("\nFirst training sample")
    print("-" * 50)
    print(f"Image ID: {train_sample['image_id'].item()}")
    print(f"Class ID: {train_sample['class_id'].item()}")
    print(f"Zero-based label: {train_sample['label'].item()}")
    print(f"Class name: {train_sample['class_name']}")
    print(f"Image shape: {tuple(train_sample['image'].shape)}")
    print(f"Mask shape: {tuple(train_sample['mask'].shape)}")
    print(f"Bounding box: {train_sample['bbox'].tolist()}")
    print(f"Mask values: {torch.unique(train_sample['mask'])}")
    print(f"Image path: {train_sample['image_path']}")
    print(f"Mask path: {train_sample['mask_path']}")

    print("\nFirst test sample")
    print("-" * 50)
    print(f"Image ID: {test_sample['image_id'].item()}")
    print(f"Class ID: {test_sample['class_id'].item()}")
    print(f"Zero-based label: {test_sample['label'].item()}")
    print(f"Class name: {test_sample['class_name']}")
    print(f"Image shape: {tuple(test_sample['image'].shape)}")
    print(f"Mask shape: {tuple(test_sample['mask'].shape)}")
    print(f"Bounding box: {test_sample['bbox'].tolist()}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(train_loader))

    print("\nDataLoader batch")
    print("-" * 50)
    print(f"Image batch shape: {tuple(batch['image'].shape)}")
    print(f"Label batch shape: {tuple(batch['label'].shape)}")
    print(f"Mask batch shape: {tuple(batch['mask'].shape)}")
    print(f"Bounding-box batch shape: {tuple(batch['bbox'].shape)}")

    assert len(train_dataset) == 5994
    assert len(test_dataset) == 5794

    assert train_sample["image"].shape == (3, 224, 224)
    assert train_sample["mask"].shape == (1, 224, 224)
    assert train_sample["bbox"].shape == (4,)

    assert train_sample["label"].min() >= 0
    assert train_sample["label"].max() <= 199

    assert set(torch.unique(train_sample["mask"]).tolist()).issubset(
        {0.0, 1.0}
    )

    print("\nDataset loader verification completed successfully.")


if __name__ == "__main__":
    test_dataset()