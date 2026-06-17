from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from visual_explanation_robustness.utils.paths import (
    IMAGE_ROOT,
    SEGMENTATION_ROOT,
)


class CUBClassificationDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        transform: Callable | None = None,
    ) -> None:
        self.metadata = metadata.reset_index(
            drop=True
        )

        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        row = self.metadata.iloc[index]

        relative_path = Path(
            row["image_path"]
        )

        image_path = (
            IMAGE_ROOT / relative_path
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image_tensor = self.transform(
                    image
                )
            else:
                image_tensor = TF.to_tensor(
                    image
                )

        return {
            "image": image_tensor,
            "label": torch.tensor(
                int(row["class_id"]) - 1,
                dtype=torch.long,
            ),
            "class_id": torch.tensor(
                int(row["class_id"]),
                dtype=torch.long,
            ),
            "image_id": torch.tensor(
                int(row["image_id"]),
                dtype=torch.long,
            ),
            "class_name": row[
                "class_name"
            ],
            "relative_image_path": str(
                relative_path
            ),
            "image_path": str(
                image_path
            ),
        }


class CUBExplanationDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        image_transform: Callable,
        image_size: int = 224,
    ) -> None:
        self.metadata = metadata.reset_index(
            drop=True
        )

        self.image_transform = (
            image_transform
        )

        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        row = self.metadata.iloc[index]

        relative_image_path = Path(
            row["image_path"]
        )

        image_path = (
            IMAGE_ROOT
            / relative_image_path
        )

        relative_mask_path = (
            relative_image_path
            .with_suffix(".png")
        )

        mask_path = (
            SEGMENTATION_ROOT
            / relative_mask_path
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found: {mask_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            original_width, original_height = (
                image.size
            )
            image_tensor = (
                self.image_transform(image)
            )

        with Image.open(mask_path) as mask:
            mask = mask.convert("L")

            resize_size = int(
                round(
                    self.image_size
                    * 256
                    / 224
                )
            )

            mask = TF.resize(
                mask,
                resize_size,
                interpolation=(
                    InterpolationMode.NEAREST
                ),
            )

            mask = TF.center_crop(
                mask,
                [
                    self.image_size,
                    self.image_size,
                ],
            )

            mask_tensor = (
                TF.pil_to_tensor(mask)
                .float()
                / 255.0
            )

            mask_tensor = (
                mask_tensor > 0.5
            ).float()

        bbox = self._transform_bbox(
            row=row,
            original_width=original_width,
            original_height=original_height,
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "bbox": bbox,
            "label": torch.tensor(
                int(row["class_id"]) - 1,
                dtype=torch.long,
            ),
            "class_id": torch.tensor(
                int(row["class_id"]),
                dtype=torch.long,
            ),
            "image_id": torch.tensor(
                int(row["image_id"]),
                dtype=torch.long,
            ),
            "class_name": row[
                "class_name"
            ],
            "relative_image_path": str(
                relative_image_path
            ),
            "image_path": str(
                image_path
            ),
            "mask_path": str(
                mask_path
            ),
            "original_size": torch.tensor(
                [
                    original_height,
                    original_width,
                ],
                dtype=torch.long,
            ),
        }

    def _transform_bbox(
        self,
        row: pd.Series,
        original_width: int,
        original_height: int,
    ) -> torch.Tensor:
        x1 = float(row["bbox_x"])
        y1 = float(row["bbox_y"])

        x2 = x1 + float(
            row["bbox_width"]
        )

        y2 = y1 + float(
            row["bbox_height"]
        )

        resize_size = int(
            round(
                self.image_size
                * 256
                / 224
            )
        )

        scale = resize_size / min(
            original_width,
            original_height,
        )

        resized_width = (
            original_width * scale
        )

        resized_height = (
            original_height * scale
        )

        crop_left = max(
            0.0,
            (
                resized_width
                - self.image_size
            )
            / 2.0,
        )

        crop_top = max(
            0.0,
            (
                resized_height
                - self.image_size
            )
            / 2.0,
        )

        transformed_x1 = (
            x1 * scale
            - crop_left
        )

        transformed_y1 = (
            y1 * scale
            - crop_top
        )

        transformed_x2 = (
            x2 * scale
            - crop_left
        )

        transformed_y2 = (
            y2 * scale
            - crop_top
        )

        transformed_x1 = max(
            0.0,
            min(
                transformed_x1,
                float(self.image_size),
            ),
        )

        transformed_y1 = max(
            0.0,
            min(
                transformed_y1,
                float(self.image_size),
            ),
        )

        transformed_x2 = max(
            0.0,
            min(
                transformed_x2,
                float(self.image_size),
            ),
        )

        transformed_y2 = max(
            0.0,
            min(
                transformed_y2,
                float(self.image_size),
            ),
        )

        return torch.tensor(
            [
                transformed_x1,
                transformed_y1,
                transformed_x2,
                transformed_y2,
            ],
            dtype=torch.float32,
        )