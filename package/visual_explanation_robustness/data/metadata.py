from __future__ import annotations

from pathlib import Path

import pandas as pd

from visual_explanation_robustness.utils.paths import (
    CUB_ROOT,
    METADATA_PATH,
)


def load_raw_metadata() -> pd.DataFrame:
    images = pd.read_csv(
        CUB_ROOT / "images.txt",
        sep=r"\s+",
        names=["image_id", "image_path"],
    )

    labels = pd.read_csv(
        CUB_ROOT / "image_class_labels.txt",
        sep=r"\s+",
        names=["image_id", "class_id"],
    )

    split = pd.read_csv(
        CUB_ROOT / "train_test_split.txt",
        sep=r"\s+",
        names=["image_id", "is_train"],
    )

    boxes = pd.read_csv(
        CUB_ROOT / "bounding_boxes.txt",
        sep=r"\s+",
        names=[
            "image_id",
            "bbox_x",
            "bbox_y",
            "bbox_width",
            "bbox_height",
        ],
    )

    metadata = (
        images
        .merge(labels, on="image_id")
        .merge(split, on="image_id")
        .merge(boxes, on="image_id")
        .sort_values("image_id")
        .reset_index(drop=True)
    )

    metadata["label"] = (
        metadata["class_id"] - 1
    )

    metadata["class_name"] = (
        metadata["image_path"]
        .map(lambda value: Path(value).parent.name)
    )

    return metadata


def load_metadata(
    metadata_path: str | Path | None = None,
) -> pd.DataFrame:
    path = (
        METADATA_PATH
        if metadata_path is None
        else Path(metadata_path)
    )

    if not path.exists():
        metadata = load_raw_metadata()

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata.to_csv(
            path,
            index=False,
        )

        return metadata

    metadata = pd.read_csv(path)

    required_columns = {
        "image_id",
        "image_path",
        "class_id",
        "is_train",
    }

    missing_columns = (
        required_columns
        - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            "Metadata is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if "label" not in metadata.columns:
        metadata["label"] = (
            metadata["class_id"] - 1
        )

    if "class_name" not in metadata.columns:
        metadata["class_name"] = (
            metadata["image_path"]
            .map(
                lambda value: Path(value).parent.name
            )
        )

    rename_map = {
        "x": "bbox_x",
        "y": "bbox_y",
        "width": "bbox_width",
        "height": "bbox_height",
    }

    metadata = metadata.rename(
        columns={
            old: new
            for old, new in rename_map.items()
            if old in metadata.columns
            and new not in metadata.columns
        }
    )

    return metadata


def load_class_names() -> dict[int, str]:
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