from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CUB_ROOT = PROJECT_ROOT / "data" / "raw" / "CUB_200_2011"
IMAGE_ROOT = CUB_ROOT / "images"
SEGMENTATION_ROOT = PROJECT_ROOT / "data" / "raw" / "segmentations"


def load_metadata() -> pd.DataFrame:
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
        names=["image_id", "x", "y", "width", "height"],
    )

    metadata = (
        images
        .merge(labels, on="image_id")
        .merge(split, on="image_id")
        .merge(boxes, on="image_id")
    )

    return metadata


def check_sample(row: pd.Series) -> dict:
    image_path = IMAGE_ROOT / row["image_path"]

    relative_mask_path = Path(row["image_path"]).with_suffix(".png")
    mask_path = SEGMENTATION_ROOT / relative_mask_path

    result = {
        "image_id": row["image_id"],
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "image_exists": image_path.exists(),
        "mask_exists": mask_path.exists(),
        "image_width": None,
        "image_height": None,
        "mask_width": None,
        "mask_height": None,
        "same_dimensions": False,
        "bounding_box_valid": False,
        "bbox_x1": None,
        "bbox_y1": None,
        "bbox_x2": None,
        "bbox_y2": None,
    }

    if image_path.exists():
        with Image.open(image_path) as image:
            image_width, image_height = image.size

            result["image_width"] = image_width
            result["image_height"] = image_height

            x = float(row["x"])
            y = float(row["y"])
            width = float(row["width"])
            height = float(row["height"])

            # Convert and clamp bounding-box coordinates
            # so they stay inside the image boundaries.
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(image_width, int(round(x + width)))
            y2 = min(image_height, int(round(y + height)))

            result["bbox_x1"] = x1
            result["bbox_y1"] = y1
            result["bbox_x2"] = x2
            result["bbox_y2"] = y2

            result["bounding_box_valid"] = (
                width > 0
                and height > 0
                and x2 > x1
                and y2 > y1
            )

    if mask_path.exists():
        with Image.open(mask_path) as mask:
            mask_width, mask_height = mask.size

            result["mask_width"] = mask_width
            result["mask_height"] = mask_height

    if result["image_exists"] and result["mask_exists"]:
        result["same_dimensions"] = (
            result["image_width"] == result["mask_width"]
            and result["image_height"] == result["mask_height"]
        )

    return result


def main() -> None:
    required_files = [
        CUB_ROOT / "images.txt",
        CUB_ROOT / "image_class_labels.txt",
        CUB_ROOT / "train_test_split.txt",
        CUB_ROOT / "bounding_boxes.txt",
        CUB_ROOT / "classes.txt",
    ]

    missing_files = [
        path for path in required_files
        if not path.exists()
    ]

    if missing_files:
        print("Missing required metadata files:")

        for path in missing_files:
            print(f"  - {path}")

        raise FileNotFoundError(
            "CUB metadata files are missing."
        )

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_ROOT}"
        )

    if not SEGMENTATION_ROOT.exists():
        raise FileNotFoundError(
            f"Segmentation directory not found: {SEGMENTATION_ROOT}"
        )

    metadata = load_metadata()

    print(f"Total metadata records: {len(metadata)}")
    print(
        f"Training images: "
        f"{(metadata['is_train'] == 1).sum()}"
    )
    print(
        f"Test images: "
        f"{(metadata['is_train'] == 0).sum()}"
    )
    print(
        f"Number of classes: "
        f"{metadata['class_id'].nunique()}"
    )

    verification_records = [
        check_sample(row)
        for _, row in metadata.iterrows()
    ]

    verification = pd.DataFrame(verification_records)

    print("\nDataset verification")
    print("-" * 50)

    print(
        f"Images found: "
        f"{verification['image_exists'].sum()}"
    )

    print(
        f"Masks found: "
        f"{verification['mask_exists'].sum()}"
    )

    print(
        "Matching image-mask dimensions: "
        f"{verification['same_dimensions'].sum()}"
    )

    print(
        "Valid bounding boxes: "
        f"{verification['bounding_box_valid'].sum()}"
    )

    problem_rows = verification[
        ~verification[
            [
                "image_exists",
                "mask_exists",
                "same_dimensions",
                "bounding_box_valid",
            ]
        ].all(axis=1)
    ]

    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(
        output_dir / "cub_metadata.csv",
        index=False,
    )

    problem_rows.to_csv(
        output_dir / "dataset_verification_problems.csv",
        index=False,
    )

    if problem_rows.empty:
        print("\nVerification completed successfully.")
        print(
            "All images, masks, and bounding boxes are valid."
        )
    else:
        print(
            f"\nFound {len(problem_rows)} problematic records."
        )
        print(
            "See "
            "data/processed/dataset_verification_problems.csv"
        )


if __name__ == "__main__":
    main()