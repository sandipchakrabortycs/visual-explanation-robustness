from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from importlib import import_module


cub_module = import_module("02_cub_dataset")
CUBDataset = cub_module.CUBDataset


IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalization for visualization.
    """
    image = image * IMAGENET_STD + IMAGENET_MEAN
    return image.clamp(0, 1)


def visualize_sample(
    sample: dict,
    output_path: Path,
) -> None:
    image = denormalize_image(sample["image"]).permute(1, 2, 0)
    mask = sample["mask"].squeeze(0)

    bbox = sample["bbox"].tolist()
    x1, y1, x2, y2 = bbox

    box_width = x2 - x1
    box_height = y2 - y1

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    figure.suptitle(
        f"{sample['class_name']} | "
        f"Image ID: {sample['image_id'].item()}",
        fontsize=14,
    )

    axes[0].imshow(image)
    axes[0].set_title("Resized image")
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].imshow(
        mask,
        alpha=0.45,
        cmap="Reds",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Segmentation-mask overlay")
    axes[1].axis("off")

    axes[2].imshow(image)

    rectangle = patches.Rectangle(
        (x1, y1),
        box_width,
        box_height,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )

    axes[2].add_patch(rectangle)
    axes[2].set_title("Bounding-box overlay")
    axes[2].axis("off")

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def visualize_random_samples(
    split: str,
    number_of_samples: int = 6,
    seed: int = 42,
) -> None:
    dataset = CUBDataset(
        split=split,
        image_size=(224, 224),
        normalize=True,
    )

    if number_of_samples > len(dataset):
        raise ValueError(
            "number_of_samples cannot exceed dataset size."
        )

    random_generator = random.Random(seed)

    selected_indices = random_generator.sample(
        range(len(dataset)),
        number_of_samples,
    )

    output_directory = (
        PROJECT_ROOT
        / "figures"
        / "dataset_visualization"
        / split
    )

    print(f"\nVisualizing {split} samples")
    print("-" * 50)

    for order, index in enumerate(selected_indices, start=1):
        sample = dataset[index]

        image_id = sample["image_id"].item()

        output_path = (
            output_directory
            / f"{order:02d}_image_{image_id}.png"
        )

        visualize_sample(
            sample=sample,
            output_path=output_path,
        )

        print(
            f"Saved sample {order}: "
            f"{sample['class_name']} -> {output_path}"
        )


def main() -> None:
    visualize_random_samples(
        split="train",
        number_of_samples=6,
        seed=42,
    )

    visualize_random_samples(
        split="test",
        number_of_samples=6,
        seed=84,
    )

    print("\nVisualization completed successfully.")
    print(
        "Check figures/dataset_visualization/"
        "train and test."
    )


if __name__ == "__main__":
    main()