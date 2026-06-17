from __future__ import annotations

from torchvision import transforms


IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


def create_train_transform(
    image_size: int = 224,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.70, 1.0),
                ratio=(0.80, 1.25),
            ),
            transforms.RandomHorizontalFlip(
                p=0.5
            ),
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.20,
                saturation=0.20,
                hue=0.05,
            ),
            transforms.RandomRotation(
                degrees=10
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def create_evaluation_transform(
    image_size: int = 224,
) -> transforms.Compose:
    resize_size = int(
        round(
            image_size * 256 / 224
        )
    )

    return transforms.Compose(
        [
            transforms.Resize(
                resize_size
            ),
            transforms.CenterCrop(
                image_size
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def create_unnormalized_evaluation_transform(
    image_size: int = 224,
) -> transforms.Compose:
    resize_size = int(
        round(
            image_size * 256 / 224
        )
    )

    return transforms.Compose(
        [
            transforms.Resize(
                resize_size
            ),
            transforms.CenterCrop(
                image_size
            ),
            transforms.ToTensor(),
        ]
    )