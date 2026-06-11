# Dataset

This project uses the Caltech-UCSD Birds-200-2011 dataset and its
corresponding segmentation annotations.

The dataset is not included in this repository because of its size
and distribution conditions.

## Expected directory structure

```text
data/raw/
├── CUB_200_2011/
│   ├── images/
│   ├── parts/
│   ├── attributes/
│   ├── bounding_boxes.txt
│   ├── image_class_labels.txt
│   ├── images.txt
│   └── train_test_split.txt
│
└── segmentations/