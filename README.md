# Beyond Prediction Robustness: Evaluating the Stability, Localization, and Faithfulness of Visual Explanations under Image Corruption

## Overview

Deep learning models used for image classification may experience changes in both predictions and explanations when input images are affected by corruption. Although prediction robustness has been widely studied, the robustness of visual explanations remains comparatively underexplored.

This project investigates whether visual explanation methods remain stable, correctly localized, and faithful to model predictions when images are exposed to different types and severities of corruption.

The study uses the **Caltech-UCSD Birds-200-2011 (CUB-200-2011)** dataset together with its segmentation annotations. The dataset enables evaluation of both classification performance and the spatial alignment of explanations with the actual object of interest.

## Research Title

**Beyond Prediction Robustness: Evaluating the Stability, Localization, and Faithfulness of Visual Explanations under Image Corruption**

## Research Objectives

The primary objectives of this study are to:

1. Evaluate the effect of image corruption on classification performance.
2. Measure the stability of visual explanations across corruption types and severity levels.
3. Evaluate whether explanations remain localized within the relevant object region.
4. Assess the faithfulness of explanations to model predictions.
5. Examine whether prediction robustness and explanation robustness behave similarly.
6. Compare the robustness characteristics of different visual explanation methods.

## Research Questions

The study is guided by the following research questions:

* **RQ1:** How do different image corruptions affect classification performance?
* **RQ2:** How stable are visual explanations when the input image is corrupted?
* **RQ3:** Do explanations remain localized within the relevant object region under corruption?
* **RQ4:** Does image corruption affect the faithfulness of visual explanations?
* **RQ5:** Is prediction robustness associated with explanation robustness?
* **RQ6:** Do different explanation methods exhibit different levels of robustness?

## Dataset

This project uses the **Caltech-UCSD Birds-200-2011**, commonly known as **CUB-200-2011**.

The dataset contains:

* 200 bird species
* 11,788 images
* Image-level class labels
* Official training and testing splits
* Bounding-box annotations
* Part-location annotations
* Attribute annotations
* Segmentation masks distributed separately

The image files and segmentation masks are not included in this repository because of their size and distribution conditions.

### Expected Dataset Structure

```text
data/
└── raw/
    ├── CUB_200_2011/
    │   ├── images/
    │   ├── parts/
    │   ├── attributes/
    │   ├── bounding_boxes.txt
    │   ├── classes.txt
    │   ├── image_class_labels.txt
    │   ├── images.txt
    │   └── train_test_split.txt
    │
    └── segmentations/
```

## Proposed Experimental Framework

The experimental pipeline will consist of the following stages:

```text
Original Images
      │
      ▼
Image Classification Model
      │
      ├── Baseline Predictions
      └── Baseline Visual Explanations
      │
      ▼
Image Corruption at Multiple Severity Levels
      │
      ▼
Predictions and Explanations on Corrupted Images
      │
      ├── Prediction Robustness Evaluation
      ├── Explanation Stability Evaluation
      ├── Localization Evaluation
      └── Faithfulness Evaluation
```

## Image Corruptions

The study will examine multiple categories of image corruption.

Potential corruption categories include:

### Noise Corruptions

* Gaussian noise
* Shot noise
* Impulse noise

### Blur Corruptions

* Gaussian blur
* Motion blur
* Defocus blur

### Weather and Environmental Corruptions

* Fog
* Frost
* Snow

### Digital Corruptions

* JPEG compression
* Pixelation
* Contrast alteration
* Brightness alteration

Each corruption will be applied at multiple severity levels to evaluate progressive degradation.

The final corruption set may be revised during experimental design and pilot testing.

## Classification Models

The study will use one or more convolutional neural-network or vision-transformer-based image classifiers.

Potential models include:

* ResNet
* DenseNet
* EfficientNet
* Vision Transformer

Pretrained models may be fine-tuned on the CUB-200-2011 training set.

The final model selection and training configuration will be documented after baseline experiments.

## Visual Explanation Methods

Potential visual explanation methods include:

* Grad-CAM
* Grad-CAM++
* Integrated Gradients
* SmoothGrad
* Score-CAM

The methods will be selected to represent different explanation-generation mechanisms.

## Evaluation Dimensions

### 1. Prediction Robustness

Prediction robustness will be evaluated using classification metrics such as:

* Accuracy
* Top-k accuracy
* Macro F1-score
* Probability change
* Prediction consistency
* Corruption error

### 2. Explanation Stability

Explanation stability measures the similarity between explanations generated for an original image and its corrupted version.

Potential measures include:

* Structural Similarity Index
* Pearson correlation
* Spearman rank correlation
* Cosine similarity
* Intersection over Union of salient regions
* Top-k attribution overlap

### 3. Explanation Localization

Localization evaluates whether the explanation highlights the actual bird region.

The segmentation masks will be used as the reference object regions.

Potential localization measures include:

* Saliency-mask Intersection over Union
* Pointing-game accuracy
* Energy inside the segmentation mask
* Saliency precision
* Saliency recall

### 4. Explanation Faithfulness

Faithfulness evaluates whether the highlighted image regions genuinely influence the model prediction.

Potential faithfulness measures include:

* Deletion score
* Insertion score
* Confidence drop after masking salient regions
* Area over the perturbation curve
* Comparison with random-region perturbation

## Repository Structure

```text
visual-explanation-robustness/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── experiment_config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── corrupted/
│   └── README.md
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_corruption_analysis.ipynb
│   └── 04_explanation_analysis.ipynb
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── train_model.py
│   ├── generate_corruptions.py
│   ├── generate_explanations.py
│   └── evaluate_explanations.py
│
├── src/
│   ├── data/
│   ├── models/
│   ├── corruptions/
│   ├── explanations/
│   ├── evaluation/
│   └── utils/
│
├── models/
│
├── results/
│   ├── figures/
│   ├── tables/
│   ├── metrics/
│   └── explanations/
│
└── docs/
```

The repository structure may evolve as the implementation progresses.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/visual-explanation-robustness.git
cd visual-explanation-robustness
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

The implementation is currently under development.

The planned workflow is:

```bash
python scripts/prepare_dataset.py
python scripts/train_model.py
python scripts/generate_corruptions.py
python scripts/generate_explanations.py
python scripts/evaluate_explanations.py
```

The exact commands and configuration options will be updated as the corresponding modules are implemented.

## Reproducibility

To support reproducibility, the project will record:

* Dataset split
* Random seeds
* Model architecture
* Pretrained weights
* Training hyperparameters
* Image preprocessing procedures
* Corruption type and severity
* Explanation-method parameters
* Evaluation thresholds
* Software-library versions

Experiment settings will be maintained in configuration files inside the `configs/` directory.

## Current Status

The project is currently in the initial development stage.

Completed:

* Research topic selected
* Preliminary research title finalized
* CUB-200-2011 dataset downloaded
* Segmentation annotations downloaded
* Initial repository structure prepared

In progress:

* Dataset validation and exploration
* Experimental protocol design
* Model-selection strategy
* Corruption-selection strategy
* Explanation-method selection
* Evaluation-metric implementation

## Data and Model Files

The following files are intentionally excluded from GitHub:

* Original dataset images
* Segmentation masks
* Processed image datasets
* Corrupted image datasets
* Trained model checkpoints
* Large NumPy arrays
* Temporary experiment outputs
* Environment variables and credentials

These exclusions are controlled through the `.gitignore` file.

## Ethical and Responsible Use

This project is intended for academic research on the reliability and robustness of explainable artificial intelligence methods.

The dataset and any pretrained models used in the study remain subject to their respective licences and usage conditions.

## Authors

**Mr. Sandip Chakraborty**
Department of Computational Sciences
Brainware University

## Citation

A citation entry will be added after the associated research paper is published.

## Licence

No open-source licence has currently been assigned.

The repository is private, and all rights are reserved by the author unless otherwise stated.
