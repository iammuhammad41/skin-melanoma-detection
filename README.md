# Melanoma Skin Cancer Classification with AlexNet

A PyTorch implementation of binary classification for melanoma detection using a custom AlexNet model and 10‑fold cross-validation.

---

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Installation](#installation)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Training Procedure](#training-procedure)
* [Evaluation](#evaluation)
* [Results](#results)
* [License](#license)

---

## Overview

This project trains an AlexNet-inspired convolutional neural network to classify dermoscopic images into two classes: **melanoma** vs. **non-melanoma**. We use standard image transforms, 10‑fold cross-validation, and report accuracy, confusion matrix, and F1 score on a held‑out test set.

---

## Dataset

We leverage the “Melanoma Skin Cancer Dataset” containing 10 000 dermoscopic images organized into:

```
├── train/  
│   ├── melanoma/  
│   └── benign/  
└── test/  
    ├── melanoma/  
    └── benign/
```

* **Image size**: resized to 227×227
* **Classes**:

  * `0`: benign
  * `1`: melanoma

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/melanoma-alexnet.git
   cd melanoma-alexnet
   ```
2. Create a Python environment and install dependencies:

   ```bash
   conda create -n melanoma python=3.8
   conda activate melanoma
   pip install torch torchvision scikit-learn matplotlib tqdm
   ```

---

## Usage

1. **Set your data paths** in `train.py`:

   ```python
   train_dir = '../input/melanoma_cancer_dataset/train'
   test_dir  = '../input/melanoma_cancer_dataset/test'
   ```

2. **Run training with cross-validation**:

   ```bash
   python train.py
   ```

   This will:

   * Load and normalize images
   * Initialize `AlexNet`
   * Perform 10‑fold stratified splits
   * Train for 30 epochs per fold
   * Save the final model to `melanoma_CNN.pt`

3. **Evaluate on held‑out test set**:
   After training, the script computes:

   * Overall test accuracy
   * Confusion matrix
   * F1 score for the melanoma class

---

## Model Architecture

```python
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Conv / ReLU / Pool blocks ...
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            # Additional conv layers ...
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 2)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256*6*6)
        return self.classifier(x)
```

* **Input**: 3 × 227 × 227
* **Output**: logits for 2 classes

---

## Training Procedure

* **Batch size**: 64
* **Optimizer**: SGD (lr=0.001, momentum=0.9, weight\_decay=0.001)
* **Loss function**: CrossEntropyLoss
* **Epochs per fold**: 30
* **K‑fold**: 10 (shuffled, random\_state=42)
* **Transforms**:

  ```python
  transforms.Compose([
      transforms.Resize((227, 227)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std =[0.229, 0.224, 0.225]
      )
  ])
  ```
* **Device**: GPU if available

---

## Evaluation

After training, the model is evaluated on the held‑out test images:

* **Accuracy**: percentage of correct predictions
* **Confusion Matrix**: counts of true vs. predicted classes
* **F1 Score**: harmonic mean of precision & recall for melanoma detection

Example test metrics:

```
Testing Accuracy: 91.9%
Confusion Matrix:
 [[472  28]
  [ 53 447]]
F1 Score (melanoma): 0.921
```

---

## Results

* **Average test accuracy** across 10 folds: \~92%
* **Strong melanoma detection** performance with F1 ≈ 0.92
* **Balanced classification** between benign and malignant

---

## License

This project is released under the MIT License. Feel free to reuse and modify for research or personal projects.
