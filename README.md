# EuroSAT Land Use Classification
A CNN project comparing a custom TinyVGG architecture against a pretrained ResNet18 baseline on EuroSAT geospatial imagery from Sentinel-2.

## 1. Dataset
* **Dataset:** EuroSAT Sentinel-2 satellite imagery, containing 27,000 images across 10 classes (`AnnualCrop`, `Forest`, `Highway`, `Industrial`, etc.) at 64x64 resolution.
* **Splits:** Evaluated on predetermined splits (`train.csv`, `validation.csv`, `test.csv`).

### Download Instructions
1. Download the pre-split dataset from Kaggle: [EuroSAT Dataset](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset).
2. Extract the archive.
3. Place the contents directly into `datasets/EuroSAT/`.

### Expected Directory Structure
```text
datasets/
└── EuroSAT/
    ├── AnnualCrop/
    ├── Forest/
    ├── HerbaceousVegetation/
    ├── Highway/
    ├── Industrial/
    ├── Pasture/
    ├── PermanentCrop/
    ├── Residential/
    ├── River/
    ├── SeaLake/
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

## 2. Architecture & Baselines
* **TinyVGG (From Scratch):**
  * Two convolutional blocks (`Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d`).
  * Fully connected head with a hidden linear layer of 64 nodes (~180K trainable parameters).
* **ResNet18 (Transfer Learning — Feature Extraction):**
  * Pretrained `torchvision.models.resnet18` (`ImageNet1K` weights) with the entire convolutional backbone frozen (`requires_grad = False`).
  * Replaced final `fc` linear layer (`512 -> 10`, 5,130 trainable parameters).

## 3. Results

| Model | Test Accuracy | Macro F1 |
| :--- | :--- | :--- |
| **TinyVGG (Scratch)** | 91.33% | 0.9113 |
| **ResNet18 (Frozen Backbone)** | 86.41% | 0.8606 |

Confusion matrices, loss curves, and per-class classification reports are stored in `results/`.

## 4. Error Analysis & Failure Modes
Misclassified test predictions are logged in `results/tinyvgg_misclassified.json` and `results/resnet18_misclassified.json`. Training `TinyVGG` from scratch outperforms the frozen `ResNet18` baseline (91.33% vs. 86.41% accuracy) because ImageNet object-centric filters do not adapt to low-resolution (64x64) top-down satellite textures when only the linear classification head is trained. Across both models, errors concentrate in visually overlapping classes: `PermanentCrop` is most frequently confused with `HerbaceousVegetation` (31 errors in TinyVGG, 33 in ResNet18) due to nearly identical green canopy textures at 10m spatial resolution, while `River` is commonly misclassified as `Highway` (19 errors in TinyVGG, 22 in ResNet18) because both appear as narrow, winding linear corridors cutting across land patches.

## 5. How to Run

### Setup Environment
```bash
git clone https://github.com/Saksham-mb/cnn-image-classification.git
cd cnn-image-classification
pip install -r requirements.txt
```

### Train the Models
Because trained weights are excluded from Git tracking, run the training script first to train both models for 20 epochs and generate `models/TinyVGG_model.pth`, `models/ResNet_model.pth`, and `results/figures/loss_curves.png`:
```bash
python src/train.py
```

### Evaluate the Models
Run the evaluation script to compute metrics, produce the per-class classification report, and generate confusion matrices in `results/figures/`:
```bash
python src/evaluate.py
```