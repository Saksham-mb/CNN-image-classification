# EuroSAT Land Use Classification with TinyVGG
A CNN using TinyVGG architecture showcasing modular code and end-to-end reproducible results on EuroSAT geospatial imagery from Sentinel-2.

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
* **TinyVGG:**
  * Two convolutional blocks (`Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d`).
  * Fully connected head with a hidden linear layer of 64 nodes.
  * Parameter Count: ~180K parameters.

## 3. Results

| Model | Test Accuracy | Macro F1 |
| :--- | :--- | :--- |
| **TinyVGG (Scratch)** | 94.52% | 0.73 |

Confusion matrix, loss curves and other metrics in `results/`

## 4. Error Analysis & Failure Modes
Inspection of the misclassified samples is stored in `results/misclassified.json`.

## 5. How to Run

### Setup Environment
```bash
git clone https://github.com/<Saksham-mb>/cnn-image-classification.git
cd cnn-image-classification
pip install -r requirements.txt
```

### Train the Model
Because trained weights are excluded from Git tracking, run the training script first to train for 20 epochs and generate `models/TinyVGG_model.pth` along with loss curves:
```bash
python src/train.py
```

### Evaluate the Model
Run the evaluation script to compute metrics, produce the classification report, and generate `results/figures/confusion_matrix.png`:
```bash
python src/evaluate.py
```
