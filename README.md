A CNN using TinyVGG architecture showcasing modular code and end-to-end reproducible results, on EuroSAT geospatial imagery from Sentinel-2.

## 1. Dataset
* **Dataset:** EuroSAT Sentinel-2 satellite imagery, containing 27,000 images across 10 classes (`AnnualCrop`, `Forest`, `Highway`, `Industrial`, etc.) at 64x64 resolution.
* **Splits:** Evaluated on predetermined splits (`train.csv`, `validation.csv`, `test.csv`).

### Download Instructions
1. Download the pre-split dataset from Kaggle: [EuroSAT Dataset](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) (or via `torchvision.datasets.EuroSAT`).
2. Extract the archive.
3. Place the contents directly into `datasets/EuroSAT/`.

### Expected Directory Structure
Your local environment must match this exact structure before running the training script:

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

## 2. Architecture & Baselines
1. **TinyVGG (Scratch Baseline):**
   * Two convolutional blocks (Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU -> MaxPool2D).
   * Parameter Count: ~180K parameters.

## 3. Results

| Model | Test Accuracy | Macro F1|
| :--- | :--- | :--- |
| **TinyVGG (Scratch)** | 94.52% | 0.73 |

Check confusion matrix, loss curves and other metrics in `results/`

## 4. Error Analysis & Failure Modes
Inspection of the 20 misclassified samples (`results/misclassified.json`).

## 5. How to Run

### Setup Environment
```bash
git clone [https://github.com/](https://github.com/)<your-username>/cnn-image-classification.git
cd cnn-image-classification
pip install -r requirements.txt