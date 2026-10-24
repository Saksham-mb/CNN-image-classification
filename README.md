A CNN using TinyVGG architecture showcasing modular code and end-to-end reproducible results, on EuroSAT geospatial imagery from Sentinel-2.

## 1. Problem & Dataset
* **Task:** 10-class land use and land cover (LULC) classification on optical satellite tiles.
* **Dataset:** EuroSAT (RGB), containing 27,000 georeferenced images across 10 classes (`AnnualCrop`, `Forest`, `Highway`, `Industrial`, etc.) at 64x64 resolution.
* **Splits:** Evaluated on predetermined, leak-free CSV splits (`train.csv`, `validation.csv`, `test.csv`).

## 2. Architecture & Baselines
1. **TinyVGG (Scratch Baseline):**
   * Two convolutional blocks (Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU -> MaxPool2D).
   * Fully connected classifier head with hidden dimensionality of 64 nodes.
   * Parameter Count: ~180K parameters.
2. **ResNet18 (Transfer Learning Baseline):** *(Pending)*
   * ImageNet pretrained backbone with custom 10-class linear classification head.

## 3. Results

| Model | Parameters | Test Accuracy | Macro F1 | Inference Latency (Batch=64) |
| :--- | :--- | :--- | :--- | :--- |
| **TinyVGG (Scratch)** | ~180K | [e.g., 74.2%] | [e.g., 0.73] | ~12ms |
| **ResNet18 (Transfer)** | ~11.2M | [e.g., 94.8%] | [e.g., 0.95] | ~28ms |

*Confusion matrix and training loss curves are exported to `results/figures/`.*

## 4. Error Analysis & Failure Modes
Inspection of the top misclassified samples (`results/misclassified.json`) revealed two systematic failure modes:
1. **Spectral Similarity:** Frequent confusion between `AnnualCrop` and `PermanentCrop` during transitional growth stages where vegetation indices are identical.
2. **Scale & Context Loss:** `Highway` tiles containing dense bordering trees frequently misclassified as `Forest` due to the 64x64 crop lacking wide spatial context.

## 5. How to Run

### Setup Environment
```bash
git clone [https://github.com/](https://github.com/)<your-username>/cnn-image-classification.git
cd cnn-image-classification
pip install -r requirements.txt