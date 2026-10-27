import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch

from data import test_dataloaders
from model import TinyVGG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

model = TinyVGG(in_channels=3, hidden_channels=16, nodes_layer=64, out_classes=10).to(device)
model.load_state_dict(torch.load(MODEL_DIR / "TinyVGG_model.pth", map_location=device))
model.eval()

test_loader = test_dataloaders(batch_size=64)

all_preds = []
all_labels = []
misclassified = []
current_sample_idx = 1

with torch.inference_mode():
  for features, labels in test_loader:
    features, labels = features.to(device), labels.to(device)
    logits = model(features)
    preds = torch.argmax(logits, dim=1)

    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    for p, y in zip(preds_np, labels_np):
      if p != y:
        misclassified.append(
            {"sample_idx": current_sample_idx, "true": int(y), "pred": int(p)}
        )
      current_sample_idx += 1

    all_preds.extend(preds_np)
    all_labels.extend(labels_np)

all_preds=np.array(all_preds)
all_labels=np.array(all_labels)

report=classification_report(all_labels, all_preds, output_dict=True)
accuracy=float((all_preds == all_labels).mean())*100
print(f"Test Accuracy: {accuracy:.2f}%")

metrics= {
    "test_accuracy": round(accuracy, 2),
    "macro_f1": round(report["macro avg"]["f1-score"], 4),
    "total_misclassified": f"{len(misclassified)} out of {len(all_labels)}",
    "classification_report": report,
}

with open(RESULTS_DIR / "metrics.json", "w") as f:
  json.dump(metrics, f, indent=2)

with open(RESULTS_DIR / "misclassified.json", "w") as f:
  json.dump(misclassified[:20], f, indent=2)

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
for i in range(cm.shape[0]):
  for j in range(cm.shape[1]):
    ax.text(x=j,y=i,s=cm[i, j],va="center",ha="center",size="small" if cm[i, j] > 999 else "medium")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix (Test Acc: {accuracy:.2f}%)")
plt.savefig(FIGURES_DIR / "confusion_matrix.png", bbox_inches="tight")
plt.close()

print(f"Metrics saved to {RESULTS_DIR / 'metrics.json'} and {FIGURES_DIR / 'confusion_matrix.png'}")