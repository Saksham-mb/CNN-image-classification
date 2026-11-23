import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch

from data import test_dataloaders
from model import ResNet18, TinyVGG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = PROJECT_ROOT / "datasets" / "EuroSAT"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model: torch.nn.Module, model_name: str, checkpoint_path: Path, test_loader, class_names: list[str]) -> dict:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    misclassified = []
    current_sample_idx = 0

    with torch.inference_mode():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            preds = torch.argmax(logits, dim=1)

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            for p, y in zip(preds_np, labels_np):
                if p!=y:
                    misclassified.append({
                        "sample_idx": current_sample_idx,
                        "true_id": int(y),
                        "true_class": class_names[int(y)],
                        "pred_id": int(p),
                        "pred_class": class_names[int(p)],
                    })
                current_sample_idx += 1
            all_preds.extend(preds_np)
            all_labels.extend(labels_np)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    accuracy = float((all_preds == all_labels).mean()) * 100
    macro_f1 = float(report["macro avg"]["f1-score"])

    model_name = model_name.lower()

    with open(RESULTS_DIR / f"{model_name}_misclassified.json", "w") as f:
        json.dump(misclassified, f, indent=2)

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="left")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j,y=i,s=cm[i, j],va="center",ha="center",size="small" if cm[i, j] > 999 else "medium")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} Confusion Matrix (Acc: {accuracy:.2f}%, Macro F1: {macro_f1:.4f})")
    plt.savefig(FIGURES_DIR / f"{model_name}_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    return {
        "test_accuracy": round(accuracy, 2),
        "macro_f1": round(macro_f1, 4),
        "misclassified_count": len(misclassified),
        "total_samples": int(len(all_labels)),
        "classification_report": report,
    }

with open(DATA_DIR / "label_map.json") as f:
    label_map = json.load(f)

class_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]
test_loader = test_dataloaders(batch_size=64)

models_to_evaluate = {
      "TinyVGG":(TinyVGG(in_channels=3, hidden_channels=16, nodes_layer=64, out_classes=10), MODEL_DIR / "TinyVGG_model.pth"),
      "ResNet18":(ResNet18(out_classes=10),MODEL_DIR / "ResNet_model.pth")
    }

all_metrics = {}
for name, (model, ckpt_path) in models_to_evaluate.items():
    all_metrics[name] = evaluate_model(model, name, ckpt_path, test_loader, class_names)

with open(RESULTS_DIR / "metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

print("\n| Model | Test Accuracy | Macro F1 | Misclassified |")
for name, res in all_metrics.items():
    print(f"| {name} | {res['test_accuracy']:.2f}% | {res['macro_f1']:.4f} | {res['misclassified_count']}/{res['total_samples']} |")