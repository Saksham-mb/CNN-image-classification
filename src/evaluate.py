import torch
from pathlib import Path
from data import test_dataloaders
from model import TinyVGG

PROJECT_ROOT=Path(__file__).resolve().parent.parent
MODEL_DIR=PROJECT_ROOT / 'models'

model=TinyVGG(3, 16, 64, 10)
model.load_state_dict(torch.load(MODEL_DIR / 'TinyVGG_model.pth'))

test_dataloader=test_dataloaders(100)

with torch.inference_mode():
    model.eval()
    for features, labels in test_dataloaders:
        preds_logits=model(features)
        y=torch.argmax(preds_logits, dim=1)