import matplotlib.pyplot as plt
from model import TinyVGG
import torch
from data import train_dataloaders, val_dataloaders
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

epochs = 20
lr = 0.001

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)

train_model = TinyVGG(in_channels=3,hidden_channels=16,nodes_layer=64,out_classes=10).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=train_model.parameters(), lr=lr)

train_dataloader = train_dataloaders(batch_size=64)
val_dataloader = val_dataloaders(batch_size=100)

train_losses = []
val_losses = []

for epoch in range(epochs):
    train_model.train()
    train_loss=0
    test_loss=0
    for features, labels in train_dataloader:
        features, labels = features.to(device), labels.to(device)
        y_train_logit=train_model(features)
        loss=loss_fn(y_train_logit, labels)
        train_loss+=loss.detach().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_losses.append(train_loss/len(train_dataloader))

    train_model.eval()
    with torch.inference_mode():
        for features, labels in val_dataloader:
            features, labels = features.to(device), labels.to(device)
            y_val_logit = train_model(features)
            loss = loss_fn(y_val_logit, labels)
            val_loss += loss.item()
            preds = torch.argmax(y_val_logit, dim=1)
    epoch_val_loss = val_loss / len(val_dataloader)
    val_losses.append(epoch_val_loss)

torch.save(train_model.state_dict(), MODEL_DIR / "TinyVGG_model.pth")

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, "b", label="Train Loss")
plt.plot(range(1, epochs + 1), val_losses, "r", label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(FIGURES_DIR / "loss_curves.png", bbox_inches="tight")
plt.close()