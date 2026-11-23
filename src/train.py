import matplotlib.pyplot as plt
from model import TinyVGG, ResNet18
from utilities import train_model
import torch
from data import train_dataloaders, val_dataloaders
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

epochs = 20
lr = 0.001

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

model_0 = TinyVGG(in_channels=3,hidden_channels=16,nodes_layer=64,out_classes=10).to(device)
model_1 = ResNet18(out_classes=10).to(device)

train_dataloader = train_dataloaders(batch_size=64)
val_dataloader = val_dataloaders(batch_size=100)

train_losses_0, val_losses_0 = train_model(model_0, train_dataloader, val_dataloader, epochs=epochs, lr=lr)
train_losses_1, val_losses_1 = train_model(model_1, train_dataloader, val_dataloader, epochs=epochs, lr=lr)

plt.figure(figsize=(8, 5))
fir, axs = plt.subplots(1, 2, figsize=(12, 8))

axs[0].plot(range(1, epochs + 1), train_losses_0, "b", label="Train Loss")
axs[0].plot(range(1, epochs + 1), val_losses_0, "r", label="Validation Loss")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[0].set_title("TinyVGG")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(range(1, epochs + 1), train_losses_1, "b", label="Train Loss")
axs[1].plot(range(1, epochs + 1), val_losses_1, "r", label="Validation Loss")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Loss")
axs[1].set_title("ResNet18")
axs[1].legend()
axs[1].grid(True)

plt.suptitle("Loss curve comparison of TinyVGG and ResNet18 on training and validation data")
plt.savefig(FIGURES_DIR / "loss_curves.png", bbox_inches="tight")
plt.close()