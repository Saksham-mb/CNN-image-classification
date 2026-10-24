import matplotlib.pyplot as plt
from model import TinyVGG
import torch
from data import train_dataloaders, val_dataloaders
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PROJECT_ROOT=Path(__file__).resolve().parent.parent
MODEL_DIR=PROJECT_ROOT / 'models'
FIGURES_DIR=PROJECT_ROOT / 'results' / 'figures'

epochs = 1
lr=0.001

torch.manual_seed(42)

train_model=TinyVGG(3, 16, 64, 10)
loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(params=train_model.parameters(), lr=lr)

train_dataloader=train_dataloaders(batch_size=64)
val_dataloader=train_dataloaders(batch_size=100)

train_losses=[]
test_losses=[]
train_acc_per_epoch = []
val_acc_per_epoch = []

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
            y_test_logit=train_model(features)
            loss=loss_fn(y_test_logit, labels)

            test_loss+=loss.detach().item()
        test_losses.append(test_loss/len(val_dataloader))

torch.save(train_model.state_dict(), MODEL_DIR / "TinyVGG_model.pth")

plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.title('Train vs Test losses')
plt.plot(range(epochs), train_losses, c='b')
plt.plot(range(epochs), test_losses, c='r')
plt.savefig(FIGURES_DIR / 'loss_curves.png')
plt.savefig(FIGURES_DIR / "loss_curves.png", bbox_inches="tight")
plt.show()
plt.close()