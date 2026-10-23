print("Importing Modules...")

from model import TinyVGG
import torch
from data import train_dataloaders
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parent.parent
MODEL_DIR=PROJECT_ROOT / 'models'

epochs = 1
lr=0.001

print("Model Initializing...")

train_model=TinyVGG(3, 16, 64, 10)
loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(params=train_model.parameters(), lr=lr)

train_dataloader=train_dataloaders(batch_size=64)

print("Model Training...")
i=0

print(train_model)

for epoch in range(epochs):
    for features, labels in train_dataloader:
        i+=1
        print(i)
        train_model.train()
        y_train_logit=train_model(features)
        loss=loss_fn(y_train_logit, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
print("Model training complete")

torch.save(train_model.state_dict(), MODEL_DIR / "TinyVGG_model.pth")
print("Model saved successfully")