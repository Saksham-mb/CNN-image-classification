import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_dataloader, val_dataloader, epochs: int, lr: int):
    loss_fn = torch.nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params=trainable_params, lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss=0
        val_loss=0
        for features, labels in train_dataloader:
            features, labels = features.to(device), labels.to(device)
            y_train_logit=model(features)
            loss=loss_fn(y_train_logit, labels)
            train_loss+=loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_losses.append(train_loss/len(train_dataloader))

        model.eval()
        with torch.inference_mode():
            for features, labels in val_dataloader:
                features, labels = features.to(device), labels.to(device)
                y_val_logit = model(features)
                loss = loss_fn(y_val_logit, labels)
                val_loss += loss.item()
                preds = torch.argmax(y_val_logit, dim=1)
        epoch_val_loss = val_loss / len(val_dataloader)
        if epoch_val_loss<best_val_loss:
            best_val_loss=epoch_val_loss
            torch.save(model.state_dict(), MODEL_DIR / f"{model.__class__.__name__}_model.pth")
        val_losses.append(epoch_val_loss)
    return train_losses, val_losses