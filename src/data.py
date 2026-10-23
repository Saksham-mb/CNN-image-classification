from pathlib import Path
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "datasets" / "EuroSAT"


class EuroSATDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel_path = row["Filename"]
        label = row["Label"]

        full_img_path = self.img_dir / img_rel_path
        image = Image.open(full_img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

train_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)


def train_dataloaders(batch_size: int = 32):
    train_dataset = EuroSATDataset(csv_file=DATA_DIR / "train.csv",img_dir=DATA_DIR,transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    return train_loader

def test_dataloaders(batch_size: int = 32):
    test_dataset = EuroSATDataset(csv_file=DATA_DIR / "test.csv",img_dir=DATA_DIR,transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return test_loader