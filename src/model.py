import torch
from torch import nn

class TinyVGG(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, nodes_layer: int, out_classes):
        super().__init__()
        self.conv_block_1=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        hidden_in_channels=hidden_channels*13*13
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_in_channels, out_features=nodes_layer),
            nn.ReLU(),
            nn.Linear(in_features=nodes_layer, out_features=out_classes)
        )
    def forward(self, x):
        x=self.conv_block_1(x)
        x=self.conv_block_2(x)
        return self.classifier(x)