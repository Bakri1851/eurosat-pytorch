from torch import nn
import torch

# A small CNN for image classification
class SmallCNN(nn.Module):
    def __init__(self, base_channels = 32, num_classes = 10):
        super().__init__()

        # block 1:  3 -> C            64x64 -> 32x32
        # ReLU and MaxPool2d are applied after each Conv2d layer to introduce non-linearity and reduce the spatial dimensions.

        self.block1 = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # block 2:  C -> 2C           32x32 -> 16x16
        self.block2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels* 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # block 3:  2C -> 4C          16x16 -> 8x8
        self.block3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels* 4),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # head
        self.head = nn.Linear(base_channels * 4, num_classes)


    def forward(self, x):
        # x arrives as (B, 3, 64, 64)
        x = self.block1(x)      # -> (B,  C, 32, 32)
        x = self.block2(x)      # -> (B, 2C, 16, 16)
        x = self.block3(x)      # -> (B, 4C,  8,  8)
        x = torch.mean(x, dim=(2, 3))  # -> global average pool -> (B, 4C)
        x = self.head(x)    # -> (B, num_classes)

        return x # -> (B, num_classes)

