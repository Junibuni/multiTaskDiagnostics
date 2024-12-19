import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ClassificationHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # GAP 적용
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)  # (B, C, 1, 1)
        x = torch.flatten(x, 1)  # (B, C)
        x = self.fc(x)  # (B, num_classes)
        return torch.sigmoid(x)  # multi label