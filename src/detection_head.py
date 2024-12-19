import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_anchors=1, num_classes=1):
        super(DetectionHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        self.box_regression = nn.Conv2d(256, num_anchors * 4, kernel_size=1)

        self.box_classification = nn.Conv2d(256, num_anchors * num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        boxes = self.box_regression(x)  # (B, num_anchors * 4, H, W)
        box_scores = self.box_classification(x)  # (B, num_anchors * num_classes, H, W)

        return boxes, box_scores
