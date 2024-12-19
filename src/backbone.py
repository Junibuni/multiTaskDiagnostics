import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SharedBackbone(nn.Module):
    def __init__(self, weights=ResNet18_Weights.DEFAULT):
        super(SharedBackbone, self).__init__()
        resnet = resnet18(weights=weights)
        # FC 제거 (convolution layer만 사용)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Feature Map

    def forward(self, x):
        features = self.encoder(x)  #(B, C, H/32, W/32)
        return features
