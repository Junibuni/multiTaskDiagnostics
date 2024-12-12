import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_anchors=9, num_classes=1):
        super(DetectionHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, num_anchors * (4 + num_classes), kernel_size=1)  # 4 for box, classes
        )
    
    def forward(self, x):
        return self.head(x)
