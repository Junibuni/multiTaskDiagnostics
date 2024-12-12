import torch.nn as nn

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SegmentationHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)  # Final output layer
        )
    
    def forward(self, x):
        return self.head(x)
