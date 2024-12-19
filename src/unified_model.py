import torch
import torch.nn as nn
from src.backbone import SharedBackbone
from src.classification_head import ClassificationHead
from src.detection_head import DetectionHead

class UnifiedModel(nn.Module):
    def __init__(self, num_classes=14, num_detection_classes=1):
        super(UnifiedModel, self).__init__()
        self.backbone = SharedBackbone()
        
        self.classification_head = ClassificationHead(in_channels=512, num_classes=num_classes)
        self.detection_head = DetectionHead(in_channels=512, num_anchors=1, num_classes=num_detection_classes)

    def forward(self, x):
        features = self.backbone(x)  # (B, 512, H/32, W/32)
        
        classification_output = self.classification_head(features)  # (B, num_classes)
        
        detection_boxes, detection_scores = self.detection_head(features)
        # detection_boxes: (B, num_anchors * 4, H/32, W/32)
        # detection_scores: (B, num_anchors * num_classes, H/32, W/32)
        
        return classification_output, detection_boxes, detection_scores
