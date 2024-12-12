import torch
import torch.nn as nn
from src.backbone import SAMBackbone
from src.segmentation_head import SegmentationHead
from src.detection_head import DetectionHead
from src.classification_head import ClassificationHead

class UnifiedModel(nn.Module):
    def __init__(self, sam_checkpoint, backbone_type="vit_h", num_classes=1, num_detection_classes=1):
        super(UnifiedModel, self).__init__()
        
        self.backbone = SAMBackbone(sam_checkpoint, model_type=backbone_type)
        
        self.segmentation_head = SegmentationHead(in_channels=256, num_classes=num_classes)
        self.detection_head = DetectionHead(in_channels=256, num_classes=num_detection_classes)
        self.classification_head = ClassificationHead(in_channels=256, num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        segmentation_output = self.segmentation_head(features)
        detection_output = self.detection_head(features)
        classification_output = self.classification_head(features)
        
        return segmentation_output, detection_output, classification_output
