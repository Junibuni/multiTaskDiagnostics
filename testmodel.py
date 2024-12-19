from unified_model import UnifiedModel
import torch

model = UnifiedModel(num_classes=14, num_detection_classes=1)

x = torch.randn(4, 3, 224, 224)

classification_output, detection_boxes, detection_scores = model(x)

print("Classification Output Shape:", classification_output.shape)  # (B, 14)
print("Detection Boxes Shape:", detection_boxes.shape)  # (B, num_anchors * 4, H/32, W/32)
print("Detection Scores Shape:", detection_scores.shape)  # (B, num_anchors * num_classes, H/32, W/32)

