from models.backbone import SAMBackbone
import torch

# Initialize SAM backbone
checkpoint_path = "checkpoints/sam_vit_h_4b8939.pth"
backbone = SAMBackbone(sam_checkpoint=checkpoint_path)

# Test with a dummy image
dummy_input = torch.randn(1, 3, 1024, 1024)  # Batch size 1, 3 channels, 1024x1024
features = backbone(dummy_input)

print(f"Feature map shape: {features.shape}")
