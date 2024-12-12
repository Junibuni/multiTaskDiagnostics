import torch
import torch.nn as nn
from segment_anything import sam_model_registry

class SAMBackbone(nn.Module):
    def __init__(self, sam_checkpoint, model_type="vit_h"):
        super(SAMBackbone, self).__init__()
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

        self.encoder = sam.image_encoder

    def forward(self, x):
        features = self.encoder(x)
        return features
