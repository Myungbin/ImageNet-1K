import torch
import torch.nn as nn
from ImageNet.model.vision_transformer import ViT


class VisionTransformer(nn.Module):
    def __init__(self, num_classes, model="vit_large"):
        vision_transformer = ViT()
        super(VisionTransformer, self).__init__()
        self.backbone = getattr(vision_transformer, model)(patch_size=14)
        self.classifier = nn.Linear(self.backbone.embed_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.backbone.norm(x)
        x = self.classifier(x)
        return x


class DINO(nn.Module):
    def __init__(self, num_classes):
        super(DINO, self).__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", f"dinov2_vitl14")
        self.clf = nn.Sequential(nn.Linear(self.model.embed_dim, num_classes))

    def forward(self, x):
        x = self.model(x)
        x = self.clf(x)
        return x
