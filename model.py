import torch
import torch.nn as nn
from torchvision import models

class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()

        self.backbone = models.convnext_tiny(
            weights=None  # IMPORTANT: no pretrained here
        )

        in_features = self.backbone.classifier[2].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)