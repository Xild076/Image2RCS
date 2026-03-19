import torch
import torch.nn as nn
import torchvision.models as models

class ImageRegressionModel(nn.Module):
    def __init__(self, out_dim: int = 1, hidden_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, out_dim),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(pretrained: bool = True, out_dim: int = 1, hidden_dim: int = 256) -> ImageRegressionModel:
    return ImageRegressionModel(out_dim=out_dim, hidden_dim=hidden_dim, pretrained=pretrained)

