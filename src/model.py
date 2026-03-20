from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models


SUPPORTED_MODEL_TYPES = ("resnet18_baseline", "resnet18_se")
DEFAULT_MODEL_TYPE = "resnet18_baseline"


def _resnet18_weights(pretrained: bool):
    if not pretrained:
        return None
    try:
        return models.ResNet18_Weights.DEFAULT
    except Exception:
        return None


class ImageRegressionModel(nn.Module):
    """
    Baseline architecture kept compatible with legacy checkpoints.
    """

    def __init__(
        self,
        out_dim: int = 1,
        hidden_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        backbone = models.resnet18(weights=_resnet18_weights(pretrained))
        in_features = int(backbone.fc.in_features)
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ImageRegressionModelSE(nn.Module):
    """
    Basic squeeze-excitation style channel attention on global feature vector.
    """

    def __init__(
        self,
        out_dim: int = 1,
        hidden_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.2,
        se_reduction: int = 16,
    ):
        super().__init__()
        backbone = models.resnet18(weights=_resnet18_weights(pretrained))
        in_features = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.in_features = in_features

        reduction = max(1, int(se_reduction))
        squeeze_dim = max(8, in_features // reduction)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_features, squeeze_dim),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze_dim, in_features),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(in_features, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        gates = self.channel_attention(features)
        attended = features * gates
        return self.head(attended)


def normalize_model_type(model_type: str | None) -> str:
    if model_type is None:
        return DEFAULT_MODEL_TYPE
    token = str(model_type).strip().lower()
    aliases = {
        "baseline": "resnet18_baseline",
        "resnet18": "resnet18_baseline",
        "resnet18_baseline": "resnet18_baseline",
        "se": "resnet18_se",
        "resnet18_se": "resnet18_se",
    }
    normalized = aliases.get(token, token)
    if normalized not in SUPPORTED_MODEL_TYPES:
        supported = ", ".join(SUPPORTED_MODEL_TYPES)
        raise ValueError(f"Unsupported model type: {model_type}. Expected one of: {supported}")
    return normalized


def resolve_model_config(
    model_type: str,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    se_reduction: int = 16,
) -> dict[str, Any]:
    normalized_type = normalize_model_type(model_type)
    config: dict[str, Any] = {
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
    }
    if normalized_type == "resnet18_se":
        config["se_reduction"] = int(se_reduction)
    return config


def build_model(
    pretrained: bool = True,
    out_dim: int = 1,
    hidden_dim: int = 256,
    model_type: str = DEFAULT_MODEL_TYPE,
    model_config: dict[str, Any] | None = None,
) -> nn.Module:
    normalized_type = normalize_model_type(model_type)
    cfg = dict(model_config or {})
    resolved_hidden_dim = int(cfg.get("hidden_dim", hidden_dim))
    resolved_dropout = float(cfg.get("dropout", 0.2))

    if normalized_type == "resnet18_baseline":
        return ImageRegressionModel(
            out_dim=out_dim,
            hidden_dim=resolved_hidden_dim,
            pretrained=pretrained,
            dropout=resolved_dropout,
        )

    if normalized_type == "resnet18_se":
        resolved_reduction = int(cfg.get("se_reduction", 16))
        return ImageRegressionModelSE(
            out_dim=out_dim,
            hidden_dim=resolved_hidden_dim,
            pretrained=pretrained,
            dropout=resolved_dropout,
            se_reduction=resolved_reduction,
        )

    raise ValueError(f"Unsupported model type: {model_type}")
