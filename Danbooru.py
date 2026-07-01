import argparse
import copy
import csv
import json
import logging
import math
import os
import random
import re
import sys
import time
from difflib import SequenceMatcher
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover - environment dependent
    Image = None
    ImageOps = None

try:
    import torchvision.transforms as transforms
    from torchvision.transforms import InterpolationMode

    TORCHVISION_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    transforms = None
    InterpolationMode = None
    TORCHVISION_AVAILABLE = False

try:
    import timm

    USE_TIMM = True
except Exception:  # pragma: no cover - environment dependent
    timm = None
    USE_TIMM = False

try:
    from transformers import AutoModel, AutoProcessor

    TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    AutoModel = None
    AutoProcessor = None
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("danbooru_modern")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
LABEL_SEPARATORS = [",", ";", "|", "\t"]


# -------------------- Configuration --------------------
@dataclass
class ModelConfig:
    # Model family. ConvNeXt V2 is a better modern default than EfficientNet-B4 for many image tagging datasets.
    model_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    pretrained: bool = True
    backbone_mode: str = "pool"  # pool | features
    out_indices: Tuple[int, ...] = (1, 2, 3, 4)

    # Feature/fusion head.
    feature_dim: int = 768
    classifier_dropout: float = 0.2
    projection_dropout: float = 0.1
    fusion_type: str = "attention"  # only used by backbone_mode='features'
    attention_module: str = "none"  # none | se | cbam, only used by features mode
    use_cbam: bool = False  # backward-compatible alias
    pool_type: str = "gem"  # avg | gem, only used by features mode

    # Loss.
    loss_type: str = "asl"  # asl | bce | focal | smooth_bce
    use_focal_loss: bool = False  # backward-compatible alias
    use_label_smoothing: bool = False  # backward-compatible alias
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    asl_gamma_pos: float = 0.0
    asl_gamma_neg: float = 4.0
    asl_clip: float = 0.05
    asl_eps: float = 1e-8


def _tuple_from_any(value: Any) -> Tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(x) for x in value)
    if isinstance(value, list):
        return tuple(int(x) for x in value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return ()
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    raise TypeError(f"Cannot convert {value!r} to tuple[int, ...]")


def _coerce_config(config: Optional[Union[ModelConfig, argparse.Namespace, Dict[str, Any]]] = None) -> ModelConfig:
    if config is None:
        return ModelConfig()
    if isinstance(config, ModelConfig):
        return config

    valid_keys = {field.name for field in fields(ModelConfig)}
    if isinstance(config, argparse.Namespace):
        data = vars(config)
    elif isinstance(config, dict):
        data = dict(config)
    else:
        raise TypeError(f"Unsupported config type: {type(config)!r}")

    clean_data = {key: value for key, value in data.items() if key in valid_keys and value is not None}
    if "out_indices" in clean_data:
        clean_data["out_indices"] = _tuple_from_any(clean_data["out_indices"])

    # Backward compatibility with earlier checkpoints.
    if clean_data.get("use_focal_loss"):
        clean_data["loss_type"] = "focal"
    if clean_data.get("use_label_smoothing"):
        clean_data["loss_type"] = "smooth_bce"
    if clean_data.get("use_cbam") and clean_data.get("attention_module") in (None, "none"):
        clean_data["attention_module"] = "cbam"

    return ModelConfig(**clean_data)


def _config_to_dict(config: Union[ModelConfig, argparse.Namespace, Dict[str, Any]]) -> Dict[str, Any]:
    cfg = _coerce_config(config)
    data = asdict(cfg)
    data["out_indices"] = list(cfg.out_indices)
    return data


# -------------------- Basic modules --------------------
class GeM(nn.Module):
    """Generalized Mean Pooling."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * float(p))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"GeM expects [B, C, H, W], got {tuple(x.shape)}")
        p = self.p.clamp(min=self.eps)
        x = x.clamp(min=self.eps).pow(p)
        x = F.avg_pool2d(x, kernel_size=(x.size(-2), x.size(-1)))
        return x.pow(1.0 / p)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(in_channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        attention = torch.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.spatial_attention(self.channel_attention(x))


class SqueezeExciteLite(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(in_channels // reduction, 1)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.net(x)


class FeatureProjection(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention_module: str = "none",
        pool_type: str = "gem",
        dropout: float = 0.1,
    ):
        super().__init__()
        if attention_module == "cbam":
            self.attention = CBAM(in_channels)
        elif attention_module == "se":
            self.attention = SqueezeExciteLite(in_channels)
        elif attention_module == "none":
            self.attention = nn.Identity()
        else:
            raise ValueError("attention_module must be one of: none, se, cbam")
        if pool_type == "gem":
            self.pool = GeM(p=3.0)
        elif pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError("pool_type must be avg or gem")
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(self.pool(self.attention(x)))


# -------------------- Backbones and model --------------------
def _require_timm() -> None:
    if not USE_TIMM:
        raise ImportError("timm is required. Install it with: pip install timm")


def _fallback_model_names(model_name: str) -> List[str]:
    fallbacks = [
        model_name,
        "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "convnextv2_tiny.fcmae_ft_in1k",
        "convnextv2_tiny",
        "convnext_tiny.fb_in22k_ft_in1k",
        "convnext_tiny",
        "tf_efficientnetv2_s.in21k_ft_in1k",
        "tf_efficientnet_b3_ns",
        "efficientnet_b0",
    ]
    unique: List[str] = []
    for name in fallbacks:
        if name and name not in unique:
            unique.append(name)
    return unique


class TimmPoolBackbone(nn.Module):
    """General timm image backbone returning one global feature vector."""

    def __init__(self, model_name: str, pretrained: bool):
        super().__init__()
        _require_timm()
        self.model_name = model_name
        self.requested_pretrained = pretrained
        self.loaded_pretrained = False
        self.model = self._create_model(model_name, pretrained)
        try:
            self.out_dim = int(getattr(self.model, "num_features", 0) or 0)
        except Exception:
            self.out_dim = 0
        if self.out_dim <= 0:
            self.out_dim = self._infer_out_dim()
        logger.info("Loaded pool backbone: %s | dim=%d | pretrained=%s", self.model_name, self.out_dim, self.loaded_pretrained)

    def _create_model(self, model_name: str, pretrained: bool) -> nn.Module:
        errors: List[str] = []
        for use_pretrained in ([pretrained, False] if pretrained else [False]):
            for name in _fallback_model_names(model_name):
                try:
                    model = timm.create_model(name, pretrained=use_pretrained, num_classes=0, global_pool="avg")
                    self.model_name = name
                    self.loaded_pretrained = bool(use_pretrained)
                    if pretrained and not use_pretrained:
                        logger.warning("Could not load pretrained weights for %s. Using random init.", name)
                    return model
                except Exception as exc:  # pragma: no cover - depends on installed timm/model registry
                    errors.append(f"{name}(pretrained={use_pretrained}): {exc}")
        raise RuntimeError("Failed to create timm backbone. Last errors:\n" + "\n".join(errors[-8:]))

    def _infer_out_dim(self) -> int:
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            x = torch.zeros(1, 3, 224, 224)
            y = self.model(x)
            if isinstance(y, (list, tuple)):
                y = y[-1]
            if y.ndim > 2:
                y = F.adaptive_avg_pool2d(y, 1).flatten(1)
            dim = int(y.shape[1])
        if was_training:
            self.model.train()
        return dim

    def forward(self, x: Tensor) -> Tensor:
        y = self.model(x)
        if isinstance(y, (list, tuple)):
            y = y[-1]
        if y.ndim == 4:
            y = F.adaptive_avg_pool2d(y, 1).flatten(1)
        if y.ndim != 2:
            raise RuntimeError(f"Expected backbone vector [B, C], got {tuple(y.shape)}")
        return y


class TimmFeatureBackbone(nn.Module):
    """Feature-only timm backbone for CNN-like multi-scale fusion."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        out_indices: Sequence[int],
        feature_dim: int,
        attention_module: str,
        pool_type: str,
        projection_dropout: float,
    ):
        super().__init__()
        _require_timm()
        self.model_name = model_name
        self.requested_pretrained = pretrained
        self.loaded_pretrained = False
        self.out_indices = tuple(out_indices)
        self.backbone = self._create_model(model_name, pretrained)
        self.feature_channels = self._get_channels()
        self.projections = nn.ModuleList(
            [
                FeatureProjection(
                    in_channels=channels,
                    out_channels=feature_dim,
                    attention_module=attention_module,
                    pool_type=pool_type,
                    dropout=projection_dropout,
                )
                for channels in self.feature_channels
            ]
        )
        logger.info(
            "Loaded feature backbone: %s | channels=%s | pretrained=%s",
            self.model_name,
            self.feature_channels,
            self.loaded_pretrained,
        )

    def _create_model(self, model_name: str, pretrained: bool) -> nn.Module:
        errors: List[str] = []
        # For features_only not every Transformer-like timm model is supported. Fall back toward CNNs.
        for use_pretrained in ([pretrained, False] if pretrained else [False]):
            for name in _fallback_model_names(model_name):
                try:
                    model = timm.create_model(
                        name,
                        pretrained=use_pretrained,
                        features_only=True,
                        out_indices=self.out_indices,
                    )
                    self.model_name = name
                    self.loaded_pretrained = bool(use_pretrained)
                    if pretrained and not use_pretrained:
                        logger.warning("Could not load pretrained feature weights for %s. Using random init.", name)
                    return model
                except Exception as exc:  # pragma: no cover
                    errors.append(f"{name}(pretrained={use_pretrained}): {exc}")
        raise RuntimeError("Failed to create feature backbone. Last errors:\n" + "\n".join(errors[-8:]))

    def _get_channels(self) -> List[int]:
        info = getattr(self.backbone, "feature_info", None)
        if info is None:
            raise RuntimeError("features_only backbone has no feature_info")
        if hasattr(info, "channels"):
            channels = [int(x) for x in info.channels()]
        elif hasattr(info, "get_dicts"):
            channels = [int(item["num_chs"]) for item in info.get_dicts()]
        else:
            channels = [int(item["num_chs"]) for item in info]
        if len(channels) != len(self.out_indices):
            raise RuntimeError(f"Expected {len(self.out_indices)} feature scales, got {channels}")
        return channels

    def forward_raw_features(self, x: Tensor) -> List[Tensor]:
        features = list(self.backbone(x))
        if len(features) != len(self.projections):
            raise RuntimeError(f"Expected {len(self.projections)} raw features, got {len(features)}")
        return features

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        raw_features = self.forward_raw_features(x)
        return {f"scale_{idx}": proj(feat) for idx, (feat, proj) in enumerate(zip(raw_features, self.projections))}


class MultiScaleFusion(nn.Module):
    def __init__(self, num_scales: int, feature_dim: int, fusion_type: str = "attention"):
        super().__init__()
        if fusion_type not in {"attention", "weighted", "pyramid", "concat"}:
            raise ValueError("fusion_type must be one of: attention, weighted, pyramid, concat")
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        self.fusion_type = fusion_type
        if fusion_type == "attention":
            num_heads = 8 if feature_dim % 8 == 0 else 4 if feature_dim % 4 == 0 else 1
            self.attention = nn.MultiheadAttention(feature_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
            self.norm = nn.LayerNorm(feature_dim)
            self.dropout = nn.Dropout(0.1)
        elif fusion_type == "weighted":
            self.scale_weights = nn.Parameter(torch.ones(num_scales))
            self.norm = nn.LayerNorm(feature_dim)
        elif fusion_type == "pyramid":
            self.pyramid = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(feature_dim * 2, feature_dim),
                        nn.LayerNorm(feature_dim),
                        nn.SiLU(inplace=True),
                        nn.Dropout(0.1),
                    )
                    for _ in range(num_scales - 1)
                ]
            )
            self.norm = nn.LayerNorm(feature_dim)
        else:
            self.concat = nn.Sequential(
                nn.Linear(feature_dim * num_scales, feature_dim * 2),
                nn.LayerNorm(feature_dim * 2),
                nn.SiLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(feature_dim * 2, feature_dim),
                nn.LayerNorm(feature_dim),
            )

    def _ordered(self, features: Dict[str, Tensor]) -> List[Tensor]:
        values: List[Tensor] = []
        for idx in range(self.num_scales):
            key = f"scale_{idx}"
            if key not in features:
                raise KeyError(f"Missing feature key: {key}")
            value = features[key]
            if value.ndim != 2 or value.shape[1] != self.feature_dim:
                raise ValueError(f"{key} must be [B, {self.feature_dim}], got {tuple(value.shape)}")
            values.append(value)
        return values

    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        values = self._ordered(features)
        if self.fusion_type == "attention":
            stacked = torch.stack(values, dim=1)
            fused, _ = self.attention(stacked, stacked, stacked, need_weights=False)
            return self.norm(self.dropout(fused).mean(dim=1))
        if self.fusion_type == "weighted":
            weights = torch.softmax(self.scale_weights, dim=0).to(values[0].device, values[0].dtype)
            return self.norm(torch.stack([w * v for w, v in zip(weights, values)], dim=0).sum(dim=0))
        if self.fusion_type == "pyramid":
            fused = values[-1]
            for idx in range(self.num_scales - 2, -1, -1):
                fused = self.pyramid[idx](torch.cat([fused, values[idx]], dim=1))
            return self.norm(fused)
        return self.concat(torch.cat(values, dim=1))


# -------------------- Loss functions --------------------
class AsymmetricLossMultiLabel(nn.Module):
    """ASL for long-tail multi-label classification."""

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
        pos_weight: Optional[Tensor] = None,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        targets = targets.to(device=logits.device, dtype=logits.dtype)
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        log_pos = torch.log(xs_pos.clamp(min=self.eps))
        log_neg = torch.log(xs_neg.clamp(min=self.eps))
        loss = targets * log_pos + (1.0 - targets) * log_neg
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = xs_pos * targets + xs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * (1.0 - pt).pow(gamma)
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
            loss = loss * (targets * pos_weight + (1.0 - targets))
        loss = -loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[Union[float, Tensor]] = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        pos_weight: Optional[Tensor] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        targets = targets.to(device=inputs.device, dtype=inputs.dtype)
        pos_weight = self.pos_weight.to(device=inputs.device, dtype=inputs.dtype) if self.pos_weight is not None else None
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight, reduction="none")
        probabilities = torch.sigmoid(inputs)
        pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        focal_weight = (1.0 - pt).pow(self.gamma)
        if self.alpha is None:
            alpha_factor = 1.0
        elif isinstance(self.alpha, Tensor):
            alpha = self.alpha.to(device=inputs.device, dtype=inputs.dtype)
            alpha_factor = targets * alpha + (1.0 - targets) * (1.0 - alpha)
        else:
            alpha = float(self.alpha)
            alpha_factor = targets * alpha + (1.0 - targets) * (1.0 - alpha)
        loss = alpha_factor * focal_weight * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing: float = 0.05, reduction: str = "mean", pos_weight: Optional[Tensor] = None):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = smoothing
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        targets = targets.to(device=inputs.device, dtype=inputs.dtype)
        smoothed = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        pos_weight = self.pos_weight.to(device=inputs.device, dtype=inputs.dtype) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(inputs, smoothed, pos_weight=pos_weight, reduction=self.reduction)


def build_training_criterion(config: ModelConfig, pos_weight: Optional[Tensor] = None) -> nn.Module:
    if config.loss_type == "asl":
        return AsymmetricLossMultiLabel(
            gamma_pos=config.asl_gamma_pos,
            gamma_neg=config.asl_gamma_neg,
            clip=config.asl_clip,
            eps=config.asl_eps,
            pos_weight=pos_weight,
        )
    if config.loss_type == "focal":
        return FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma, pos_weight=pos_weight)
    if config.loss_type == "smooth_bce":
        return LabelSmoothingBCE(smoothing=config.label_smoothing, pos_weight=pos_weight)
    if config.loss_type == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    raise ValueError("loss_type must be one of: asl, bce, focal, smooth_bce")


class DeepDanbooruModern(nn.Module):
    def __init__(self, num_classes: int, config: Optional[Union[ModelConfig, argparse.Namespace, Dict[str, Any]]] = None):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_classes = int(num_classes)
        self.config = _coerce_config(config)
        self._validate_config()

        if self.config.backbone_mode == "pool":
            self.backbone = TimmPoolBackbone(self.config.model_name, self.config.pretrained)
            self.pre_classifier = nn.Sequential(
                nn.Linear(self.backbone.out_dim, self.config.feature_dim),
                nn.LayerNorm(self.config.feature_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(self.config.projection_dropout),
            )
            self.fusion = nn.Identity()
        elif self.config.backbone_mode == "features":
            self.backbone = TimmFeatureBackbone(
                model_name=self.config.model_name,
                pretrained=self.config.pretrained,
                out_indices=self.config.out_indices,
                feature_dim=self.config.feature_dim,
                attention_module=self.config.attention_module,
                pool_type=self.config.pool_type,
                projection_dropout=self.config.projection_dropout,
            )
            self.pre_classifier = nn.Identity()
            self.fusion = MultiScaleFusion(
                num_scales=len(self.backbone.feature_channels),
                feature_dim=self.config.feature_dim,
                fusion_type=self.config.fusion_type,
            )
        else:
            raise ValueError("backbone_mode must be pool or features")

        self.classifier = self._build_classifier()
        self.criterion = build_training_criterion(self.config)
        self._init_new_weights()

    def _validate_config(self) -> None:
        if self.config.backbone_mode not in {"pool", "features"}:
            raise ValueError("backbone_mode must be pool or features")
        if self.config.loss_type not in {"asl", "bce", "focal", "smooth_bce"}:
            raise ValueError("loss_type must be asl, bce, focal, or smooth_bce")
        if not 0.0 <= self.config.classifier_dropout < 1.0:
            raise ValueError("classifier_dropout must be in [0, 1)")
        if not 0.0 <= self.config.projection_dropout < 1.0:
            raise ValueError("projection_dropout must be in [0, 1)")
        if self.config.backbone_mode == "pool" and self.config.fusion_type == "none":
            return

    def _build_classifier(self) -> nn.Module:
        dim = self.config.feature_dim
        hidden = max(dim, min(dim * 2, 2048))
        return nn.Sequential(
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(self.config.classifier_dropout * 0.5),
            nn.Linear(hidden, self.num_classes),
        )

    def _init_new_weights(self) -> None:
        def init_layer(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.pre_classifier.apply(init_layer)
        if isinstance(self.fusion, nn.Module):
            self.fusion.apply(init_layer)
        self.classifier.apply(init_layer)

    def extract_features(self, x: Tensor) -> Tensor:
        if self.config.backbone_mode == "pool":
            return self.pre_classifier(self.backbone(x))
        features = self.backbone(x)
        return self.fusion(features)

    def forward(self, x: Tensor, targets: Optional[Tensor] = None) -> Dict[str, Tensor]:
        features = self.extract_features(x)
        logits = self.classifier(features)
        probabilities = torch.sigmoid(logits)
        output = {"logits": logits, "probabilities": probabilities, "features": features}
        if targets is not None:
            targets = targets.to(device=logits.device, dtype=logits.dtype)
            if targets.shape != logits.shape:
                raise ValueError(f"targets must have shape {tuple(logits.shape)}, got {tuple(targets.shape)}")
            output["loss"] = self.criterion(logits, targets)
        return output

    def predict(self, x: Tensor, threshold: float = 0.5, return_probs: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        was_training = self.training
        self.eval()
        with torch.inference_mode():
            probabilities = self(x)["probabilities"]
            predictions = (probabilities >= threshold).to(probabilities.dtype)
        if was_training:
            self.train()
        return (predictions, probabilities) if return_probs else predictions

    def predict_top_k(self, x: Tensor, k: int = 5) -> Tuple[Tensor, Tensor]:
        k = max(1, min(int(k), self.num_classes))
        was_training = self.training
        self.eval()
        with torch.inference_mode():
            probabilities = self(x)["probabilities"]
            top_probs, top_indices = torch.topk(probabilities, k=k, dim=1)
        if was_training:
            self.train()
        return top_indices, top_probs

    def get_feature_importance(self, x: Tensor) -> Dict[str, Tensor]:
        was_training = self.training
        self.eval()
        with torch.inference_mode():
            if self.config.backbone_mode == "features" and hasattr(self.backbone, "forward_raw_features"):
                raw = self.backbone.forward_raw_features(x)
                result = {f"scale_{idx}": feat.abs().mean(dim=(2, 3)) for idx, feat in enumerate(raw)}
            else:
                feat = self.extract_features(x)
                result = {"global": feat.abs()}
        if was_training:
            self.train()
        return result


# Backward-compatible aliases.
DeepDanbooruV3 = DeepDanbooruModern


def create_model(num_classes: int, **kwargs: Any) -> DeepDanbooruModern:
    valid_keys = {field.name for field in fields(ModelConfig)}
    data = {key: value for key, value in kwargs.items() if key in valid_keys and value is not None}
    if "out_indices" in data:
        data["out_indices"] = _tuple_from_any(data["out_indices"])
    config = ModelConfig(**{**asdict(ModelConfig()), **data})
    return DeepDanbooruModern(num_classes, config=config)


# -------------------- Dataset and transforms --------------------
class SquarePad:
    def __init__(self, fill: Union[int, Tuple[int, int, int]] = 255):
        self.fill = fill

    def __call__(self, image: Any) -> Any:
        if ImageOps is None:
            return image
        w, h = image.size
        if w == h:
            return image
        size = max(w, h)
        left = (size - w) // 2
        top = (size - h) // 2
        right = size - w - left
        bottom = size - h - top
        return ImageOps.expand(image, border=(left, top, right, bottom), fill=self.fill)


class DataAugmentation:
    @staticmethod
    def _interpolation() -> Any:
        if InterpolationMode is not None:
            return InterpolationMode.BICUBIC
        return 3

    @staticmethod
    def get_train_transforms(input_size: int = 384, resize_mode: str = "crop", augmentation_policy: str = "randaugment"):
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for transforms")
        interpolation = DataAugmentation._interpolation()
        ops: List[Any] = []
        if resize_mode == "crop":
            ops.append(transforms.RandomResizedCrop(input_size, scale=(0.65, 1.0), ratio=(0.75, 1.333), interpolation=interpolation))
        elif resize_mode == "pad":
            ops.extend([SquarePad(fill=255), transforms.Resize((input_size, input_size), interpolation=interpolation)])
        elif resize_mode == "squash":
            ops.append(transforms.Resize((input_size, input_size), interpolation=interpolation))
        else:
            raise ValueError("resize_mode must be crop, pad, or squash")
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
        if augmentation_policy == "randaugment" and hasattr(transforms, "RandAugment"):
            ops.append(transforms.RandAugment(num_ops=2, magnitude=7, interpolation=interpolation))
        elif augmentation_policy == "trivialaugment" and hasattr(transforms, "TrivialAugmentWide"):
            ops.append(transforms.TrivialAugmentWide(interpolation=interpolation))
        elif augmentation_policy == "augmix" and hasattr(transforms, "AugMix"):
            ops.append(transforms.AugMix(interpolation=interpolation))
        elif augmentation_policy == "light":
            ops.extend([
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.03),
                transforms.RandomRotation(degrees=8, interpolation=interpolation),
            ])
        elif augmentation_policy == "none":
            pass
        elif augmentation_policy not in {"randaugment", "trivialaugment", "augmix", "light", "none"}:
            raise ValueError("augmentation_policy must be randaugment, trivialaugment, augmix, light, or none")
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if hasattr(transforms, "RandomErasing") and augmentation_policy != "none":
            ops.append(transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"))
        return transforms.Compose(ops)

    @staticmethod
    def get_val_transforms(input_size: int = 384, resize_mode: str = "pad"):
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for transforms")
        interpolation = DataAugmentation._interpolation()
        if resize_mode == "crop":
            ops = [transforms.Resize(int(input_size * 1.14), interpolation=interpolation), transforms.CenterCrop(input_size)]
        elif resize_mode == "pad":
            ops = [SquarePad(fill=255), transforms.Resize((input_size, input_size), interpolation=interpolation)]
        elif resize_mode == "squash":
            ops = [transforms.Resize((input_size, input_size), interpolation=interpolation)]
        else:
            raise ValueError("resize_mode must be crop, pad, or squash")
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transforms.Compose(ops)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device(device_arg: Optional[str] = None) -> torch.device:
    return torch.device(device_arg or ("cuda" if torch.cuda.is_available() else "cpu"))


def load_tags(tags_path: Union[str, Path]) -> List[str]:
    path = Path(tags_path)
    if not path.exists():
        raise FileNotFoundError(f"Tags file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            tags = [str(item) for item in data]
        elif isinstance(data, dict) and "tags" in data:
            tags = [str(item) for item in data["tags"]]
        elif isinstance(data, dict) and "tag_to_idx" in data:
            mapping = {str(k): int(v) for k, v in data["tag_to_idx"].items()}
            tags = [tag for tag, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
        else:
            raise ValueError("Unsupported json tag file")
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("CSV tag file must have a header")
            col = "tag" if "tag" in reader.fieldnames else "name" if "name" in reader.fieldnames else reader.fieldnames[0]
            tags = [row[col].strip() for row in reader if row.get(col, "").strip()]
    else:
        tags = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        tags = [tag for tag in tags if tag and not tag.startswith("#")]
    if not tags:
        raise ValueError(f"No tags found in {path}")
    duplicates = sorted({tag for tag in tags if tags.count(tag) > 1})
    if duplicates:
        raise ValueError(f"Duplicate tags found: {duplicates[:10]}")
    return tags


def save_tags(tags: Sequence[str], path: Union[str, Path]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text("\n".join(str(tag) for tag in tags) + "\n", encoding="utf-8")


def _split_label_string(value: str) -> List[Union[str, int]]:
    value = value.strip()
    if not value:
        return []
    for sep in LABEL_SEPARATORS:
        if sep in value:
            parts = [item.strip() for item in value.split(sep) if item.strip()]
            break
    else:
        parts = [item.strip() for item in value.split() if item.strip()]
    if parts and all(part.lstrip("+-").isdigit() for part in parts):
        return [int(part) for part in parts]
    return parts


def parse_labels(value: Any) -> Union[List[str], List[int], Tensor]:
    if value is None:
        return []
    if isinstance(value, Tensor):
        return value.float()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value.startswith("[") or value.startswith("{"):
            try:
                return parse_labels(json.loads(value))
            except Exception:
                pass
        return _split_label_string(value)
    if isinstance(value, dict):
        return [str(key) for key, flag in value.items() if bool(flag)]
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        if all(isinstance(item, (int, float, bool)) for item in value):
            return [int(item) for item in value]
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError(f"Unsupported label value type: {type(value)!r}")


def _find_image_key(row: Dict[str, Any]) -> str:
    candidates = ["image", "filename", "file", "path", "image_path", "filepath"]
    for key in candidates:
        if key in row and str(row[key]).strip():
            return key
    raise KeyError(f"Could not find image column. Tried: {candidates}")


def _find_label_key(row: Dict[str, Any]) -> Optional[str]:
    candidates = ["labels", "tags", "tag", "classes", "class", "target", "targets"]
    for key in candidates:
        if key in row:
            return key
    return None


def read_annotations(annotation_path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(annotation_path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "items" in data:
            rows = data["items"]
        elif isinstance(data, dict) and "annotations" in data:
            rows = data["annotations"]
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError("JSON annotation file must be a list or contain items/annotations")
    elif suffix in {".jsonl", ".ndjson"}:
        rows = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
    else:
        raise ValueError(f"Unsupported annotation format: {suffix}. Use csv/json/jsonl.")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No annotation rows found in {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("All annotation entries must be dict/object")
    return rows


def build_tags_from_annotations(annotation_path: Union[str, Path]) -> List[str]:
    rows = read_annotations(annotation_path)
    tag_set: set[str] = set()
    max_index = -1
    found_vector = False
    for row in rows:
        label_key = _find_label_key(row)
        if label_key is None:
            continue
        labels = parse_labels(row[label_key])
        if isinstance(labels, Tensor):
            found_vector = True
            max_index = max(max_index, int(labels.numel()) - 1)
        elif labels and all(isinstance(item, int) for item in labels):
            values = [int(item) for item in labels]
            if set(values).issubset({0, 1}) and len(values) > 2:
                found_vector = True
                max_index = max(max_index, len(values) - 1)
            else:
                max_index = max(max_index, max(values))
        else:
            tag_set.update(str(item) for item in labels)
    if tag_set:
        return sorted(tag_set)
    if found_vector or max_index >= 0:
        return [f"class_{idx}" for idx in range(max_index + 1)]
    raise ValueError("Could not build tags from annotations. Provide --tags_path.")


class DanbooruTagDataset(Dataset):
    def __init__(
        self,
        annotation_path: Union[str, Path],
        image_dir: Optional[Union[str, Path]] = None,
        tags: Optional[Sequence[str]] = None,
        tags_path: Optional[Union[str, Path]] = None,
        input_size: int = 384,
        is_train: bool = True,
        strict_images: bool = True,
        resize_mode: str = "crop",
        augmentation_policy: str = "randaugment",
    ):
        if Image is None:
            raise ImportError("Pillow is required. Install it with: pip install pillow")
        self.annotation_path = Path(annotation_path)
        self.image_dir = Path(image_dir) if image_dir is not None else self.annotation_path.parent
        self.input_size = int(input_size)
        self.is_train = bool(is_train)
        self.resize_mode = resize_mode
        self.augmentation_policy = augmentation_policy
        if tags_path is not None:
            self.tags = load_tags(tags_path)
        elif tags is not None:
            self.tags = [str(tag) for tag in tags]
        else:
            self.tags = build_tags_from_annotations(annotation_path)
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(self.tags)}
        self.num_classes = len(self.tags)
        self.transform = (
            DataAugmentation.get_train_transforms(input_size, resize_mode=resize_mode, augmentation_policy=augmentation_policy)
            if is_train
            else DataAugmentation.get_val_transforms(input_size, resize_mode=("pad" if resize_mode == "crop" else resize_mode))
        )
        raw_rows = read_annotations(annotation_path)
        self.samples: List[Tuple[Path, Tensor]] = []
        missing_images: List[str] = []
        unknown_labels: set[str] = set()
        for row in raw_rows:
            image_key = _find_image_key(row)
            label_key = _find_label_key(row)
            image_path = self._resolve_image_path(str(row[image_key]))
            if not image_path.exists():
                missing_images.append(str(image_path))
                continue
            labels_raw = parse_labels(row[label_key]) if label_key is not None else []
            target, unknown = self._labels_to_target(labels_raw)
            unknown_labels.update(unknown)
            self.samples.append((image_path, target))
        if strict_images and missing_images:
            preview = "\n".join(missing_images[:10])
            raise FileNotFoundError(f"Missing {len(missing_images)} images. First examples:\n{preview}")
        if not self.samples:
            raise ValueError(f"No valid samples loaded from {annotation_path}")
        if unknown_labels:
            logger.warning("Ignored %d labels not present in tag list. Examples: %s", len(unknown_labels), sorted(unknown_labels)[:20])

    def _resolve_image_path(self, image_value: str) -> Path:
        path = Path(image_value)
        if path.is_absolute():
            return path
        direct = self.image_dir / path
        if direct.exists():
            return direct
        return self.annotation_path.parent / path

    def _labels_to_target(self, labels_raw: Union[List[str], List[int], Tensor]) -> Tuple[Tensor, List[str]]:
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        unknown: List[str] = []
        if isinstance(labels_raw, Tensor):
            labels_raw = labels_raw.flatten().tolist()
        if labels_raw and all(isinstance(item, int) for item in labels_raw):
            values = [int(item) for item in labels_raw]
            if len(values) == self.num_classes and set(values).issubset({0, 1}):
                return torch.tensor(values, dtype=torch.float32), []
            for idx in values:
                if 0 <= idx < self.num_classes:
                    target[idx] = 1.0
                else:
                    unknown.append(str(idx))
            return target, unknown
        for label in labels_raw:
            label = str(label).strip()
            if not label:
                continue
            idx = self.tag_to_idx.get(label)
            if idx is None:
                unknown.append(label)
            else:
                target[idx] = 1.0
        return target, unknown

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path, target = self.samples[index]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        return {"image": self.transform(image), "target": target.clone(), "path": str(image_path)}

    def label_counts(self) -> Tensor:
        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        for _, target in self.samples:
            counts += target
        return counts

    def sample_weights(self) -> Tensor:
        counts = self.label_counts()
        inv = 1.0 / torch.clamp(counts, min=1.0)
        weights = []
        for _, target in self.samples:
            pos = target > 0.5
            weights.append(float(inv[pos].mean().item()) if bool(pos.any()) else float(inv.mean().item()))
        return torch.tensor(weights, dtype=torch.float32)


def collate_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "target": torch.stack([item["target"] for item in batch], dim=0),
        "path": [item["path"] for item in batch],
    }


def create_dataloader(
    dataset: DanbooruTagDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = True,
    use_weighted_sampler: bool = False,
) -> DataLoader:
    sampler = None
    if use_weighted_sampler:
        sampler = WeightedRandomSampler(dataset.sample_weights(), num_samples=len(dataset), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        collate_fn=collate_batch,
        drop_last=False,
    )


# -------------------- Training utilities --------------------
def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


class ModelEma:
    def __init__(self, model: nn.Module, decay: float = 0.9998, device: Optional[torch.device] = None):
        self.ema = copy.deepcopy(unwrap_model(model)).eval()
        self.decay = float(decay)
        self.device = device
        if device is not None:
            self.ema.to(device=device)
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = unwrap_model(model)
        source_state = source.state_dict()
        ema_state = self.ema.state_dict()
        for key, ema_value in ema_state.items():
            source_value = source_state[key].detach()
            if self.device is not None:
                source_value = source_value.to(device=self.device)
            if not torch.is_floating_point(ema_value):
                ema_value.copy_(source_value)
            else:
                ema_value.mul_(self.decay).add_(source_value.to(dtype=ema_value.dtype), alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "ema_state_dict": self.ema.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.ema.load_state_dict(state["ema_state_dict"])


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    base = unwrap_model(model)
    backbone = getattr(base, "backbone", None)
    if backbone is None:
        return
    for parameter in backbone.parameters():
        parameter.requires_grad = trainable


def build_pos_weight(dataset: DanbooruTagDataset, clamp_max: float = 20.0) -> Tensor:
    positive = dataset.label_counts()
    total = float(len(dataset))
    negative = torch.clamp(torch.tensor(total) - positive, min=0.0)
    pos_weight = negative / torch.clamp(positive, min=1.0)
    return pos_weight.clamp(min=1.0, max=clamp_max)


def make_optimizer(model: nn.Module, lr: float, weight_decay: float, backbone_lr_mult: float = 0.25) -> torch.optim.Optimizer:
    decay_head: List[nn.Parameter] = []
    no_decay_head: List[nn.Parameter] = []
    decay_backbone: List[nn.Parameter] = []
    no_decay_backbone: List[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = ".backbone." in name or name.startswith("backbone.") or "backbone.model" in name
        no_decay = parameter.ndim <= 1 or name.endswith(".bias") or ".norm" in name.lower()
        if is_backbone and no_decay:
            no_decay_backbone.append(parameter)
        elif is_backbone:
            decay_backbone.append(parameter)
        elif no_decay:
            no_decay_head.append(parameter)
        else:
            decay_head.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay_backbone, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
            {"params": no_decay_backbone, "lr": lr * backbone_lr_mult, "weight_decay": 0.0},
            {"params": decay_head, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay_head, "lr": lr, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def make_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: str, epochs: int, steps_per_epoch: int = 1) -> Optional[Any]:
    if scheduler_type == "none":
        return None
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    if scheduler_type == "onecycle":
        max_lr = max(group["lr"] for group in optimizer.param_groups)
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=max(steps_per_epoch, 1))
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.1)
    raise ValueError("scheduler_type must be one of: none, cosine, onecycle, step")


def _make_grad_scaler(enabled: bool) -> Optional[Any]:
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type=device.type, enabled=False)


def model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 384, 384)) -> Dict[str, Union[int, float, Tuple[int, int, int]]]:
    base = unwrap_model(model)
    total = sum(p.numel() for p in base.parameters())
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    param_size_mb = total * 4 / (1024 ** 2)
    buffer_size_mb = sum(b.numel() for b in base.buffers()) * 4 / (1024 ** 2)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "model_size_mb": param_size_mb,
        "buffer_size_mb": buffer_size_mb,
        "total_size_mb": param_size_mb + buffer_size_mb,
        "input_size": input_size,
    }


def compute_multilabel_metrics(logits: Tensor, targets: Tensor, threshold: Union[float, Tensor] = 0.5) -> Dict[str, float]:
    probabilities = torch.sigmoid(logits)
    if isinstance(threshold, Tensor):
        thr = threshold.to(device=probabilities.device, dtype=probabilities.dtype).view(1, -1)
    else:
        thr = torch.tensor(float(threshold), device=probabilities.device, dtype=probabilities.dtype)
    predictions = probabilities >= thr
    targets_bool = targets >= 0.5
    tp = (predictions & targets_bool).sum().item()
    fp = (predictions & ~targets_bool).sum().item()
    fn = (~predictions & targets_bool).sum().item()
    tn = (~predictions & ~targets_bool).sum().item()
    micro_precision = tp / max(tp + fp, 1)
    micro_recall = tp / max(tp + fn, 1)
    micro_f1 = 2 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-12)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    exact_match = (predictions == targets_bool).all(dim=1).float().mean().item()
    tp_c = (predictions & targets_bool).sum(dim=0).float()
    fp_c = (predictions & ~targets_bool).sum(dim=0).float()
    fn_c = (~predictions & targets_bool).sum(dim=0).float()
    precision_c = tp_c / torch.clamp(tp_c + fp_c, min=1.0)
    recall_c = tp_c / torch.clamp(tp_c + fn_c, min=1.0)
    f1_c = 2 * precision_c * recall_c / torch.clamp(precision_c + recall_c, min=1e-12)
    return {
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(precision_c.mean().item()),
        "macro_recall": float(recall_c.mean().item()),
        "macro_f1": float(f1_c.mean().item()),
        "accuracy": float(accuracy),
        "exact_match": float(exact_match),
    }


def compute_mean_average_precision(logits: Tensor, targets: Tensor) -> float:
    scores = torch.sigmoid(logits).detach().float().cpu()
    labels = (targets.detach().float().cpu() >= 0.5).float()
    aps: List[float] = []
    for class_idx in range(labels.size(1)):
        y_true = labels[:, class_idx]
        positives = int(y_true.sum().item())
        if positives == 0:
            continue
        order = torch.argsort(scores[:, class_idx], descending=True)
        sorted_true = y_true[order]
        tp = torch.cumsum(sorted_true, dim=0)
        ranks = torch.arange(1, sorted_true.numel() + 1, dtype=torch.float32)
        precision_at_k = tp / ranks
        aps.append(float(((precision_at_k * sorted_true).sum() / max(positives, 1)).item()))
    return float(sum(aps) / len(aps)) if aps else 0.0


def calibrate_thresholds(logits: Tensor, targets: Tensor, steps: int = 41, min_threshold: float = 0.05, max_threshold: float = 0.95) -> Tensor:
    probabilities = torch.sigmoid(logits).float().cpu()
    labels = (targets.float().cpu() >= 0.5)
    grid = torch.linspace(min_threshold, max_threshold, steps=max(2, steps))
    thresholds = torch.full((probabilities.size(1),), 0.5, dtype=torch.float32)
    for class_idx in range(probabilities.size(1)):
        y = labels[:, class_idx]
        if int(y.sum().item()) == 0:
            continue
        best_f1 = -1.0
        best_t = 0.5
        p = probabilities[:, class_idx]
        for t in grid:
            pred = p >= t
            tp = (pred & y).sum().item()
            fp = (pred & ~y).sum().item()
            fn = (~pred & y).sum().item()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t.item())
        thresholds[class_idx] = best_t
    return thresholds


def format_metrics(metrics: Dict[str, float]) -> str:
    keys = ["loss", "micro_f1", "macro_f1", "micro_precision", "micro_recall", "mAP", "accuracy", "exact_match"]
    return " | ".join(f"{key}={metrics[key]:.4f}" for key in keys if key in metrics)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Tuple[Tensor, Tensor]:
    return batch["image"].to(device, non_blocking=True), batch["target"].to(device, non_blocking=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: Optional[Any] = None,
    amp: bool = False,
    grad_clip: Optional[float] = None,
    log_interval: int = 20,
    grad_accum_steps: int = 1,
    ema: Optional[ModelEma] = None,
    scheduler: Optional[Any] = None,
    scheduler_per_step: bool = False,
    channels_last: bool = False,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    start_time = time.time()
    use_amp = bool(amp and device.type == "cuda")
    grad_accum_steps = max(int(grad_accum_steps), 1)
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        images, targets = move_batch_to_device(batch, device)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with _autocast_context(device, use_amp):
            output = model(images, targets)
            loss = output["loss"] / grad_accum_steps
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        should_step = step % grad_accum_steps == 0 or step == len(loader)
        if should_step:
            if grad_clip is not None and grad_clip > 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), grad_clip)
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            if scheduler is not None and scheduler_per_step:
                scheduler.step()
        batch_size = images.size(0)
        total_loss += float(loss.item()) * grad_accum_steps * batch_size
        total_samples += batch_size
        if log_interval > 0 and step % log_interval == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            logger.info(
                "Epoch %d | step %d/%d | loss=%.4f | %.2f img/s",
                epoch,
                step,
                len(loader),
                total_loss / max(total_samples, 1),
                total_samples / elapsed,
            )
    return {"loss": total_loss / max(total_samples, 1)}


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: Union[float, Tensor] = 0.5,
    compute_map: bool = False,
    channels_last: bool = False,
) -> Tuple[Dict[str, float], Tensor, Tensor]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    logits_list: List[Tensor] = []
    targets_list: List[Tensor] = []
    for batch in loader:
        images, targets = move_batch_to_device(batch, device)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        output = model(images, targets)
        batch_size = images.size(0)
        total_loss += float(output["loss"].item()) * batch_size
        total_samples += batch_size
        logits_list.append(output["logits"].detach().cpu())
        targets_list.append(targets.detach().cpu())
    logits = torch.cat(logits_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    metrics = compute_multilabel_metrics(logits, targets, threshold=threshold)
    metrics["loss"] = total_loss / max(total_samples, 1)
    if compute_map:
        metrics["mAP"] = compute_mean_average_precision(logits, targets)
    return metrics, logits, targets


def _safe_torch_load(filepath: Union[str, Path], device: Union[str, torch.device]) -> Dict[str, Any]:
    try:
        return torch.load(filepath, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(filepath, map_location=device)


def save_training_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    best_metric: float,
    tags: Sequence[str],
    input_size: int,
    resize_mode: str,
    augmentation_policy: str,
    metrics: Optional[Dict[str, Any]] = None,
    thresholds: Optional[Tensor] = None,
    ema: Optional[ModelEma] = None,
) -> None:
    base = unwrap_model(model)
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    checkpoint: Dict[str, Any] = {
        "model_state_dict": base.state_dict(),
        "model_config": _config_to_dict(base.config),
        "num_classes": base.num_classes,
        "model_name": getattr(base.backbone, "model_name", base.config.model_name),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "tags": list(tags),
        "input_size": int(input_size),
        "resize_mode": resize_mode,
        "augmentation_policy": augmentation_policy,
        "metrics": metrics or {},
    }
    if thresholds is not None:
        checkpoint["thresholds"] = thresholds.detach().cpu().tolist()
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    if ema is not None:
        checkpoint["ema"] = ema.state_dict()
    torch.save(checkpoint, filepath)
    logger.info("Checkpoint saved: %s", filepath)


def load_checkpoint_metadata(checkpoint_path: Union[str, Path], device: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    return _safe_torch_load(checkpoint_path, device)


def _config_for_checkpoint(checkpoint: Dict[str, Any]) -> ModelConfig:
    config = _coerce_config(checkpoint.get("model_config", ModelConfig()))
    # Old checkpoints from the previous project had feature-only projection/fusion state dicts and no backbone_mode.
    state = checkpoint.get("model_state_dict", {})
    if "backbone_mode" not in checkpoint.get("model_config", {}) and any(k.startswith("backbone.projections") for k in state):
        config = replace(config, backbone_mode="features")
    config = replace(config, pretrained=False)
    if "model_name" in checkpoint:
        config = replace(config, model_name=checkpoint["model_name"])
    return config


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    num_classes: Optional[int] = None,
    use_ema: bool = False,
) -> DeepDanbooruModern:
    checkpoint = load_checkpoint_metadata(checkpoint_path, device)
    config = _config_for_checkpoint(checkpoint)
    classes = int(num_classes or checkpoint.get("num_classes") or len(checkpoint.get("tags", [])) or 1000)
    model = DeepDanbooruModern(classes, config=config).to(device)
    if use_ema and "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"]["ema_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    return model


# -------------------- Training main --------------------
def train_main(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = get_device(args.device)
    logger.info("Using device: %s", device)
    if args.channels_last and device.type == "cuda":
        logger.info("Using channels_last memory format")

    tags = load_tags(args.tags_path) if args.tags_path else build_tags_from_annotations(args.train_annotations)
    save_tags(tags, output_dir / "tags.txt")
    logger.info("Number of tags/classes: %d", len(tags))

    train_dataset = DanbooruTagDataset(
        annotation_path=args.train_annotations,
        image_dir=args.image_dir,
        tags=tags,
        input_size=args.input_size,
        is_train=True,
        strict_images=not args.allow_missing_images,
        resize_mode=args.resize_mode,
        augmentation_policy=args.augmentation_policy,
    )
    val_dataset = (
        DanbooruTagDataset(
            annotation_path=args.val_annotations,
            image_dir=args.val_image_dir or args.image_dir,
            tags=tags,
            input_size=args.input_size,
            is_train=False,
            strict_images=not args.allow_missing_images,
            resize_mode=args.val_resize_mode or ("pad" if args.resize_mode == "crop" else args.resize_mode),
            augmentation_policy="none",
        )
        if args.val_annotations
        else None
    )
    train_loader = create_dataloader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        use_weighted_sampler=args.use_weighted_sampler,
    )
    val_loader = create_dataloader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers) if val_dataset else None

    model_kwargs = model_kwargs_from_args(args)
    model = create_model(num_classes=len(tags), **model_kwargs).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if args.use_pos_weight:
        pos_weight = build_pos_weight(train_dataset, clamp_max=args.pos_weight_max).to(device)
        unwrap_model(model).criterion = build_training_criterion(unwrap_model(model).config, pos_weight=pos_weight)
        logger.info("Using positive class weights with max %.2f", args.pos_weight_max)

    if args.freeze_backbone_epochs > 0:
        set_backbone_trainable(model, False)
        logger.info("Backbone frozen for first %d epoch(s)", args.freeze_backbone_epochs)

    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, backbone_lr_mult=args.backbone_lr_mult)
    scheduler = make_scheduler(optimizer, args.scheduler, args.epochs, steps_per_epoch=math.ceil(len(train_loader) / max(args.grad_accum_steps, 1)))
    scheduler_per_step = args.scheduler == "onecycle"
    scaler = _make_grad_scaler(args.amp and device.type == "cuda")
    ema = ModelEma(model, decay=args.ema_decay, device=device) if args.ema else None

    start_epoch = 1
    best_metric = -math.inf
    calibrated_thresholds: Optional[Tensor] = None
    if args.resume:
        checkpoint = load_checkpoint_metadata(args.resume, device)
        unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if ema is not None and "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))
        if "thresholds" in checkpoint:
            calibrated_thresholds = torch.tensor(checkpoint["thresholds"], dtype=torch.float32)
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    stats = model_summary(model, input_size=(3, args.input_size, args.input_size))
    logger.info("Model: %s", unwrap_model(model).config.model_name)
    logger.info("Backbone mode: %s | Loss: %s", unwrap_model(model).config.backbone_mode, unwrap_model(model).config.loss_type)
    logger.info("Parameters: total=%s trainable=%s", f"{stats['total_params']:,}", f"{stats['trainable_params']:,}")

    if args.compile and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile")
        model = torch.compile(model)  # type: ignore[assignment]

    history: List[Dict[str, Any]] = []
    for epoch in range(start_epoch, args.epochs + 1):
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, True)
            optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, backbone_lr_mult=args.backbone_lr_mult)
            scheduler = make_scheduler(optimizer, args.scheduler, args.epochs, steps_per_epoch=math.ceil(len(train_loader) / max(args.grad_accum_steps, 1)))
            logger.info("Backbone unfrozen")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            scaler=scaler,
            amp=args.amp,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            grad_accum_steps=args.grad_accum_steps,
            ema=ema,
            scheduler=scheduler,
            scheduler_per_step=scheduler_per_step,
            channels_last=args.channels_last,
        )
        logger.info("Epoch %d train | %s", epoch, format_metrics(train_metrics))

        val_metrics: Dict[str, float] = {}
        if val_loader is not None:
            eval_model = ema.ema if ema is not None and args.validate_ema else model
            val_metrics, val_logits, val_targets = validate(
                eval_model,
                val_loader,
                device,
                threshold=calibrated_thresholds if calibrated_thresholds is not None else args.threshold,
                compute_map=args.compute_map,
                channels_last=args.channels_last,
            )
            if args.calibrate_thresholds:
                calibrated_thresholds = calibrate_thresholds(val_logits, val_targets, steps=args.threshold_steps)
                calibrated_metrics = compute_multilabel_metrics(val_logits, val_targets, threshold=calibrated_thresholds)
                val_metrics.update({f"calibrated_{k}": v for k, v in calibrated_metrics.items()})
                val_metrics["micro_f1"] = calibrated_metrics["micro_f1"]
                val_metrics["macro_f1"] = calibrated_metrics["macro_f1"]
                logger.info("Thresholds calibrated on validation set")
            logger.info("Epoch %d valid | %s", epoch, format_metrics(val_metrics))

        if scheduler is not None and not scheduler_per_step:
            scheduler.step()

        monitor_value = val_metrics.get(args.monitor, -val_metrics.get("loss", train_metrics["loss"])) if val_metrics else -train_metrics["loss"]
        is_best = monitor_value > best_metric
        if is_best:
            best_metric = monitor_value

        row = {"epoch": epoch, "train": train_metrics, "valid": val_metrics, "best_metric": best_metric}
        history.append(row)
        (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

        save_training_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_metric,
            tags,
            args.input_size,
            args.resize_mode,
            args.augmentation_policy,
            metrics={"train": train_metrics, "valid": val_metrics},
            thresholds=calibrated_thresholds,
            ema=ema,
        )
        if is_best:
            save_training_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_metric,
                tags,
                args.input_size,
                args.resize_mode,
                args.augmentation_policy,
                metrics={"train": train_metrics, "valid": val_metrics},
                thresholds=calibrated_thresholds,
                ema=ema,
            )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_training_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_metric,
                tags,
                args.input_size,
                args.resize_mode,
                args.augmentation_policy,
                metrics={"train": train_metrics, "valid": val_metrics},
                thresholds=calibrated_thresholds,
                ema=ema,
            )
    logger.info("Training completed. Best %s: %.4f", args.monitor, best_metric)
    return 0


# -------------------- Inference --------------------
def collect_image_paths(input_path: Union[str, Path], recursive: bool = True) -> List[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Input file is not a supported image: {path}")
        return [path]
    pattern = "**/*" if recursive else "*"
    images = [item for item in path.glob(pattern) if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS]
    images.sort()
    if not images:
        raise ValueError(f"No images found in {path}")
    return images


def load_image_tensor(image_path: Union[str, Path], input_size: int, resize_mode: str = "pad") -> Tensor:
    if Image is None:
        raise ImportError("Pillow is required. Install it with: pip install pillow")
    transform = DataAugmentation.get_val_transforms(input_size, resize_mode=resize_mode)
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    return transform(image)


def _probabilities_with_tta(model: nn.Module, images: Tensor, tta: str = "none") -> Tensor:
    probs = model(images)["probabilities"]
    if tta in {"hflip", "flip"}:
        probs = (probs + model(torch.flip(images, dims=[3]))["probabilities"]) * 0.5
    elif tta != "none":
        raise ValueError("tta must be none or hflip")
    return probs


def decode_predictions(
    probabilities: Tensor,
    tags: Sequence[str],
    threshold: float,
    top_k: int,
    thresholds: Optional[Tensor] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    probabilities = probabilities.detach().cpu()
    if thresholds is not None:
        thr = thresholds.detach().cpu().view(1, -1)
    else:
        thr = torch.full((1, probabilities.size(1)), float(threshold))
    for row_idx, row in enumerate(probabilities):
        k = min(top_k, row.numel())
        top_probs, top_indices = torch.topk(row, k=k)
        selected_indices = torch.where(row >= thr[row_idx if thr.size(0) > 1 else 0])[0].tolist()
        selected = [{"tag": tags[idx], "probability": float(row[idx].item())} for idx in selected_indices]
        selected.sort(key=lambda item: item["probability"], reverse=True)
        top = [{"tag": tags[int(idx)], "probability": float(prob)} for idx, prob in zip(top_indices.tolist(), top_probs.tolist())]
        results.append({"selected": selected, "top_k": top})
    return results


@torch.inference_mode()
def infer_images(
    model: nn.Module,
    image_paths: Sequence[Path],
    tags: Sequence[str],
    device: torch.device,
    input_size: int,
    batch_size: int,
    threshold: float,
    top_k: int,
    resize_mode: str = "pad",
    thresholds: Optional[Tensor] = None,
    tta: str = "none",
    channels_last: bool = False,
) -> List[Dict[str, Any]]:
    model.eval()
    outputs: List[Dict[str, Any]] = []
    if thresholds is not None:
        thresholds = thresholds.float().cpu()
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        images = torch.stack([load_image_tensor(path, input_size, resize_mode=resize_mode) for path in batch_paths], dim=0).to(device)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        probabilities = _probabilities_with_tta(model, images, tta=tta)
        decoded = decode_predictions(probabilities, tags=tags, threshold=threshold, top_k=top_k, thresholds=thresholds)
        for path, item in zip(batch_paths, decoded):
            outputs.append({"image": str(path), **item})
    return outputs


def write_inference_results(results: Sequence[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    if output_path.suffix.lower() == ".csv":
        with output_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "selected_tags", "selected_json", "top_k"])
            writer.writeheader()
            for item in results:
                writer.writerow(
                    {
                        "image": item["image"],
                        "selected_tags": " ".join(tag_item["tag"] for tag_item in item["selected"]),
                        "selected_json": json.dumps(item["selected"], ensure_ascii=False),
                        "top_k": json.dumps(item["top_k"], ensure_ascii=False),
                    }
                )
    else:
        output_path.write_text(json.dumps(list(results), ensure_ascii=False, indent=2), encoding="utf-8")



# -------------------- Semantic ontology and open-vocabulary inference --------------------
# This block extends the closed-set Danbooru classifier with an optional visual-language
# branch. It lets the project keep high precision on trained tags while still scoring
# candidate tags that were not present during supervised training.

DEFAULT_SEMANTIC_TAG_BANK: Dict[str, List[str]] = {
    "color": [
        "black hair", "brown hair", "blonde hair", "blue hair", "pink hair", "red hair", "green hair", "silver hair",
        "white hair", "purple hair", "orange hair", "multicolored hair", "black eyes", "brown eyes", "blue eyes",
        "green eyes", "red eyes", "purple eyes", "yellow eyes", "pink eyes", "heterochromia", "pale skin", "dark skin",
    ],
    "clothing": [
        "school uniform", "sailor uniform", "dress", "white dress", "black dress", "kimono", "yukata", "maid outfit",
        "jacket", "hoodie", "sweater", "shirt", "t-shirt", "skirt", "shorts", "pants", "thighhighs", "kneehighs",
        "boots", "sneakers", "hat", "cap", "hair ribbon", "bow", "gloves", "scarf", "swimsuit", "bikini", "armor",
    ],
    "body_and_character": [
        "1girl", "1boy", "solo", "two girls", "multiple girls", "multiple boys", "child", "teenage girl", "adult woman",
        "long hair", "short hair", "medium hair", "twintails", "ponytail", "braid", "animal ears", "cat ears", "fox ears",
        "horns", "wings", "tail", "glasses", "large breasts", "small breasts", "smile", "open mouth", "crying", "angry",
    ],
    "action": [
        "standing", "sitting", "lying", "running", "walking", "jumping", "dancing", "fighting", "holding sword",
        "holding gun", "holding food", "holding umbrella", "looking at viewer", "looking away", "waving", "pointing", "hugging",
        "sleeping", "eating", "drinking", "reading", "playing instrument", "singing", "casting magic",
    ],
    "composition": [
        "portrait", "upper body", "full body", "close-up", "cowboy shot", "from above", "from below", "side view",
        "back view", "wide shot", "indoors", "outdoors", "city background", "forest background", "sky background", "night",
        "day", "sunset", "dynamic pose", "simple background", "white background", "transparent background",
    ],
    "style": [
        "anime style", "manga style", "chibi", "pixel art", "watercolor", "oil painting", "digital art", "sketch",
        "lineart", "realistic", "semi-realistic", "game cg", "official art", "fan art", "highly detailed", "flat color",
    ],
}


def normalize_tag_text(tag: str) -> str:
    tag = str(tag).strip().lower()
    tag = tag.replace("_", " ").replace("-", " ")
    tag = re.sub(r"\s+", " ", tag)
    tag = re.sub(r"[^a-z0-9\s:/().+]+", "", tag)
    return tag.strip()


def tag_to_prompt(tag: str, prefix: str = "anime image with") -> str:
    text = normalize_tag_text(tag)
    if not text:
        return prefix
    # Keep compact Danbooru phrases, but convert them into natural text for VLM scoring.
    text = text.replace("1girl", "one girl").replace("1boy", "one boy")
    return f"{prefix} {text}"


def load_tag_list(path: Optional[Union[str, Path]]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tag list not found: {p}")
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "tags" in data:
                return [str(x) for x in data["tags"]]
            values: List[str] = []
            for value in data.values():
                if isinstance(value, list):
                    values.extend(str(x) for x in value)
                else:
                    values.append(str(value))
            return values
        if isinstance(data, list):
            return [str(x) for x in data]
        raise ValueError("Unsupported JSON tag list")
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def builtin_semantic_tags(categories: Optional[Sequence[str]] = None) -> List[str]:
    if not categories:
        selected = DEFAULT_SEMANTIC_TAG_BANK.keys()
    else:
        selected = categories
    tags: List[str] = []
    for category in selected:
        if category not in DEFAULT_SEMANTIC_TAG_BANK:
            logger.warning("Unknown semantic tag category ignored: %s", category)
            continue
        tags.extend(DEFAULT_SEMANTIC_TAG_BANK[category])
    # keep order, remove duplicates
    seen: set[str] = set()
    out: List[str] = []
    for tag in tags:
        key = normalize_tag_text(tag)
        if key and key not in seen:
            seen.add(key)
            out.append(tag)
    return out


def category_for_semantic_tag(tag: str) -> str:
    key = normalize_tag_text(tag)
    for category, tags in DEFAULT_SEMANTIC_TAG_BANK.items():
        if key in {normalize_tag_text(x) for x in tags}:
            return category
    return "open_vocab"


class SemanticTagOntology:
    """Canonical tag manager.

    The ontology does not delete labels. It keeps original labels as evidence and maps them to a
    canonical expression. Manual aliases always win; automatic grouping is deliberately conservative.
    """

    def __init__(self, tag_to_canonical: Optional[Dict[str, str]] = None, metadata: Optional[Dict[str, Any]] = None):
        self.tag_to_canonical = tag_to_canonical or {}
        self.metadata = metadata or {}

    @staticmethod
    def from_alias_file(alias_path: Optional[Union[str, Path]]) -> "SemanticTagOntology":
        mapping: Dict[str, str] = {}
        metadata: Dict[str, Any] = {"manual_aliases": False}
        if not alias_path:
            return SemanticTagOntology(mapping, metadata)
        path = Path(alias_path)
        if not path.exists():
            raise FileNotFoundError(f"Alias file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Alias JSON must be an object")
        # Supported forms:
        # {"canonical tag": ["alias a", "alias b"]}
        # {"aliases": {"canonical tag": [..]}, "preferred": {...}}
        aliases = data.get("aliases", data)
        if not isinstance(aliases, dict):
            raise ValueError("aliases must be a JSON object")
        for canonical, alias_values in aliases.items():
            canonical = str(canonical).strip()
            if isinstance(alias_values, str):
                alias_values = [alias_values]
            if not isinstance(alias_values, list):
                continue
            mapping[normalize_tag_text(canonical)] = canonical
            for alias in alias_values:
                mapping[normalize_tag_text(str(alias))] = canonical
        metadata["manual_aliases"] = True
        metadata["alias_path"] = str(path)
        return SemanticTagOntology(mapping, metadata)

    @staticmethod
    def build(
        tags: Sequence[str],
        alias_path: Optional[Union[str, Path]] = None,
        fuzzy_threshold: float = 0.97,
        strip_parentheses_for_alias: bool = False,
    ) -> "SemanticTagOntology":
        ontology = SemanticTagOntology.from_alias_file(alias_path)
        mapping = dict(ontology.tag_to_canonical)
        groups: Dict[str, List[str]] = {}
        for tag in tags:
            norm = normalize_tag_text(tag)
            if not norm:
                continue
            base = norm
            if strip_parentheses_for_alias:
                base = re.sub(r"\s*\([^)]*\)", "", base).strip()
            groups.setdefault(base, []).append(str(tag))
        # Exact normalized groups are safe to canonicalize.
        for _, members in groups.items():
            canonical = sorted(members, key=lambda x: (len(x), x))[0]
            for member in members:
                mapping.setdefault(normalize_tag_text(member), canonical)
        # Very conservative fuzzy grouping for small character variants only.
        keys = sorted(groups.keys())
        used: set[str] = set()
        for i, a in enumerate(keys):
            if a in used:
                continue
            cluster = [a]
            for b in keys[i + 1 :]:
                if b in used:
                    continue
                if abs(len(a) - len(b)) > 2:
                    continue
                score = SequenceMatcher(None, a, b).ratio()
                if score >= fuzzy_threshold:
                    cluster.append(b)
                    used.add(b)
            if len(cluster) > 1:
                members = [m for key in cluster for m in groups[key]]
                canonical = sorted(members, key=lambda x: (len(x), x))[0]
                for member in members:
                    mapping.setdefault(normalize_tag_text(member), canonical)
        return SemanticTagOntology(mapping, {**ontology.metadata, "fuzzy_threshold": fuzzy_threshold})

    @staticmethod
    def load(path: Optional[Union[str, Path]]) -> "SemanticTagOntology":
        if not path:
            return SemanticTagOntology()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Ontology not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        return SemanticTagOntology(
            tag_to_canonical={str(k): str(v) for k, v in data.get("tag_to_canonical", {}).items()},
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        path.write_text(
            json.dumps({"tag_to_canonical": self.tag_to_canonical, "metadata": self.metadata}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def canonical(self, tag: str) -> str:
        return self.tag_to_canonical.get(normalize_tag_text(tag), str(tag))

    def canonicalize_items(self, items: Sequence[Dict[str, Any]], score_key: str = "score") -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for item in items:
            tag = str(item.get("tag", ""))
            if not tag:
                continue
            canonical = self.canonical(tag)
            score = float(item.get(score_key, item.get("probability", 0.0)))
            if canonical not in merged or score > float(merged[canonical].get(score_key, merged[canonical].get("probability", -1.0))):
                new_item = dict(item)
                new_item["tag"] = canonical
                new_item.setdefault("original_tags", [])
                new_item["original_tags"] = sorted(set([tag] + list(new_item.get("original_tags", []))))
                merged[canonical] = new_item
            else:
                merged[canonical].setdefault("original_tags", [])
                merged[canonical]["original_tags"] = sorted(set(list(merged[canonical]["original_tags"]) + [tag]))
        return sorted(merged.values(), key=lambda x: float(x.get(score_key, x.get("probability", 0.0))), reverse=True)


def build_cooccurrence_graph(annotation_path: Union[str, Path], tags: Sequence[str], min_count: int = 3, top_k: int = 20) -> Dict[str, Any]:
    rows = read_annotations(annotation_path)
    tag_set = set(tags)
    counts: Dict[str, int] = {tag: 0 for tag in tags}
    pair_counts: Dict[Tuple[str, str], int] = {}
    for row in rows:
        label_key = _find_label_key(row)
        if label_key is None:
            continue
        labels = parse_labels(row[label_key])
        if isinstance(labels, Tensor):
            continue
        labels_str = [str(x) for x in labels if str(x) in tag_set]
        unique = sorted(set(labels_str))
        for tag in unique:
            counts[tag] = counts.get(tag, 0) + 1
        for i, a in enumerate(unique):
            for b in unique[i + 1 :]:
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
    neighbors: Dict[str, List[Dict[str, Any]]] = {tag: [] for tag in tags}
    for (a, b), c in pair_counts.items():
        if c < min_count:
            continue
        # Confidence-like values in both directions.
        neighbors[a].append({"tag": b, "count": c, "confidence": c / max(counts.get(a, 1), 1)})
        neighbors[b].append({"tag": a, "count": c, "confidence": c / max(counts.get(b, 1), 1)})
    for tag in neighbors:
        neighbors[tag] = sorted(neighbors[tag], key=lambda x: (x["confidence"], x["count"]), reverse=True)[:top_k]
    return {"counts": counts, "neighbors": neighbors, "min_count": min_count, "top_k": top_k}


def build_ontology_main(args: argparse.Namespace) -> int:
    tags = load_tags(args.tags_path) if args.tags_path else build_tags_from_annotations(args.annotations)
    if args.extra_candidate_tags:
        tags = list(tags) + load_tag_list(args.extra_candidate_tags)
    if args.include_builtin_semantics:
        tags = list(tags) + builtin_semantic_tags(args.semantic_categories.split(",") if args.semantic_categories else None)
    # de-duplicate while preserving order
    seen: set[str] = set()
    tags = [x for x in tags if not (normalize_tag_text(x) in seen or seen.add(normalize_tag_text(x)))]
    ontology = SemanticTagOntology.build(
        tags,
        alias_path=args.alias_path,
        fuzzy_threshold=args.fuzzy_threshold,
        strip_parentheses_for_alias=args.strip_parentheses_for_alias,
    )
    payload = {"tag_to_canonical": ontology.tag_to_canonical, "metadata": ontology.metadata, "tags": tags}
    if args.annotations:
        payload["cooccurrence"] = build_cooccurrence_graph(args.annotations, tags, min_count=args.min_cooccurrence, top_k=args.cooccurrence_top_k)
    output = Path(args.output)
    ensure_dir(output.parent)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Ontology saved to %s | tags=%d | mapped=%d", output, len(tags), len(ontology.tag_to_canonical))
    return 0


def make_local_crops(image: Any, mode: str = "grid", max_crops: int = 10) -> List[Tuple[str, Any]]:
    if Image is None:
        raise ImportError("Pillow is required")
    image = image.convert("RGB")
    w, h = image.size
    crops: List[Tuple[str, Any]] = [("global", image)]
    if mode == "none" or max_crops <= 1:
        return crops[:max_crops]
    # Center crop preserves central subjects.
    side = int(min(w, h) * 0.72)
    if side > 8:
        left = max((w - side) // 2, 0)
        top = max((h - side) // 2, 0)
        crops.append(("center", image.crop((left, top, left + side, top + side))))
    if mode in {"grid", "strong"}:
        # 2x2 crops for local details.
        cw = max(w // 2, 1)
        ch = max(h // 2, 1)
        boxes = [
            (0, 0, cw, ch),
            (w - cw, 0, w, ch),
            (0, h - ch, cw, h),
            (w - cw, h - ch, w, h),
        ]
        names = ["top_left", "top_right", "bottom_left", "bottom_right"]
        for name, box in zip(names, boxes):
            crops.append((name, image.crop(box)))
    if mode == "strong":
        # Add three horizontal and vertical strips. Useful for clothes, legs, face/hair.
        for i in range(3):
            y0 = int(h * i / 3)
            y1 = int(h * (i + 1) / 3)
            crops.append((f"hstrip_{i}", image.crop((0, y0, w, y1))))
            x0 = int(w * i / 3)
            x1 = int(w * (i + 1) / 3)
            crops.append((f"vstrip_{i}", image.crop((x0, 0, x1, h))))
    return crops[: max(1, max_crops)]


class OpenVocabularyTagger:
    """CLIP/SigLIP-style zero-shot tag scorer using Hugging Face transformers.

    It is intentionally optional. The supervised model still works without transformers; open-vocab
    inference requires: pip install transformers accelerate safetensors sentencepiece
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-384",
        device: Optional[Union[str, torch.device]] = None,
        dtype: str = "auto",
        prompt_prefix: str = "anime image with",
        batch_size: int = 64,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required for open-vocabulary inference. Install: pip install transformers")
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if dtype == "fp16" and self.device.type == "cuda":
            torch_dtype = torch.float16
        elif dtype == "bf16" and self.device.type == "cuda":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = None
        self.processor = AutoProcessor.from_pretrained(model_name)
        kwargs: Dict[str, Any] = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self.model = AutoModel.from_pretrained(model_name, **kwargs).to(self.device).eval()
        self.prompt_prefix = prompt_prefix
        self.batch_size = max(1, int(batch_size))
        self._text_cache: Dict[str, Tensor] = {}
        logger.info("Loaded open-vocabulary VLM: %s on %s", model_name, self.device)

    def _model_dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32

    @torch.inference_mode()
    def encode_texts(self, tags: Sequence[str]) -> Tensor:
        prompts = [tag_to_prompt(tag, self.prompt_prefix) for tag in tags]
        cache_key = "\n".join(prompts)
        if cache_key in self._text_cache:
            return self._text_cache[cache_key]
        features: List[Tensor] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            inputs = self.processor(text=batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            if hasattr(self.model, "get_text_features"):
                out = self.model.get_text_features(**inputs)
            else:
                raw = self.model(**inputs)
                out = getattr(raw, "text_embeds", None) or getattr(raw, "pooler_output", None)
                if out is None:
                    raise RuntimeError("Could not obtain text features from model output")
            out = F.normalize(out.float(), dim=-1)
            features.append(out.cpu())
        result = torch.cat(features, dim=0)
        self._text_cache[cache_key] = result
        return result

    @torch.inference_mode()
    def encode_images(self, images: Sequence[Any]) -> Tensor:
        features: List[Tensor] = []
        for start in range(0, len(images), self.batch_size):
            batch = [img.convert("RGB") for img in images[start : start + self.batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            if hasattr(self.model, "get_image_features"):
                out = self.model.get_image_features(**inputs)
            else:
                raw = self.model(**inputs)
                out = getattr(raw, "image_embeds", None) or getattr(raw, "pooler_output", None)
                if out is None:
                    raise RuntimeError("Could not obtain image features from model output")
            out = F.normalize(out.float(), dim=-1)
            features.append(out.cpu())
        return torch.cat(features, dim=0)

    def _logit_scale(self) -> float:
        scale = getattr(self.model, "logit_scale", None)
        if scale is None:
            return 10.0
        try:
            value = float(scale.exp().detach().cpu().item())
        except Exception:
            value = 10.0
        return max(1.0, min(value, 100.0))

    @torch.inference_mode()
    def score_image_tags(
        self,
        image: Any,
        tags: Sequence[str],
        local_mode: str = "grid",
        max_crops: int = 10,
        score_temperature: float = 1.0,
    ) -> List[Dict[str, Any]]:
        if not tags:
            return []
        crop_pairs = make_local_crops(image, mode=local_mode, max_crops=max_crops)
        crop_names = [name for name, _ in crop_pairs]
        crops = [crop for _, crop in crop_pairs]
        image_features = self.encode_images(crops)  # [num_crops, dim]
        text_features = self.encode_texts(tags)  # [num_tags, dim]
        logits = image_features @ text_features.T
        logits = logits * (self._logit_scale() / max(float(score_temperature), 1e-6))
        probs = torch.sigmoid(logits)
        max_probs, best_crop_indices = probs.max(dim=0)
        global_probs = probs[0]
        mean_probs = probs.mean(dim=0)
        results: List[Dict[str, Any]] = []
        for idx, tag in enumerate(tags):
            best_crop = int(best_crop_indices[idx].item())
            local_score = float(max_probs[idx].item())
            global_score = float(global_probs[idx].item())
            # Mixed score: global keeps semantics stable; local recovers small details.
            score = 0.65 * local_score + 0.25 * global_score + 0.10 * float(mean_probs[idx].item())
            results.append(
                {
                    "tag": str(tag),
                    "score": float(score),
                    "open_vocab_score": float(score),
                    "global_score": global_score,
                    "local_score": local_score,
                    "best_region": crop_names[best_crop],
                    "category": category_for_semantic_tag(str(tag)),
                    "source": "open_vocab_vlm",
                }
            )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


def _load_pil_image(path: Union[str, Path]) -> Any:
    if Image is None:
        raise ImportError("Pillow is required. Install it with: pip install pillow")
    with Image.open(path) as img:
        return img.convert("RGB")


def prepare_open_vocab_candidates(
    trained_tags: Sequence[str],
    candidate_tags_path: Optional[str] = None,
    include_trained_tags: bool = True,
    include_builtin_semantics: bool = True,
    semantic_categories: Optional[str] = None,
    max_candidates: int = 4096,
) -> List[str]:
    candidates: List[str] = []
    if include_trained_tags:
        candidates.extend(str(x) for x in trained_tags)
    candidates.extend(load_tag_list(candidate_tags_path))
    if include_builtin_semantics:
        cats = [x.strip() for x in semantic_categories.split(",") if x.strip()] if semantic_categories else None
        candidates.extend(builtin_semantic_tags(cats))
    seen: set[str] = set()
    out: List[str] = []
    for tag in candidates:
        key = normalize_tag_text(tag)
        if key and key not in seen:
            seen.add(key)
            out.append(str(tag))
        if len(out) >= max_candidates:
            break
    return out


def fuse_closed_and_open_predictions(
    closed_items: Sequence[Dict[str, Any]],
    open_items: Sequence[Dict[str, Any]],
    ontology: Optional[SemanticTagOntology] = None,
    closed_weight: float = 0.62,
    open_weight: float = 0.38,
    min_score: float = 0.25,
    top_k: int = 50,
) -> List[Dict[str, Any]]:
    combined: Dict[str, Dict[str, Any]] = {}
    for item in closed_items:
        tag = str(item.get("tag", ""))
        if not tag:
            continue
        canonical = ontology.canonical(tag) if ontology is not None else tag
        score = float(item.get("probability", item.get("score", 0.0)))
        entry = combined.setdefault(canonical, {"tag": canonical, "score": 0.0, "closed_score": 0.0, "open_vocab_score": 0.0, "sources": [], "original_tags": []})
        entry["closed_score"] = max(float(entry.get("closed_score", 0.0)), score)
        entry["score"] = max(float(entry.get("score", 0.0)), closed_weight * score)
        entry["sources"].append("closed_classifier")
        entry["original_tags"].append(tag)
    for item in open_items:
        tag = str(item.get("tag", ""))
        if not tag:
            continue
        canonical = ontology.canonical(tag) if ontology is not None else tag
        score = float(item.get("score", item.get("open_vocab_score", 0.0)))
        entry = combined.setdefault(canonical, {"tag": canonical, "score": 0.0, "closed_score": 0.0, "open_vocab_score": 0.0, "sources": [], "original_tags": []})
        entry["open_vocab_score"] = max(float(entry.get("open_vocab_score", 0.0)), score)
        entry["score"] = max(float(entry.get("score", 0.0)), open_weight * score + closed_weight * float(entry.get("closed_score", 0.0)))
        entry["sources"].append("open_vocab_vlm")
        entry["original_tags"].append(tag)
        if "best_region" in item and ("best_region" not in entry or score >= entry.get("open_vocab_score", 0.0)):
            entry["best_region"] = item.get("best_region")
        if "category" in item:
            entry["category"] = item.get("category")
    result: List[Dict[str, Any]] = []
    for item in combined.values():
        item["sources"] = sorted(set(item.get("sources", [])))
        item["original_tags"] = sorted(set(item.get("original_tags", [])))
        if float(item.get("score", 0.0)) >= min_score:
            result.append(item)
    result.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return result[:top_k]


def summarize_hidden_structure(items: Sequence[Dict[str, Any]], per_category: int = 8) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        category = str(item.get("category", "open_vocab"))
        buckets.setdefault(category, []).append(item)
    summary: Dict[str, List[Dict[str, Any]]] = {}
    for category, values in buckets.items():
        summary[category] = sorted(values, key=lambda x: float(x.get("score", x.get("open_vocab_score", 0.0))), reverse=True)[:per_category]
    return summary


@torch.inference_mode()
def closed_set_predict_for_image(
    model: Optional[nn.Module],
    image_path: Path,
    tags: Sequence[str],
    device: torch.device,
    input_size: int,
    resize_mode: str,
    threshold: float,
    top_k: int,
    thresholds: Optional[Tensor] = None,
    tta: str = "none",
    channels_last: bool = False,
) -> Dict[str, Any]:
    if model is None:
        return {"selected": [], "top_k": []}
    image_tensor = load_image_tensor(image_path, input_size, resize_mode=resize_mode).unsqueeze(0).to(device)
    if channels_last:
        image_tensor = image_tensor.contiguous(memory_format=torch.channels_last)
    probs = _probabilities_with_tta(model, image_tensor, tta=tta)
    return decode_predictions(probs, tags, threshold=threshold, top_k=top_k, thresholds=thresholds)[0]


def semantic_infer_main(args: argparse.Namespace) -> int:
    device = get_device(args.device)
    checkpoint: Optional[Dict[str, Any]] = None
    closed_model: Optional[nn.Module] = None
    trained_tags: List[str] = []
    input_size = int(args.input_size or 384)
    resize_mode = args.resize_mode or "pad"
    thresholds: Optional[Tensor] = None

    if args.checkpoint:
        checkpoint = load_checkpoint_metadata(args.checkpoint, device)
        trained_tags = [str(x) for x in (load_tags(args.tags_path) if args.tags_path else checkpoint.get("tags", []))]
        if not trained_tags:
            raise ValueError("Checkpoint has no tags. Pass --tags_path.")
        input_size = int(args.input_size or checkpoint.get("input_size", 384))
        resize_mode = args.resize_mode or checkpoint.get("resize_mode", "pad")
        if resize_mode == "crop":
            resize_mode = "pad"
        if not args.ignore_checkpoint_thresholds and "thresholds" in checkpoint:
            thresholds = torch.tensor(checkpoint["thresholds"], dtype=torch.float32)
        closed_model = load_model_from_checkpoint(args.checkpoint, device=device, num_classes=len(trained_tags), use_ema=args.use_ema).eval()
        if args.channels_last and device.type == "cuda":
            closed_model = closed_model.to(memory_format=torch.channels_last)
        if args.compile and hasattr(torch, "compile"):
            closed_model = torch.compile(closed_model)  # type: ignore[assignment]

    candidates = prepare_open_vocab_candidates(
        trained_tags=trained_tags,
        candidate_tags_path=args.candidate_tags,
        include_trained_tags=not args.exclude_trained_from_open_vocab,
        include_builtin_semantics=args.include_builtin_semantics,
        semantic_categories=args.semantic_categories,
        max_candidates=args.max_open_vocab_candidates,
    )
    if not candidates and closed_model is None:
        raise ValueError("No checkpoint and no open-vocabulary candidates. Pass --candidate_tags or enable builtin semantics.")

    ontology = SemanticTagOntology.load(args.ontology_path) if args.ontology_path else SemanticTagOntology.build(candidates + trained_tags, alias_path=args.alias_path, fuzzy_threshold=args.fuzzy_threshold)

    tagger: Optional[OpenVocabularyTagger] = None
    if candidates:
        tagger = OpenVocabularyTagger(
            model_name=args.open_vocab_model,
            device=device,
            dtype=args.open_vocab_dtype,
            prompt_prefix=args.prompt_prefix,
            batch_size=args.open_vocab_batch_size,
        )

    image_paths = collect_image_paths(args.input, recursive=not args.no_recursive)
    logger.info("Semantic inference images=%d | closed_tags=%d | open_candidates=%d", len(image_paths), len(trained_tags), len(candidates))
    results: List[Dict[str, Any]] = []
    for idx, image_path in enumerate(image_paths, start=1):
        closed_decoded = closed_set_predict_for_image(
            closed_model,
            image_path,
            trained_tags,
            device=device,
            input_size=input_size,
            resize_mode=resize_mode,
            threshold=args.threshold,
            top_k=args.closed_top_k,
            thresholds=thresholds,
            tta=args.tta,
            channels_last=args.channels_last,
        )
        open_items: List[Dict[str, Any]] = []
        if tagger is not None:
            image = _load_pil_image(image_path)
            scored = tagger.score_image_tags(
                image,
                candidates,
                local_mode=args.local_crops,
                max_crops=args.max_crops,
                score_temperature=args.open_vocab_temperature,
            )
            open_items = [item for item in scored if float(item["score"]) >= args.open_vocab_threshold][: args.open_vocab_top_k]
            open_items = ontology.canonicalize_items(open_items, score_key="score")
        fused = fuse_closed_and_open_predictions(
            closed_decoded.get("selected", []),
            open_items,
            ontology=ontology,
            closed_weight=args.closed_weight,
            open_weight=args.open_weight,
            min_score=args.fused_threshold,
            top_k=args.top_k,
        )
        hidden = summarize_hidden_structure(open_items, per_category=args.hidden_per_category)
        results.append(
            {
                "image": str(image_path),
                "fused_tags": fused,
                "closed_selected": ontology.canonicalize_items(closed_decoded.get("selected", []), score_key="probability"),
                "closed_top_k": closed_decoded.get("top_k", []),
                "open_vocab_selected": open_items,
                "hidden_structure": hidden,
                "metadata": {
                    "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                    "open_vocab_model": args.open_vocab_model if tagger is not None else None,
                    "candidate_count": len(candidates),
                    "local_crops": args.local_crops,
                },
            }
        )
        if args.log_interval > 0 and idx % args.log_interval == 0:
            logger.info("Processed %d/%d images", idx, len(image_paths))

    if args.output:
        write_semantic_results(results, args.output)
        logger.info("Semantic inference results saved to %s", args.output)
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def write_semantic_results(results: Sequence[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    if output_path.suffix.lower() == ".csv":
        with output_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "fused_tags", "open_vocab_tags", "hidden_structure_json", "full_json"])
            writer.writeheader()
            for item in results:
                writer.writerow(
                    {
                        "image": item["image"],
                        "fused_tags": " ".join(tag_item["tag"] for tag_item in item.get("fused_tags", [])),
                        "open_vocab_tags": " ".join(tag_item["tag"] for tag_item in item.get("open_vocab_selected", [])),
                        "hidden_structure_json": json.dumps(item.get("hidden_structure", {}), ensure_ascii=False),
                        "full_json": json.dumps(item, ensure_ascii=False),
                    }
                )
    else:
        output_path.write_text(json.dumps(list(results), ensure_ascii=False, indent=2), encoding="utf-8")


def pseudo_label_main(args: argparse.Namespace) -> int:
    """Create an augmented annotation file using the open-vocabulary branch.

    This addresses missing color/clothing/action/style labels in the training data. It does not
    overwrite the source annotation; it adds high-confidence pseudo labels to a new file.
    """
    if Image is None:
        raise ImportError("Pillow is required")
    rows = read_annotations(args.annotations)
    image_dir = Path(args.image_dir) if args.image_dir else Path(args.annotations).parent
    base_tags = load_tags(args.tags_path) if args.tags_path else build_tags_from_annotations(args.annotations)
    candidates = prepare_open_vocab_candidates(
        trained_tags=base_tags,
        candidate_tags_path=args.candidate_tags,
        include_trained_tags=args.include_trained_tags,
        include_builtin_semantics=args.include_builtin_semantics,
        semantic_categories=args.semantic_categories,
        max_candidates=args.max_open_vocab_candidates,
    )
    ontology = SemanticTagOntology.load(args.ontology_path) if args.ontology_path else SemanticTagOntology.build(base_tags + candidates, alias_path=args.alias_path)
    tagger = OpenVocabularyTagger(
        model_name=args.open_vocab_model,
        device=get_device(args.device),
        dtype=args.open_vocab_dtype,
        prompt_prefix=args.prompt_prefix,
        batch_size=args.open_vocab_batch_size,
    )
    output_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        image_key = _find_image_key(row)
        label_key = _find_label_key(row) or "labels"
        image_path = Path(str(row[image_key]))
        if not image_path.is_absolute():
            candidate = image_dir / image_path
            image_path = candidate if candidate.exists() else Path(args.annotations).parent / image_path
        if not image_path.exists():
            if args.allow_missing_images:
                continue
            raise FileNotFoundError(f"Missing image: {image_path}")
        original_labels_raw = parse_labels(row.get(label_key, []))
        original_labels = [str(x) for x in original_labels_raw] if not isinstance(original_labels_raw, Tensor) else []
        image = _load_pil_image(image_path)
        scored = tagger.score_image_tags(image, candidates, local_mode=args.local_crops, max_crops=args.max_crops, score_temperature=args.open_vocab_temperature)
        pseudo = [ontology.canonical(item["tag"]) for item in scored if float(item["score"]) >= args.threshold][: args.max_pseudo_labels]
        merged = sorted(set([ontology.canonical(x) for x in original_labels] + pseudo))
        new_row = dict(row)
        new_row[label_key] = " ".join(merged)
        new_row["pseudo_labels"] = " ".join(sorted(set(pseudo)))
        output_rows.append(new_row)
        if args.log_interval > 0 and idx % args.log_interval == 0:
            logger.info("Pseudo-labeled %d/%d", idx, len(rows))
    out = Path(args.output)
    ensure_dir(out.parent)
    if out.suffix.lower() == ".jsonl":
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in output_rows) + "\n", encoding="utf-8")
    elif out.suffix.lower() == ".json":
        out.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        fieldnames = sorted(set(k for row in output_rows for k in row.keys()))
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
    logger.info("Augmented annotations saved to %s", out)
    return 0

def infer_main(args: argparse.Namespace) -> int:
    device = get_device(args.device)
    checkpoint = load_checkpoint_metadata(args.checkpoint, device)
    checkpoint_tags = checkpoint.get("tags")
    tags = load_tags(args.tags_path) if args.tags_path else checkpoint_tags
    if not tags:
        raise ValueError("No tags found in checkpoint. Pass --tags_path.")
    tags = [str(tag) for tag in tags]
    input_size = int(args.input_size or checkpoint.get("input_size", 384))
    resize_mode = args.resize_mode or checkpoint.get("resize_mode", "pad")
    if resize_mode == "crop":
        resize_mode = "pad"
    thresholds = None
    if not args.ignore_checkpoint_thresholds and "thresholds" in checkpoint:
        thresholds = torch.tensor(checkpoint["thresholds"], dtype=torch.float32)
        logger.info("Using %d calibrated thresholds from checkpoint", thresholds.numel())

    model = load_model_from_checkpoint(args.checkpoint, device=device, num_classes=len(tags), use_ema=args.use_ema)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    image_paths = collect_image_paths(args.input, recursive=not args.no_recursive)
    logger.info("Loaded %d image(s) for inference", len(image_paths))
    results = infer_images(
        model=model,
        image_paths=image_paths,
        tags=tags,
        device=device,
        input_size=input_size,
        batch_size=args.batch_size,
        threshold=args.threshold,
        top_k=args.top_k,
        resize_mode=resize_mode,
        thresholds=thresholds,
        tta=args.tta,
        channels_last=args.channels_last,
    )
    if args.output:
        write_inference_results(results, args.output)
        logger.info("Inference results saved to %s", args.output)
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


# -------------------- Export and tests --------------------
def export_onnx_main(args: argparse.Namespace) -> int:
    device = get_device(args.device)
    checkpoint = load_checkpoint_metadata(args.checkpoint, device)
    tags = checkpoint.get("tags")
    num_classes = len(tags) if tags else int(checkpoint.get("num_classes", 1000))
    input_size = int(args.input_size or checkpoint.get("input_size", 384))
    model = load_model_from_checkpoint(args.checkpoint, device=device, num_classes=num_classes, use_ema=args.use_ema).eval()

    class OnnxWrapper(nn.Module):
        def __init__(self, inner: nn.Module):
            super().__init__()
            self.inner = inner

        def forward(self, images: Tensor) -> Tensor:
            return self.inner(images)["probabilities"]

    wrapper = OnnxWrapper(model).to(device).eval()
    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["images"],
        output_names=["probabilities"],
        opset_version=args.opset,
        dynamic_axes={"images": {0: "batch"}, "probabilities": {0: "batch"}},
    )
    logger.info("ONNX exported to %s", output_path)
    return 0


def smoke_test_main(args: argparse.Namespace) -> int:
    device = get_device(args.device)
    model_kwargs = model_kwargs_from_args(args)
    model = create_model(num_classes=args.num_classes, **model_kwargs).to(device)
    stats = model_summary(model, input_size=(3, args.input_size, args.input_size))
    logger.info("Model created successfully")
    logger.info("Backbone: %s", model.config.model_name)
    logger.info("Backbone mode: %s | Loss: %s", model.config.backbone_mode, model.config.loss_type)
    logger.info("Parameters: %s", f"{stats['total_params']:,}")
    images = torch.randn(2, 3, args.input_size, args.input_size, device=device)
    targets = torch.randint(0, 2, (2, args.num_classes), device=device).float()
    model.eval()
    with torch.inference_mode():
        output = model(images, targets)
        logger.info("Forward successful: logits=%s loss=%.4f", tuple(output["logits"].shape), float(output["loss"].item()))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    output = model(images, targets)
    output["loss"].backward()
    optimizer.step()
    logger.info("Backward/training step successful")
    return 0


def test_model_compatibility() -> None:
    configs = [
        {"model_name": "efficientnet_b0", "backbone_mode": "pool", "pretrained": False, "loss_type": "asl"},
        {"model_name": "efficientnet_b0", "backbone_mode": "features", "pretrained": False, "fusion_type": "attention"},
        {"model_name": "efficientnet_b0", "backbone_mode": "features", "pretrained": False, "fusion_type": "weighted"},
    ]
    for cfg in configs:
        try:
            model = create_model(num_classes=10, **cfg).eval()
            x = torch.randn(1, 3, 224, 224)
            with torch.inference_mode():
                y = model(x)
            assert y["logits"].shape == (1, 10)
            logger.info("✓ Config works: %s", cfg)
        except Exception as exc:
            logger.warning("✗ Config failed: %s | %s", cfg, exc)


# -------------------- CLI --------------------
def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_name", type=str, default=None, help="timm model name")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=None)
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--backbone_mode", type=str, default=None, choices=["pool", "features"])
    parser.add_argument("--out_indices", type=str, default=None, help="Feature mode out indices, e.g. 1,2,3,4")
    parser.add_argument("--feature_dim", type=int, default=None)
    parser.add_argument("--classifier_dropout", type=float, default=None)
    parser.add_argument("--projection_dropout", type=float, default=None)
    parser.add_argument("--fusion_type", type=str, default=None, choices=["attention", "weighted", "pyramid", "concat"])
    parser.add_argument("--attention_module", type=str, default=None, choices=["none", "se", "cbam"])
    parser.add_argument("--pool_type", type=str, default=None, choices=["avg", "gem"])
    parser.add_argument("--loss_type", type=str, default=None, choices=["asl", "bce", "focal", "smooth_bce"])
    parser.add_argument("--focal_alpha", type=float, default=None)
    parser.add_argument("--focal_gamma", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--asl_gamma_pos", type=float, default=None)
    parser.add_argument("--asl_gamma_neg", type=float, default=None)
    parser.add_argument("--asl_clip", type=float, default=None)


def model_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    valid_keys = {field.name for field in fields(ModelConfig)}
    data: Dict[str, Any] = {}
    for key in valid_keys:
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                data[key] = value
    if "out_indices" in data:
        data["out_indices"] = _tuple_from_any(data["out_indices"])
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Danbooru modern single-file training and inference project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a multi-label tag model")
    add_model_arguments(train_parser)
    train_parser.add_argument("--train_annotations", required=True)
    train_parser.add_argument("--val_annotations", default=None)
    train_parser.add_argument("--image_dir", default=None)
    train_parser.add_argument("--val_image_dir", default=None)
    train_parser.add_argument("--tags_path", default=None)
    train_parser.add_argument("--output_dir", default="runs/danbooru_modern")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch_size", type=int, default=16)
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--input_size", type=int, default=384)
    train_parser.add_argument("--resize_mode", type=str, default="crop", choices=["crop", "pad", "squash"])
    train_parser.add_argument("--val_resize_mode", type=str, default=None, choices=["crop", "pad", "squash"])
    train_parser.add_argument("--augmentation_policy", type=str, default="randaugment", choices=["randaugment", "trivialaugment", "augmix", "light", "none"])
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--backbone_lr_mult", type=float, default=0.1)
    train_parser.add_argument("--weight_decay", type=float, default=0.05)
    train_parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "onecycle", "step"])
    train_parser.add_argument("--threshold", type=float, default=0.5)
    train_parser.add_argument("--monitor", type=str, default="micro_f1")
    train_parser.add_argument("--compute_map", action="store_true")
    train_parser.add_argument("--calibrate_thresholds", action="store_true")
    train_parser.add_argument("--threshold_steps", type=int, default=41)
    train_parser.add_argument("--use_pos_weight", action="store_true")
    train_parser.add_argument("--pos_weight_max", type=float, default=20.0)
    train_parser.add_argument("--use_weighted_sampler", action="store_true")
    train_parser.add_argument("--freeze_backbone_epochs", type=int, default=0)
    train_parser.add_argument("--grad_clip", type=float, default=1.0)
    train_parser.add_argument("--grad_accum_steps", type=int, default=1)
    train_parser.add_argument("--amp", action="store_true")
    train_parser.add_argument("--channels_last", action="store_true")
    train_parser.add_argument("--compile", action="store_true")
    train_parser.add_argument("--ema", action="store_true")
    train_parser.add_argument("--ema_decay", type=float, default=0.9998)
    train_parser.add_argument("--validate_ema", action="store_true")
    train_parser.add_argument("--resume", default=None)
    train_parser.add_argument("--save_every", type=int, default=0)
    train_parser.add_argument("--log_interval", type=int, default=20)
    train_parser.add_argument("--allow_missing_images", action="store_true")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default=None)
    train_parser.set_defaults(func=train_main)

    infer_parser = subparsers.add_parser("infer", help="Run inference on one image or a folder")
    infer_parser.add_argument("--checkpoint", required=True)
    infer_parser.add_argument("--input", required=True)
    infer_parser.add_argument("--tags_path", default=None)
    infer_parser.add_argument("--output", default=None)
    infer_parser.add_argument("--input_size", type=int, default=None)
    infer_parser.add_argument("--resize_mode", type=str, default=None, choices=["crop", "pad", "squash"])
    infer_parser.add_argument("--batch_size", type=int, default=8)
    infer_parser.add_argument("--threshold", type=float, default=0.5)
    infer_parser.add_argument("--ignore_checkpoint_thresholds", action="store_true")
    infer_parser.add_argument("--top_k", type=int, default=20)
    infer_parser.add_argument("--tta", type=str, default="none", choices=["none", "hflip"])
    infer_parser.add_argument("--use_ema", action="store_true")
    infer_parser.add_argument("--channels_last", action="store_true")
    infer_parser.add_argument("--compile", action="store_true")
    infer_parser.add_argument("--no_recursive", action="store_true")
    infer_parser.add_argument("--device", default=None)
    infer_parser.set_defaults(func=infer_main)

    ontology_parser = subparsers.add_parser("build_ontology", help="Build semantic tag ontology/canonical mapping")
    ontology_parser.add_argument("--tags_path", default=None, help="Tag file. If omitted, --annotations is used")
    ontology_parser.add_argument("--annotations", default=None, help="Optional annotations for co-occurrence graph")
    ontology_parser.add_argument("--extra_candidate_tags", default=None, help="Extra candidate tags txt/json")
    ontology_parser.add_argument("--alias_path", default=None, help="Manual alias JSON: {canonical: [aliases]}")
    ontology_parser.add_argument("--output", required=True)
    ontology_parser.add_argument("--fuzzy_threshold", type=float, default=0.97)
    ontology_parser.add_argument("--strip_parentheses_for_alias", action="store_true")
    ontology_parser.add_argument("--include_builtin_semantics", action="store_true")
    ontology_parser.add_argument("--semantic_categories", default=None, help="Comma-separated built-in categories")
    ontology_parser.add_argument("--min_cooccurrence", type=int, default=3)
    ontology_parser.add_argument("--cooccurrence_top_k", type=int, default=20)
    ontology_parser.set_defaults(func=build_ontology_main)

    semantic_parser = subparsers.add_parser("semantic_infer", help="Hybrid closed-set + open-vocabulary semantic inference")
    semantic_parser.add_argument("--checkpoint", default=None, help="Optional trained closed-set checkpoint")
    semantic_parser.add_argument("--input", required=True)
    semantic_parser.add_argument("--output", default=None)
    semantic_parser.add_argument("--tags_path", default=None)
    semantic_parser.add_argument("--candidate_tags", default=None, help="Extra open-vocabulary candidate tags txt/json")
    semantic_parser.add_argument("--ontology_path", default=None)
    semantic_parser.add_argument("--alias_path", default=None)
    semantic_parser.add_argument("--open_vocab_model", default="google/siglip-base-patch16-384")
    semantic_parser.add_argument("--open_vocab_dtype", default="auto", choices=["auto", "fp16", "bf16"])
    semantic_parser.add_argument("--open_vocab_batch_size", type=int, default=64)
    semantic_parser.add_argument("--prompt_prefix", default="anime image with")
    semantic_parser.add_argument("--include_builtin_semantics", action="store_true", default=True)
    semantic_parser.add_argument("--no_builtin_semantics", dest="include_builtin_semantics", action="store_false")
    semantic_parser.add_argument("--semantic_categories", default=None, help="color,clothing,body_and_character,action,composition,style")
    semantic_parser.add_argument("--exclude_trained_from_open_vocab", action="store_true")
    semantic_parser.add_argument("--max_open_vocab_candidates", type=int, default=4096)
    semantic_parser.add_argument("--input_size", type=int, default=None)
    semantic_parser.add_argument("--resize_mode", type=str, default=None, choices=["crop", "pad", "squash"])
    semantic_parser.add_argument("--batch_size", type=int, default=1, help="Kept for compatibility; VLM uses open_vocab_batch_size")
    semantic_parser.add_argument("--threshold", type=float, default=0.5, help="Closed-set threshold")
    semantic_parser.add_argument("--ignore_checkpoint_thresholds", action="store_true")
    semantic_parser.add_argument("--closed_top_k", type=int, default=50)
    semantic_parser.add_argument("--open_vocab_threshold", type=float, default=0.28)
    semantic_parser.add_argument("--open_vocab_top_k", type=int, default=80)
    semantic_parser.add_argument("--open_vocab_temperature", type=float, default=1.0)
    semantic_parser.add_argument("--fused_threshold", type=float, default=0.22)
    semantic_parser.add_argument("--top_k", type=int, default=80)
    semantic_parser.add_argument("--closed_weight", type=float, default=0.62)
    semantic_parser.add_argument("--open_weight", type=float, default=0.38)
    semantic_parser.add_argument("--local_crops", type=str, default="grid", choices=["none", "grid", "strong"])
    semantic_parser.add_argument("--max_crops", type=int, default=10)
    semantic_parser.add_argument("--hidden_per_category", type=int, default=8)
    semantic_parser.add_argument("--fuzzy_threshold", type=float, default=0.97)
    semantic_parser.add_argument("--tta", type=str, default="none", choices=["none", "hflip"])
    semantic_parser.add_argument("--use_ema", action="store_true")
    semantic_parser.add_argument("--channels_last", action="store_true")
    semantic_parser.add_argument("--compile", action="store_true")
    semantic_parser.add_argument("--no_recursive", action="store_true")
    semantic_parser.add_argument("--log_interval", type=int, default=20)
    semantic_parser.add_argument("--device", default=None)
    semantic_parser.set_defaults(func=semantic_infer_main)

    pseudo_parser = subparsers.add_parser("pseudo_label", help="Create augmented annotations with semantic open-vocab pseudo labels")
    pseudo_parser.add_argument("--annotations", required=True)
    pseudo_parser.add_argument("--image_dir", default=None)
    pseudo_parser.add_argument("--output", required=True)
    pseudo_parser.add_argument("--tags_path", default=None)
    pseudo_parser.add_argument("--candidate_tags", default=None)
    pseudo_parser.add_argument("--ontology_path", default=None)
    pseudo_parser.add_argument("--alias_path", default=None)
    pseudo_parser.add_argument("--open_vocab_model", default="google/siglip-base-patch16-384")
    pseudo_parser.add_argument("--open_vocab_dtype", default="auto", choices=["auto", "fp16", "bf16"])
    pseudo_parser.add_argument("--open_vocab_batch_size", type=int, default=64)
    pseudo_parser.add_argument("--prompt_prefix", default="anime image with")
    pseudo_parser.add_argument("--include_builtin_semantics", action="store_true", default=True)
    pseudo_parser.add_argument("--no_builtin_semantics", dest="include_builtin_semantics", action="store_false")
    pseudo_parser.add_argument("--include_trained_tags", action="store_true")
    pseudo_parser.add_argument("--semantic_categories", default=None)
    pseudo_parser.add_argument("--max_open_vocab_candidates", type=int, default=4096)
    pseudo_parser.add_argument("--threshold", type=float, default=0.36)
    pseudo_parser.add_argument("--max_pseudo_labels", type=int, default=40)
    pseudo_parser.add_argument("--local_crops", type=str, default="grid", choices=["none", "grid", "strong"])
    pseudo_parser.add_argument("--max_crops", type=int, default=8)
    pseudo_parser.add_argument("--open_vocab_temperature", type=float, default=1.0)
    pseudo_parser.add_argument("--allow_missing_images", action="store_true")
    pseudo_parser.add_argument("--log_interval", type=int, default=20)
    pseudo_parser.add_argument("--device", default=None)
    pseudo_parser.set_defaults(func=pseudo_label_main)

    export_parser = subparsers.add_parser("export_onnx", help="Export checkpoint to ONNX")
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--input_size", type=int, default=None)
    export_parser.add_argument("--opset", type=int, default=17)
    export_parser.add_argument("--use_ema", action="store_true")
    export_parser.add_argument("--device", default=None)
    export_parser.set_defaults(func=export_onnx_main)

    test_parser = subparsers.add_parser("test", help="Run smoke test")
    add_model_arguments(test_parser)
    test_parser.add_argument("--num_classes", type=int, default=1000)
    test_parser.add_argument("--input_size", type=int, default=224)
    test_parser.add_argument("--device", default=None)
    test_parser.set_defaults(func=smoke_test_main)

    compat_parser = subparsers.add_parser("compatibility_test", help="Test several configs")
    compat_parser.set_defaults(func=lambda args: (test_model_compatibility() or 0))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except Exception as exc:
        logger.error("Error: %s", exc)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
