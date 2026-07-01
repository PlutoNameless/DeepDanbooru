
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

try:
    from PIL import Image
except Exception:  # pragma: no cover - environment dependent
    Image = None

# torchvision is optional and is only used for data transforms.
try:
    import torchvision.transforms as transforms

    TORCHVISION_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    TORCHVISION_AVAILABLE = False
    transforms = None

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# timm is required for the EfficientNet feature extractor.
try:
    import timm

    USE_TIMM = True
    logger.info("Using timm version: %s", getattr(timm, "__version__", "unknown"))
except Exception:  # pragma: no cover - environment dependent
    USE_TIMM = False
    timm = None
    logger.warning("timm not found. Install it with: pip install timm")


# -------------------- Configuration --------------------
@dataclass
class ModelConfig:
    model_name: str = "efficientnet_b4"
    pretrained: bool = True
    use_focal_loss: bool = False
    use_label_smoothing: bool = False
    fusion_type: str = "attention"
    use_cbam: bool = True
    classifier_dropout: float = 0.5
    feature_dim: int = 512
    projection_dropout: float = 0.2
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1


def _coerce_config(config: Optional[Union[ModelConfig, argparse.Namespace, Dict[str, Any]]] = None) -> ModelConfig:
    """Convert old Namespace/dict configs to ModelConfig while ignoring unknown keys."""
    if config is None:
        return ModelConfig()
    if isinstance(config, ModelConfig):
        return config

    valid_keys = {f.name for f in fields(ModelConfig)}
    if isinstance(config, argparse.Namespace):
        data = vars(config)
    elif isinstance(config, dict):
        data = config
    else:
        raise TypeError(f"Unsupported config type: {type(config)!r}")

    clean_data = {key: value for key, value in data.items() if key in valid_keys}
    return ModelConfig(**clean_data)


def _config_to_dict(config: Union[ModelConfig, argparse.Namespace, Dict[str, Any]]) -> Dict[str, Any]:
    return asdict(_coerce_config(config))


# -------------------- Basic modules --------------------
class GeM(nn.Module):
    """Generalized Mean Pooling."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * float(p))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"GeM expects a 4D tensor [B, C, H, W], got shape {tuple(x.shape)}")
        p = self.p.clamp(min=self.eps)
        x = x.clamp(min=self.eps).pow(p)
        x = F.avg_pool2d(x, kernel_size=(x.size(-2), x.size(-1)))
        return x.pow(1.0 / p)


class ChannelAttention(nn.Module):
    """Channel attention module."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        reduced_channels = max(in_channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = torch.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial attention module."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("SpatialAttention kernel_size should be 3 or 7")
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_map = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(attention_map))
        return x * attention


class CBAM(nn.Module):
    """CBAM attention module."""

    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.spatial_attention(self.channel_attention(x))


class FeatureProjection(nn.Module):
    """Project a feature map to a unified feature vector."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 512,
        use_gem: bool = True,
        use_cbam: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.attention = CBAM(in_channels) if use_cbam else ChannelAttention(in_channels)
        self.pool = GeM(p=3.0) if use_gem else nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.attention(x)
        x = self.pool(x)
        return self.projection(x)


# -------------------- Backbone --------------------
class EfficientNetBackbone(nn.Module):
    """EfficientNet backbone with multi-scale feature output."""

    DEFAULT_OUT_INDICES: Tuple[int, int, int, int] = (1, 2, 3, 4)

    def __init__(
        self,
        model_name: str = "efficientnet_b4",
        pretrained: bool = True,
        use_cbam: bool = True,
        feature_dim: int = 512,
        projection_dropout: float = 0.2,
        out_indices: Sequence[int] = DEFAULT_OUT_INDICES,
    ):
        super().__init__()
        self.model_name = model_name
        self.requested_pretrained = pretrained
        self.loaded_pretrained = False
        self.use_cbam = use_cbam
        self.feature_dim = feature_dim
        self.out_indices = tuple(out_indices)

        if not USE_TIMM:
            raise ImportError("timm is required for EfficientNetBackbone. Install it with: pip install timm")

        self.backbone = self._create_backbone(model_name=model_name, pretrained=pretrained)
        self.feature_channels = self._get_feature_channels()
        if len(self.feature_channels) != len(self.out_indices):
            raise RuntimeError(
                f"Expected {len(self.out_indices)} feature scales, but got channels {self.feature_channels}. "
                "Check timm feature_info/out_indices compatibility."
            )

        logger.info("Loaded backbone: %s | pretrained=%s", self.model_name, self.loaded_pretrained)
        logger.info("Feature channels: %s", self.feature_channels)

        self.projections = nn.ModuleList(
            [
                FeatureProjection(
                    in_channels=channels,
                    out_channels=feature_dim,
                    use_gem=True,
                    use_cbam=use_cbam,
                    dropout=projection_dropout,
                )
                for channels in self.feature_channels
            ]
        )

    def _create_backbone(self, model_name: str, pretrained: bool) -> nn.Module:
        supported_models = [
            "tf_efficientnet_b4_ns",
            "efficientnet_b4",
            "efficientnet_b3",
            "efficientnet_b2",
            "efficientnet_b0",
        ]
        attempts = [model_name] + [name for name in supported_models if name != model_name]
        pretrained_options = [pretrained]
        if pretrained:
            pretrained_options.append(False)

        errors: List[str] = []
        for use_pretrained in pretrained_options:
            for attempt_model in attempts:
                try:
                    backbone = timm.create_model(
                        attempt_model,
                        pretrained=use_pretrained,
                        features_only=True,
                        out_indices=self.out_indices,
                    )
                    self.model_name = attempt_model
                    self.loaded_pretrained = bool(use_pretrained)
                    if pretrained and not use_pretrained:
                        logger.warning(
                            "Could not load pretrained weights. Falling back to randomly initialized %s.",
                            attempt_model,
                        )
                    return backbone
                except Exception as exc:  # pragma: no cover - depends on installed timm/models/cache
                    errors.append(f"{attempt_model}(pretrained={use_pretrained}): {exc}")

        raise RuntimeError("Failed to load an EfficientNet backbone. Attempts:\n" + "\n".join(errors[-10:]))

    def _get_feature_channels(self) -> List[int]:
        feature_info = getattr(self.backbone, "feature_info", None)
        if feature_info is None:
            raise RuntimeError("The timm backbone does not expose feature_info.")

        if hasattr(feature_info, "channels"):
            channels = list(feature_info.channels())
        elif hasattr(feature_info, "get_dicts"):
            channels = [int(info["num_chs"]) for info in feature_info.get_dicts()]
        else:
            channels = [int(info["num_chs"]) for info in feature_info]

        return channels

    def forward_raw_features(self, x: Tensor) -> List[Tensor]:
        raw_features = self.backbone(x)
        if len(raw_features) != len(self.projections):
            raise RuntimeError(
                f"Backbone returned {len(raw_features)} features, but {len(self.projections)} projections are configured."
            )
        return list(raw_features)

    def forward(self, x: Tensor, return_raw: bool = False) -> Union[Dict[str, Tensor], Tuple[Dict[str, Tensor], List[Tensor]]]:
        raw_features = self.forward_raw_features(x)
        projected_features = {
            f"scale_{idx}": projection(feature)
            for idx, (feature, projection) in enumerate(zip(raw_features, self.projections))
        }
        if return_raw:
            return projected_features, raw_features
        return projected_features


# -------------------- Multi-scale fusion --------------------
class MultiScaleFusion(nn.Module):
    """Multi-scale feature fusion."""

    def __init__(self, num_scales: int, feature_dim: int = 512, fusion_type: str = "attention"):
        super().__init__()
        if num_scales <= 0:
            raise ValueError("num_scales must be positive")
        if fusion_type not in {"attention", "weighted", "pyramid", "concat"}:
            raise ValueError("fusion_type must be one of: attention, weighted, pyramid, concat")

        self.feature_dim = feature_dim
        self.fusion_type = fusion_type
        self.num_scales = num_scales
        self._build_fusion_layers()

    def _build_fusion_layers(self) -> None:
        if self.fusion_type == "attention":
            if self.feature_dim % 8 != 0:
                raise ValueError("feature_dim must be divisible by 8 when using attention fusion")
            self.attention = nn.MultiheadAttention(
                embed_dim=self.feature_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(self.feature_dim)
            self.dropout = nn.Dropout(0.1)
        elif self.fusion_type == "weighted":
            self.scale_weights = nn.Parameter(torch.ones(self.num_scales))
            self.norm = nn.LayerNorm(self.feature_dim)
        elif self.fusion_type == "pyramid":
            self.pyramid_fusion = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.feature_dim * 2, self.feature_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.1),
                    )
                    for _ in range(self.num_scales - 1)
                ]
            )
            self.norm = nn.LayerNorm(self.feature_dim)
        else:  # concat
            self.fusion = nn.Sequential(
                nn.Linear(self.feature_dim * self.num_scales, self.feature_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(self.feature_dim * 2, self.feature_dim),
                nn.LayerNorm(self.feature_dim),
            )

    def _as_ordered_feature_list(self, features: Dict[str, Tensor]) -> List[Tensor]:
        missing = [f"scale_{idx}" for idx in range(self.num_scales) if f"scale_{idx}" not in features]
        if missing:
            raise KeyError(f"Missing feature keys: {missing}")
        feature_list = [features[f"scale_{idx}"] for idx in range(self.num_scales)]
        for idx, feature in enumerate(feature_list):
            if feature.ndim != 2 or feature.size(1) != self.feature_dim:
                raise ValueError(
                    f"features['scale_{idx}'] must have shape [B, {self.feature_dim}], got {tuple(feature.shape)}"
                )
        return feature_list

    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        feature_list = self._as_ordered_feature_list(features)

        if self.fusion_type == "attention":
            stacked = torch.stack(feature_list, dim=1)  # [B, num_scales, feature_dim]
            fused, _ = self.attention(stacked, stacked, stacked, need_weights=False)
            fused = self.dropout(fused)
            fused = self.norm(fused.mean(dim=1))
        elif self.fusion_type == "weighted":
            weights = torch.softmax(self.scale_weights, dim=0).to(feature_list[0].device, feature_list[0].dtype)
            fused = torch.stack([w * f for w, f in zip(weights, feature_list)], dim=0).sum(dim=0)
            fused = self.norm(fused)
        elif self.fusion_type == "pyramid":
            fused = feature_list[-1]
            # Fuse from deep features to shallow features.
            for idx in range(self.num_scales - 2, -1, -1):
                fused = self.pyramid_fusion[idx](torch.cat([fused, feature_list[idx]], dim=1))
            fused = self.norm(fused)
        else:  # concat
            fused = self.fusion(torch.cat(feature_list, dim=1))

        return fused


# -------------------- Loss functions --------------------
class FocalLoss(nn.Module):
    """Numerically stable multi-label focal loss for logits."""

    def __init__(
        self,
        alpha: Optional[Union[float, Tensor]] = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        pos_weight: Optional[Tensor] = None,
    ):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
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

        bce_loss = F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )

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
    """BCEWithLogitsLoss with symmetric label smoothing for multi-label targets."""

    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean",
        pos_weight: Optional[Tensor] = None,
    ):
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
        # Positive labels become 1 - smoothing/2; negative labels become smoothing/2.
        smoothed_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        pos_weight = self.pos_weight.to(device=inputs.device, dtype=inputs.dtype) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(
            inputs,
            smoothed_targets,
            pos_weight=pos_weight,
            reduction=self.reduction,
        )


# -------------------- Main model --------------------
class DeepDanbooruV3(nn.Module):
    def __init__(self, num_classes: int, config: Optional[Union[ModelConfig, argparse.Namespace, Dict[str, Any]]] = None):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.num_classes = int(num_classes)
        self.config = _coerce_config(config)
        self._validate_config()

        self.backbone = EfficientNetBackbone(
            model_name=self.config.model_name,
            pretrained=self.config.pretrained,
            use_cbam=self.config.use_cbam,
            feature_dim=self.config.feature_dim,
            projection_dropout=self.config.projection_dropout,
        )

        self.fusion = MultiScaleFusion(
            num_scales=len(self.backbone.feature_channels),
            feature_dim=self.config.feature_dim,
            fusion_type=self.config.fusion_type,
        )

        self.classifier = self._build_classifier(self.config.classifier_dropout)
        self._setup_loss_function()
        self._init_weights()

    @staticmethod
    def _get_default_config() -> ModelConfig:
        return ModelConfig()

    def _validate_config(self) -> None:
        if self.config.use_focal_loss and self.config.use_label_smoothing:
            raise ValueError("use_focal_loss and use_label_smoothing cannot both be True")
        if not 0.0 <= self.config.classifier_dropout < 1.0:
            raise ValueError("classifier_dropout must be in [0, 1)")
        if not 0.0 <= self.config.projection_dropout < 1.0:
            raise ValueError("projection_dropout must be in [0, 1)")

    def _build_classifier(self, dropout: float) -> nn.Module:
        dim = self.config.feature_dim
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.4),
            nn.Linear(dim, self.num_classes),
        )

    def _setup_loss_function(self) -> None:
        if self.config.use_focal_loss:
            self.criterion = FocalLoss(alpha=self.config.focal_alpha, gamma=self.config.focal_gamma)
        elif self.config.use_label_smoothing:
            self.criterion = LabelSmoothingBCE(smoothing=self.config.label_smoothing)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def _init_weights(self) -> None:
        def init_layer(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Do not re-initialize the pretrained timm backbone. Only initialize newly added heads.
        self.backbone.projections.apply(init_layer)
        self.fusion.apply(init_layer)
        self.classifier.apply(init_layer)

    def forward(self, x: Tensor, targets: Optional[Tensor] = None) -> Dict[str, Tensor]:
        features = self.backbone(x)
        fused_features = self.fusion(features)
        logits = self.classifier(fused_features)
        probabilities = torch.sigmoid(logits)

        output: Dict[str, Tensor] = {
            "logits": logits,
            "probabilities": probabilities,
            "features": fused_features,
        }

        if targets is not None:
            targets = targets.to(device=logits.device, dtype=logits.dtype)
            if targets.shape != logits.shape:
                raise ValueError(f"targets must have shape {tuple(logits.shape)}, got {tuple(targets.shape)}")
            output["loss"] = self.criterion(logits, targets)

        return output

    def predict(
        self,
        x: Tensor,
        threshold: float = 0.5,
        return_probs: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        was_training = self.training
        self.eval()
        with torch.inference_mode():
            probabilities = self.forward(x)["probabilities"]
            predictions = (probabilities >= threshold).to(probabilities.dtype)
        if was_training:
            self.train()
        return (predictions, probabilities) if return_probs else predictions

    def predict_top_k(self, x: Tensor, k: int = 5) -> Tuple[Tensor, Tensor]:
        if k <= 0:
            raise ValueError("k must be positive")
        k = min(k, self.num_classes)
        was_training = self.training
        self.eval()
        with torch.inference_mode():
            probabilities = self.forward(x)["probabilities"]
            top_probs, top_indices = torch.topk(probabilities, k=k, dim=1)
        if was_training:
            self.train()
        return top_indices, top_probs

    def get_feature_importance(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Return per-channel activation strength for each raw backbone feature scale.

        The uploaded version attempted mean(dim=(2, 3)) on already-projected 2D features.
        This method now uses raw 4D backbone feature maps, so it returns tensors shaped [B, C].
        """
        was_training = self.training
        self.eval()
        with torch.inference_mode():
            raw_features = self.backbone.forward_raw_features(x)
            importance = {
                f"scale_{idx}": feature.detach().abs().mean(dim=(2, 3))
                for idx, feature in enumerate(raw_features)
            }
        if was_training:
            self.train()
        return importance


# -------------------- Creation and helper functions --------------------
def create_model(num_classes: int, **kwargs: Any) -> DeepDanbooruV3:
    valid_keys = {f.name for f in fields(ModelConfig)}
    ignored_keys = sorted(set(kwargs) - valid_keys)
    if ignored_keys:
        logger.debug("Ignoring non-config keys in create_model: %s", ignored_keys)
    config_data = {key: value for key, value in kwargs.items() if key in valid_keys and value is not None}
    config = ModelConfig(**{**asdict(ModelConfig()), **config_data})
    return DeepDanbooruV3(num_classes=num_classes, config=config)


def model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)) -> Dict[str, Union[int, float, Tuple[int, int, int]]]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    param_size_mb = total_params * 4 / (1024**2)
    buffer_size_mb = sum(buffer.numel() for buffer in model.buffers()) * 4 / (1024**2)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": param_size_mb,
        "buffer_size_mb": buffer_size_mb,
        "total_size_mb": param_size_mb + buffer_size_mb,
        "input_size": input_size,
    }


def test_model_compatibility() -> None:
    logger.info("Testing model compatibility...")
    configs = [
        {"model_name": "efficientnet_b0", "fusion_type": "attention", "pretrained": False},
        {"model_name": "efficientnet_b0", "fusion_type": "weighted", "pretrained": False},
        {"model_name": "efficientnet_b0", "fusion_type": "pyramid", "pretrained": False},
        {"model_name": "efficientnet_b0", "fusion_type": "concat", "pretrained": False},
    ]
    for config in configs:
        try:
            model = create_model(num_classes=10, **config)
            model.eval()
            x = torch.randn(1, 3, 224, 224)
            with torch.inference_mode():
                output = model(x)
            assert output["logits"].shape == (1, 10)
            logger.info("✓ Config %s works", config)
        except Exception as exc:
            logger.warning("✗ Config %s failed: %s", config, exc)
    logger.info("Compatibility test completed")


def set_backbone_trainable(model: DeepDanbooruV3, trainable: bool) -> None:
    """Freeze or unfreeze only the timm backbone; projection/fusion/classifier remain unchanged."""
    for parameter in model.backbone.backbone.parameters():
        parameter.requires_grad = trainable


# -------------------- Utility classes --------------------
class ModelEnsemble(nn.Module):
    def __init__(self, models: List[DeepDanbooruV3], weights: Optional[List[float]] = None):
        super().__init__()
        if not models:
            raise ValueError("models cannot be empty")
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0] * len(models)
        if len(weights) != len(models):
            raise ValueError("weights length must match models length")
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        if torch.any(weight_tensor < 0) or float(weight_tensor.sum()) <= 0:
            raise ValueError("weights must be non-negative and have a positive sum")
        self.register_buffer("weights", weight_tensor / weight_tensor.sum())

    def forward(self, x: Tensor) -> Dict[str, Union[Tensor, List[Tensor]]]:
        predictions: List[Tensor] = []
        features_list: List[Tensor] = []
        for model in self.models:
            model.eval()
            with torch.inference_mode():
                output = model(x)
                predictions.append(output["probabilities"])
                features_list.append(output["features"])

        weights = self.weights.to(device=predictions[0].device, dtype=predictions[0].dtype)
        stacked_predictions = torch.stack(predictions, dim=0)
        weighted_pred = (weights.view(-1, 1, 1) * stacked_predictions).sum(dim=0)
        avg_features = torch.stack(features_list, dim=0).mean(dim=0)
        return {
            "probabilities": weighted_pred,
            "features": avg_features,
            "individual_predictions": predictions,
        }


class GradCAM:
    def __init__(self, model: DeepDanbooruV3, target_layer: Optional[str] = None):
        self.model = model
        self.gradients: Optional[Tensor] = None
        self.activations: Optional[Tensor] = None
        self.handles: List[Any] = []

        self.target_layer = target_layer or self._find_default_target_layer()
        self._register_hooks()

    def _find_default_target_layer(self) -> str:
        last_backbone_conv: Optional[str] = None
        last_any_conv: Optional[str] = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_any_conv = name
                if name.startswith("backbone.backbone"):
                    last_backbone_conv = name
        target_layer = last_backbone_conv or last_any_conv
        if target_layer is None:
            raise RuntimeError("No Conv2d layer found for GradCAM target_layer")
        return target_layer

    def _register_hooks(self) -> None:
        def backward_hook(module: nn.Module, grad_input: Tuple[Tensor, ...], grad_output: Tuple[Tensor, ...]) -> None:
            self.gradients = grad_output[0]

        def forward_hook(module: nn.Module, inputs: Tuple[Tensor, ...], output: Tensor) -> None:
            self.activations = output

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.handles.append(module.register_forward_hook(forward_hook))
                self.handles.append(module.register_full_backward_hook(backward_hook))
                return
        raise ValueError(f"Layer {self.target_layer!r} not found for GradCAM")

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def generate_cam(self, x: Tensor, target_class: int, batch_index: int = 0) -> Tensor:
        if target_class < 0 or target_class >= self.model.num_classes:
            raise ValueError(f"target_class must be in [0, {self.model.num_classes - 1}]")
        if batch_index < 0 or batch_index >= x.size(0):
            raise ValueError(f"batch_index must be in [0, {x.size(0) - 1}]")

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        output = self.model(x)
        target_score = output["logits"][batch_index, target_class]
        target_score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients/activations were not captured. Check target_layer.")
        if self.activations.ndim != 4 or self.gradients.ndim != 4:
            raise RuntimeError("GradCAM target_layer must output a 4D feature map [B, C, H, W].")

        gradients = self.gradients[batch_index]
        activations = self.activations[batch_index]
        weights = gradients.mean(dim=(1, 2))
        cam = (weights[:, None, None] * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach()


class DataAugmentation:
    @staticmethod
    def get_train_transforms(input_size: int = 224):
        if not TORCHVISION_AVAILABLE:
            logger.warning("torchvision is not available; returning None for train transforms")
            return None
        return transforms.Compose(
            [
                transforms.Resize((input_size + 32, input_size + 32)),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def get_val_transforms(input_size: int = 224):
        if not TORCHVISION_AVAILABLE:
            logger.warning("torchvision is not available; returning None for val transforms")
            return None
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def save_model(
    model: DeepDanbooruV3,
    filepath: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_config": _config_to_dict(model.config),
        "num_classes": model.num_classes,
        "model_name": model.backbone.model_name,
        "feature_channels": model.backbone.feature_channels,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if metrics is not None:
        checkpoint["metrics"] = metrics
    torch.save(checkpoint, filepath)
    logger.info("Model saved to %s", filepath)


def _safe_torch_load(filepath: str, device: Union[str, torch.device]) -> Dict[str, Any]:
    # PyTorch 2.6 changed torch.load behavior for some objects. This keeps compatibility with old checkpoints.
    try:
        return torch.load(filepath, map_location=device, weights_only=False)
    except TypeError:  # older PyTorch versions do not support weights_only
        return torch.load(filepath, map_location=device)


def load_model(filepath: str, num_classes: Optional[int] = None, device: Union[str, torch.device] = "cpu") -> DeepDanbooruV3:
    checkpoint = _safe_torch_load(filepath, device)
    config = _coerce_config(checkpoint.get("model_config", ModelConfig()))
    config = replace(config, pretrained=False)
    if "model_name" in checkpoint:
        config = replace(config, model_name=checkpoint["model_name"])
    classes = int(num_classes or checkpoint.get("num_classes", 1000))
    model = DeepDanbooruV3(classes, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model loaded from %s", filepath)
    return model


# -------------------- Dataset, metrics, training, inference --------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
LABEL_SEPARATORS = [",", ";", "|", "\t"]


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
    """Load tag names from txt/json/csv.

    txt: one tag per line
    json: list[str] or {"tags": [...]} or {"tag_to_idx": {...}}
    csv: first column named tag/name, or first column by default
    """
    path = Path(tags_path)
    if not path.exists():
        raise FileNotFoundError(f"Tags file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            tags = [str(item) for item in data]
        elif isinstance(data, dict) and "tags" in data:
            tags = [str(item) for item in data["tags"]]
        elif isinstance(data, dict) and "tag_to_idx" in data:
            mapping = {str(k): int(v) for k, v in data["tag_to_idx"].items()}
            tags = [tag for tag, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
        else:
            raise ValueError("Unsupported json tag file format")
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                tag_col = "tag" if "tag" in reader.fieldnames else "name" if "name" in reader.fieldnames else reader.fieldnames[0]
                tags = [row[tag_col].strip() for row in reader if row.get(tag_col, "").strip()]
            else:
                raise ValueError("CSV tag file must have a header")
    else:
        tags = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        tags = [tag for tag in tags if tag and not tag.startswith("#")]

    if not tags:
        raise ValueError(f"No tags found in {path}")
    if len(tags) != len(set(tags)):
        duplicates = sorted({tag for tag in tags if tags.count(tag) > 1})[:10]
        raise ValueError(f"Duplicate tags found: {duplicates}")
    return tags


def save_tags(tags: Sequence[str], path: Union[str, Path]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text("\n".join(str(tag) for tag in tags) + "\n", encoding="utf-8")


def _split_label_string(value: str) -> List[Union[str, int]]:
    value = value.strip()
    if not value:
        return []
    # Prefer explicit separators. If none exists, fall back to whitespace, which matches Danbooru-style tags.
    parts: List[str]
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
    """Parse labels from common annotation formats.

    Supported examples:
    - "blue_eyes,long_hair" or "blue_eyes long_hair"
    - ["blue_eyes", "long_hair"]
    - [2, 5, 10] as class indices
    - [0, 1, 0, 1] as a one-hot / multi-hot vector
    - {"blue_eyes": 1, "long_hair": true}
    """
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
        labels: List[str] = []
        for key, flag in value.items():
            if isinstance(flag, (int, float, bool)) and bool(flag):
                labels.append(str(key))
        return labels
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        if all(isinstance(item, (int, float, bool)) for item in value):
            # Keep numeric vectors/indices as integers. The dataset will disambiguate by length.
            return [int(item) for item in value]
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError(f"Unsupported label value type: {type(value)!r}")


def _find_image_key(row: Dict[str, Any]) -> str:
    candidates = ["image", "filename", "file", "path", "image_path", "filepath"]
    for key in candidates:
        if key in row and str(row[key]).strip():
            return key
    raise KeyError(f"Could not find an image path column. Tried: {candidates}")


def _find_label_key(row: Dict[str, Any]) -> Optional[str]:
    candidates = ["labels", "tags", "tag", "classes", "class", "target", "targets"]
    for key in candidates:
        if key in row:
            return key
    return None


def read_annotations(annotation_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read annotations from CSV, JSON, or JSONL.

    CSV examples:
        image,labels
        0001.jpg,"blue_eyes,long_hair"

    JSON examples:
        [{"image": "0001.jpg", "labels": ["blue_eyes", "long_hair"]}]

    JSONL examples:
        {"image": "0001.jpg", "labels": ["blue_eyes", "long_hair"]}
    """
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
        raise ValueError("All annotation entries must be objects/dicts")
    return rows


def build_tags_from_annotations(annotation_path: Union[str, Path]) -> List[str]:
    rows = read_annotations(annotation_path)
    tag_set = set()
    max_index = -1
    found_numeric_vector = False
    for row in rows:
        label_key = _find_label_key(row)
        if label_key is None:
            continue
        labels = parse_labels(row[label_key])
        if isinstance(labels, Tensor):
            found_numeric_vector = True
            max_index = max(max_index, int(labels.numel()) - 1)
        elif labels and all(isinstance(item, int) for item in labels):
            numeric = [int(item) for item in labels]
            # If it looks like a multi-hot vector, use its length; otherwise use max index.
            if set(numeric).issubset({0, 1}) and len(numeric) > 2:
                found_numeric_vector = True
                max_index = max(max_index, len(numeric) - 1)
            else:
                max_index = max(max_index, max(numeric))
        else:
            tag_set.update(str(item) for item in labels)

    if tag_set:
        return sorted(tag_set)
    if max_index >= 0 or found_numeric_vector:
        return [f"class_{idx}" for idx in range(max_index + 1)]
    raise ValueError("Could not build tags from annotations. Provide --tags_path explicitly.")


class DanbooruTagDataset(Dataset):
    """Multi-label image dataset for Danbooru-style tags."""

    def __init__(
        self,
        annotation_path: Union[str, Path],
        image_dir: Optional[Union[str, Path]] = None,
        tags: Optional[Sequence[str]] = None,
        tags_path: Optional[Union[str, Path]] = None,
        input_size: int = 224,
        is_train: bool = True,
        strict_images: bool = True,
    ):
        if Image is None:
            raise ImportError("Pillow is required for image loading. Install it with: pip install pillow")
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for dataset transforms. Install it with: pip install torchvision")

        self.annotation_path = Path(annotation_path)
        self.image_dir = Path(image_dir) if image_dir is not None else self.annotation_path.parent
        self.input_size = int(input_size)
        self.is_train = bool(is_train)
        self.strict_images = strict_images

        if tags_path is not None:
            self.tags = load_tags(tags_path)
        elif tags is not None:
            self.tags = [str(tag) for tag in tags]
        else:
            self.tags = build_tags_from_annotations(annotation_path)

        self.tag_to_idx = {tag: idx for idx, tag in enumerate(self.tags)}
        self.num_classes = len(self.tags)
        self.transform = DataAugmentation.get_train_transforms(input_size) if is_train else DataAugmentation.get_val_transforms(input_size)
        if self.transform is None:
            raise RuntimeError("Could not build torchvision transforms")

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
            preview = sorted(unknown_labels)[:20]
            logger.warning("Ignored %d labels not present in tag list. Examples: %s", len(unknown_labels), preview)

    def _resolve_image_path(self, image_value: str) -> Path:
        path = Path(image_value)
        if path.is_absolute():
            return path
        direct = self.image_dir / path
        if direct.exists():
            return direct
        # Also try relative to annotation file parent.
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
        tensor = self.transform(image)
        return {"image": tensor, "target": target.clone(), "path": str(image_path)}

    def label_counts(self) -> Tensor:
        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        for _, target in self.samples:
            counts += target
        return counts


def collate_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "target": torch.stack([item["target"] for item in batch], dim=0),
        "path": [item["path"] for item in batch],
    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=collate_batch,
        drop_last=False,
    )


def build_pos_weight(dataset: DanbooruTagDataset, clamp_max: float = 20.0) -> Tensor:
    positive = dataset.label_counts()
    total = float(len(dataset))
    negative = torch.clamp(torch.tensor(total) - positive, min=0.0)
    pos_weight = negative / torch.clamp(positive, min=1.0)
    return pos_weight.clamp(min=1.0, max=clamp_max)


def build_training_criterion(config: ModelConfig, pos_weight: Optional[Tensor] = None) -> nn.Module:
    if config.use_focal_loss:
        return FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma, pos_weight=pos_weight)
    if config.use_label_smoothing:
        return LabelSmoothingBCE(smoothing=config.label_smoothing, pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def compute_multilabel_metrics(logits: Tensor, targets: Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= threshold
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
    """Compute macro mAP on CPU. Suitable for validation, not for huge per-step usage."""
    scores = torch.sigmoid(logits).detach().float().cpu()
    labels = (targets.detach().float().cpu() >= 0.5).float()
    aps: List[float] = []
    for class_idx in range(labels.size(1)):
        y_true = labels[:, class_idx]
        positives = int(y_true.sum().item())
        if positives == 0:
            continue
        y_score = scores[:, class_idx]
        order = torch.argsort(y_score, descending=True)
        sorted_true = y_true[order]
        tp = torch.cumsum(sorted_true, dim=0)
        ranks = torch.arange(1, sorted_true.numel() + 1, dtype=torch.float32)
        precision_at_k = tp / ranks
        ap = (precision_at_k * sorted_true).sum() / max(positives, 1)
        aps.append(float(ap.item()))
    return float(sum(aps) / len(aps)) if aps else 0.0


def format_metrics(metrics: Dict[str, float]) -> str:
    keys = ["loss", "micro_f1", "macro_f1", "micro_precision", "micro_recall", "mAP", "accuracy", "exact_match"]
    parts = []
    for key in keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return " | ".join(parts)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Tuple[Tensor, Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    targets = batch["target"].to(device, non_blocking=True)
    return images, targets


def train_one_epoch(
    model: DeepDanbooruV3,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: Optional[Any] = None,
    amp: bool = False,
    grad_clip: Optional[float] = None,
    log_interval: int = 20,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    start_time = time.time()
    use_amp = amp and device.type == "cuda"

    for step, batch in enumerate(loader, start=1):
        images, targets = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(images, targets)
                loss = output["loss"]
            assert scaler is not None
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(images, targets)
            loss = output["loss"]
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
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
    model: DeepDanbooruV3,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    compute_map: bool = False,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    logits_list: List[Tensor] = []
    targets_list: List[Tensor] = []

    for batch in loader:
        images, targets = move_batch_to_device(batch, device)
        output = model(images, targets)
        loss = output["loss"]
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        logits_list.append(output["logits"].detach().cpu())
        targets_list.append(targets.detach().cpu())

    logits = torch.cat(logits_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    metrics = compute_multilabel_metrics(logits, targets, threshold=threshold)
    metrics["loss"] = total_loss / max(total_samples, 1)
    if compute_map:
        metrics["mAP"] = compute_mean_average_precision(logits, targets)
    return metrics


def save_training_checkpoint(
    filepath: Union[str, Path],
    model: DeepDanbooruV3,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    best_metric: float,
    tags: Sequence[str],
    input_size: int,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_config": _config_to_dict(model.config),
        "num_classes": model.num_classes,
        "model_name": model.backbone.model_name,
        "feature_channels": model.backbone.feature_channels,
        "epoch": epoch,
        "best_metric": best_metric,
        "tags": list(tags),
        "input_size": input_size,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    torch.save(checkpoint, filepath)
    logger.info("Checkpoint saved: %s", filepath)


def load_checkpoint_metadata(checkpoint_path: Union[str, Path], device: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    return _safe_torch_load(str(checkpoint_path), device)


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    num_classes: Optional[int] = None,
) -> DeepDanbooruV3:
    checkpoint = load_checkpoint_metadata(checkpoint_path, device)
    config = _coerce_config(checkpoint.get("model_config", ModelConfig()))
    config = replace(config, pretrained=False)
    if "model_name" in checkpoint:
        config = replace(config, model_name=checkpoint["model_name"])
    classes = int(num_classes or checkpoint.get("num_classes") or len(checkpoint.get("tags", [])) or 1000)
    model = DeepDanbooruV3(classes, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def make_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: str, epochs: int) -> Optional[Any]:
    if scheduler_type == "none":
        return None
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.1)
    raise ValueError("scheduler_type must be one of: none, cosine, step")


def train_main(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = get_device(args.device)
    logger.info("Using device: %s", device)

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
    )
    val_dataset = DanbooruTagDataset(
        annotation_path=args.val_annotations,
        image_dir=args.val_image_dir or args.image_dir,
        tags=tags,
        input_size=args.input_size,
        is_train=False,
        strict_images=not args.allow_missing_images,
    ) if args.val_annotations else None

    train_loader = create_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = create_dataloader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers) if val_dataset else None

    model_kwargs = model_kwargs_from_args(args)
    model = create_model(num_classes=len(tags), **model_kwargs).to(device)

    if args.freeze_backbone_epochs > 0:
        set_backbone_trainable(model, False)
        logger.info("Backbone frozen for first %d epoch(s)", args.freeze_backbone_epochs)

    if args.use_pos_weight:
        pos_weight = build_pos_weight(train_dataset, clamp_max=args.pos_weight_max).to(device)
        model.criterion = build_training_criterion(model.config, pos_weight=pos_weight)
        logger.info("Using positive class weights with max %.2f", args.pos_weight_max)

    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.scheduler, args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda") if hasattr(torch.cuda, "amp") else None
    start_epoch = 1
    best_metric = -math.inf

    if args.resume:
        checkpoint = load_checkpoint_metadata(args.resume, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    history: List[Dict[str, Any]] = []
    for epoch in range(start_epoch, args.epochs + 1):
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, True)
            optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
            scheduler = make_scheduler(optimizer, args.scheduler, args.epochs)
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
        )
        logger.info("Epoch %d train | %s", epoch, format_metrics(train_metrics))

        val_metrics: Dict[str, float] = {}
        if val_loader is not None:
            val_metrics = validate(model, val_loader, device, threshold=args.threshold, compute_map=args.compute_map)
            logger.info("Epoch %d valid | %s", epoch, format_metrics(val_metrics))

        if scheduler is not None:
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
            epoch=epoch,
            best_metric=best_metric,
            tags=tags,
            input_size=args.input_size,
            metrics={"train": train_metrics, "valid": val_metrics},
        )
        if is_best:
            save_training_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch,
                best_metric=best_metric,
                tags=tags,
                input_size=args.input_size,
                metrics={"train": train_metrics, "valid": val_metrics},
            )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_training_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch,
                best_metric=best_metric,
                tags=tags,
                input_size=args.input_size,
                metrics={"train": train_metrics, "valid": val_metrics},
            )

    logger.info("Training completed. Best %s: %.4f", args.monitor, best_metric)
    return 0


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


def load_image_tensor(image_path: Union[str, Path], input_size: int) -> Tensor:
    if Image is None:
        raise ImportError("Pillow is required for image loading. Install it with: pip install pillow")
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required for transforms. Install it with: pip install torchvision")
    transform = DataAugmentation.get_val_transforms(input_size)
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    return transform(image)


def decode_predictions(
    probabilities: Tensor,
    tags: Sequence[str],
    threshold: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    probabilities = probabilities.detach().cpu()
    for row in probabilities:
        k = min(top_k, row.numel())
        top_probs, top_indices = torch.topk(row, k=k)
        selected = [
            {"tag": tags[idx], "probability": float(row[idx].item())}
            for idx in torch.where(row >= threshold)[0].tolist()
        ]
        selected.sort(key=lambda item: item["probability"], reverse=True)
        top = [
            {"tag": tags[int(idx)], "probability": float(prob)}
            for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
        ]
        results.append({"selected": selected, "top_k": top})
    return results


@torch.inference_mode()
def infer_images(
    model: DeepDanbooruV3,
    image_paths: Sequence[Path],
    tags: Sequence[str],
    device: torch.device,
    input_size: int,
    batch_size: int,
    threshold: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    model.eval()
    outputs: List[Dict[str, Any]] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        images = torch.stack([load_image_tensor(path, input_size) for path in batch_paths], dim=0).to(device)
        probabilities = model(images)["probabilities"]
        decoded = decode_predictions(probabilities, tags=tags, threshold=threshold, top_k=top_k)
        for path, item in zip(batch_paths, decoded):
            item = {"image": str(path), **item}
            outputs.append(item)
    return outputs


def write_inference_results(results: Sequence[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    if output_path.suffix.lower() == ".csv":
        with output_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "selected_tags", "top_k"])
            writer.writeheader()
            for item in results:
                writer.writerow(
                    {
                        "image": item["image"],
                        "selected_tags": " ".join(tag_item["tag"] for tag_item in item["selected"]),
                        "top_k": json.dumps(item["top_k"], ensure_ascii=False),
                    }
                )
    else:
        output_path.write_text(json.dumps(list(results), ensure_ascii=False, indent=2), encoding="utf-8")


def infer_main(args: argparse.Namespace) -> int:
    device = get_device(args.device)
    checkpoint = load_checkpoint_metadata(args.checkpoint, device)
    checkpoint_tags = checkpoint.get("tags")
    tags = load_tags(args.tags_path) if args.tags_path else checkpoint_tags
    if not tags:
        raise ValueError("No tags found in checkpoint. Please pass --tags_path.")
    tags = [str(tag) for tag in tags]

    input_size = int(args.input_size or checkpoint.get("input_size", 224))
    model = load_model_from_checkpoint(args.checkpoint, device=device, num_classes=len(tags))
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
    )

    if args.output:
        write_inference_results(results, args.output)
        logger.info("Inference results saved to %s", args.output)
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def export_onnx_main(args: argparse.Namespace) -> int:
    device = get_device(args.device)
    checkpoint = load_checkpoint_metadata(args.checkpoint, device)
    tags = checkpoint.get("tags")
    num_classes = len(tags) if tags else int(checkpoint.get("num_classes", 1000))
    input_size = int(args.input_size or checkpoint.get("input_size", 224))
    model = load_model_from_checkpoint(args.checkpoint, device=device, num_classes=num_classes)
    model.eval()

    class OnnxWrapper(nn.Module):
        def __init__(self, inner: DeepDanbooruV3):
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
    logger.info("Backbone: %s", model.backbone.model_name)
    logger.info("Fusion: %s", model.config.fusion_type)
    logger.info("Parameters: %s", f"{stats['total_params']:,}")

    batch_size = 2
    images = torch.randn(batch_size, 3, args.input_size, args.input_size, device=device)
    targets = torch.randint(0, 2, (batch_size, args.num_classes), device=device).float()
    model.eval()
    with torch.inference_mode():
        output = model(images, targets)
        logger.info("Forward successful: logits=%s loss=%.4f", tuple(output["logits"].shape), float(output["loss"].item()))
        top_indices, top_probs = model.predict_top_k(images, k=5)
        logger.info("Top-k successful: %s %s", tuple(top_indices.shape), tuple(top_probs.shape))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    output = model(images, targets)
    output["loss"].backward()
    optimizer.step()
    logger.info("Backward/training step successful")
    return 0


# -------------------- CLI --------------------
def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_name", type=str, default=None, help="Backbone model name, e.g. efficientnet_b0")
    parser.add_argument("--feature_dim", type=int, default=None, help="Unified feature dimension")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=None, help="Use pretrained weights")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false", help="Do not use pretrained weights")
    loss_group = parser.add_mutually_exclusive_group()
    loss_group.add_argument("--use_focal_loss", action="store_true", help="Use focal loss")
    loss_group.add_argument("--use_label_smoothing", action="store_true", help="Use label smoothing BCE")
    parser.add_argument("--focal_alpha", type=float, default=None)
    parser.add_argument("--focal_gamma", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--fusion_type", type=str, default=None, choices=["attention", "weighted", "pyramid", "concat"])
    parser.add_argument("--use_cbam", dest="use_cbam", action="store_true", default=None, help="Use CBAM")
    parser.add_argument("--no_cbam", dest="use_cbam", action="store_false", help="Disable CBAM")
    parser.add_argument("--classifier_dropout", type=float, default=None)
    parser.add_argument("--projection_dropout", type=float, default=None)


def model_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    valid_keys = {field.name for field in fields(ModelConfig)}
    data: Dict[str, Any] = {}
    for key in valid_keys:
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                data[key] = value
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepDanbooru V3 complete single-file training and inference project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a multi-label tag model")
    add_model_arguments(train_parser)
    train_parser.add_argument("--train_annotations", required=True, help="CSV/JSON/JSONL train annotations")
    train_parser.add_argument("--val_annotations", default=None, help="CSV/JSON/JSONL validation annotations")
    train_parser.add_argument("--image_dir", default=None, help="Root directory for training images")
    train_parser.add_argument("--val_image_dir", default=None, help="Root directory for validation images")
    train_parser.add_argument("--tags_path", default=None, help="Tag file. If omitted, tags are built from train annotations")
    train_parser.add_argument("--output_dir", default="runs/deepdanbooru_v3", help="Output directory")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch_size", type=int, default=16)
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--input_size", type=int, default=224)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "step"])
    train_parser.add_argument("--threshold", type=float, default=0.5)
    train_parser.add_argument("--monitor", type=str, default="micro_f1", help="Validation metric for best.pt")
    train_parser.add_argument("--compute_map", action="store_true", help="Compute validation mAP")
    train_parser.add_argument("--use_pos_weight", action="store_true", help="Use BCE/Focal positive class weights")
    train_parser.add_argument("--pos_weight_max", type=float, default=20.0)
    train_parser.add_argument("--freeze_backbone_epochs", type=int, default=0)
    train_parser.add_argument("--grad_clip", type=float, default=None)
    train_parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision")
    train_parser.add_argument("--resume", default=None, help="Resume checkpoint path")
    train_parser.add_argument("--save_every", type=int, default=0, help="Save every N epochs; 0 disables periodic saves")
    train_parser.add_argument("--log_interval", type=int, default=20)
    train_parser.add_argument("--allow_missing_images", action="store_true")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default=None)
    train_parser.set_defaults(func=train_main)

    infer_parser = subparsers.add_parser("infer", help="Run inference on one image or a folder")
    infer_parser.add_argument("--checkpoint", required=True, help="Path to best.pt/last.pt")
    infer_parser.add_argument("--input", required=True, help="Image file or directory")
    infer_parser.add_argument("--tags_path", default=None, help="Optional tag file if checkpoint does not contain tags")
    infer_parser.add_argument("--output", default=None, help="Output .json or .csv. If omitted, prints JSON")
    infer_parser.add_argument("--input_size", type=int, default=None, help="Override checkpoint input size")
    infer_parser.add_argument("--batch_size", type=int, default=8)
    infer_parser.add_argument("--threshold", type=float, default=0.5)
    infer_parser.add_argument("--top_k", type=int, default=20)
    infer_parser.add_argument("--no_recursive", action="store_true", help="Do not recursively scan image directories")
    infer_parser.add_argument("--device", default=None)
    infer_parser.set_defaults(func=infer_main)

    export_parser = subparsers.add_parser("export_onnx", help="Export checkpoint to ONNX")
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--input_size", type=int, default=None)
    export_parser.add_argument("--opset", type=int, default=17)
    export_parser.add_argument("--device", default=None)
    export_parser.set_defaults(func=export_onnx_main)

    test_parser = subparsers.add_parser("test", help="Run a smoke test")
    add_model_arguments(test_parser)
    test_parser.add_argument("--num_classes", type=int, default=1000)
    test_parser.add_argument("--input_size", type=int, default=224)
    test_parser.add_argument("--device", default=None)
    test_parser.set_defaults(func=smoke_test_main)

    compat_parser = subparsers.add_parser("compatibility_test", help="Test several model configs")
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
