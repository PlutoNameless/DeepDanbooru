"""
DeepDanbooru V3 - fixed and cleaned implementation.

Main fixes compared with the uploaded version:
- fixes duplicate num_classes argument in CLI model creation
- fixes get_feature_importance() dimensionality bug
- fixes FocalLoss pos_weight usage
- replaces BatchNorm1d with LayerNorm so batch_size=1 works
- adds robust config serialization/loading
- adds pretrained fallback to non-pretrained weights when timm cannot download weights
- fixes CLI boolean flags such as --no_pretrained and --no_cbam
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

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


# -------------------- CLI --------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepDanbooru V3 Model - Fixed Version")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of classes")
    parser.add_argument("--model_name", type=str, default=None, help="Backbone model name")
    parser.add_argument("--feature_dim", type=int, default=None, help="Unified feature dimension")

    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=None, help="Use pretrained weights")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false", help="Do not use pretrained weights")

    loss_group = parser.add_mutually_exclusive_group()
    loss_group.add_argument("--use_focal_loss", action="store_true", help="Use focal loss instead of BCE")
    loss_group.add_argument("--use_label_smoothing", action="store_true", help="Use label smoothing BCE loss")
    parser.add_argument("--focal_alpha", type=float, default=None, help="Focal loss alpha")
    parser.add_argument("--focal_gamma", type=float, default=None, help="Focal loss gamma")
    parser.add_argument("--label_smoothing", type=float, default=None, help="Label smoothing value")

    parser.add_argument(
        "--fusion_type",
        type=str,
        default=None,
        choices=["attention", "weighted", "pyramid", "concat"],
        help="Feature fusion method",
    )
    parser.add_argument("--use_cbam", dest="use_cbam", action="store_true", default=None, help="Use CBAM attention")
    parser.add_argument("--no_cbam", dest="use_cbam", action="store_false", help="Disable CBAM attention")
    parser.add_argument("--classifier_dropout", type=float, default=None, help="Classifier dropout rate")
    parser.add_argument("--projection_dropout", type=float, default=None, help="Projection dropout rate")

    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, cuda:0, etc.")
    parser.add_argument("--input_size", type=int, default=224, help="Test input image size")
    parser.add_argument("--test", action="store_true", help="Run a forward/backward smoke test")
    parser.add_argument("--compatibility_test", action="store_true", help="Run compatibility tests")
    return parser


def _model_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    valid_keys = {f.name for f in fields(ModelConfig)}
    data: Dict[str, Any] = {}
    for key in valid_keys:
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                data[key] = value
    return data


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.compatibility_test:
            test_model_compatibility()
            return 0

        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model_kwargs = _model_kwargs_from_args(args)
        model = create_model(num_classes=args.num_classes, **model_kwargs).to(device)
        stats = model_summary(model, input_size=(3, args.input_size, args.input_size))

        logger.info("Model created successfully")
        logger.info("Classes: %s", args.num_classes)
        logger.info("Backbone: %s", model.backbone.model_name)
        logger.info("Pretrained loaded: %s", model.backbone.loaded_pretrained)
        logger.info("Fusion type: %s", model.config.fusion_type)
        logger.info("Total parameters: %s", f"{stats['total_params']:,}")
        logger.info("Trainable parameters: %s", f"{stats['trainable_params']:,}")
        logger.info("Model size: %.2f MB", stats["total_size_mb"])

        if args.test:
            logger.info("Running smoke test...")
            batch_size = 2
            input_tensor = torch.randn(batch_size, 3, args.input_size, args.input_size, device=device)
            target_tensor = torch.randint(0, 2, (batch_size, args.num_classes), device=device).float()

            model.eval()
            with torch.inference_mode():
                output = model(input_tensor, target_tensor)
                logger.info("✓ Forward pass successful")
                for key, value in output.items():
                    if isinstance(value, torch.Tensor):
                        logger.info("  %s: %s", key, tuple(value.shape) if value.ndim > 0 else float(value.item()))

                predictions = model.predict(input_tensor, threshold=0.5)
                logger.info("✓ Predictions shape: %s", tuple(predictions.shape))

                top_indices, top_probs = model.predict_top_k(input_tensor, k=5)
                logger.info("✓ Top-5 predictions: %s, %s", tuple(top_indices.shape), tuple(top_probs.shape))

                importance = model.get_feature_importance(input_tensor)
                logger.info("✓ Feature importance scales: %s", {k: tuple(v.shape) for k, v in importance.items()})

            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            optimizer.zero_grad(set_to_none=True)
            output = model(input_tensor, target_tensor)
            loss = output["loss"]
            loss.backward()
            optimizer.step()
            logger.info("✓ Training step successful. Loss: %.4f", float(loss.item()))
            logger.info("All tests passed")

        return 0

    except Exception as exc:
        logger.error("Error: %s", exc)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
