import argparse
from typing import Optional, Dict, List, Union, Tuple
import warnings
import math
import logging
import sys
import os

import torch
import torch.nn.functional as F
from torch import nn, Tensor

# 尝试导入 torchvision，用于数据增强
try:
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    transforms = None

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入 timm，提供更好的错误处理
try:
    import timm
    USE_TIMM = True
    logger.info(f"Using timm version: {timm.__version__}")
except ImportError:
    USE_TIMM = False
    logger.warning("timm not found. Please install with: pip install timm")

# -------------------- 基础模块 --------------------
class GeM(nn.Module):
    """广义均值池化 (Generalized Mean Pooling)"""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x: Tensor, p: float = 3.0, eps: float = 1e-6) -> Tensor:
        x_clamped = torch.clamp(x, min=eps)
        return F.avg_pool2d(x_clamped.pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced_channels = max(in_channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = torch.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_map = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(attention_map))
        return x * attention


class CBAM(nn.Module):
    """CBAM注意力模块"""
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class FeatureProjection(nn.Module):
    """特征投影模块"""
    def __init__(self, in_channels: int, out_channels: int = 512, use_gem: bool = True, use_cbam: bool = True):
        super().__init__()
        if use_cbam:
            self.attention = CBAM(in_channels)
        else:
            self.attention = ChannelAttention(in_channels)

        if use_gem:
            self.pool = GeM(p=3.0)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.attention(x)
        x = self.pool(x)
        return self.projection(x)


# -------------------- 主干网络 --------------------
class EfficientNetBackbone(nn.Module):
    """EfficientNet主干网络，支持多尺度特征输出"""
    def __init__(self, model_name: str = 'efficientnet_b4', pretrained: bool = True, use_cbam: bool = True):
        super().__init__()
        self.model_name = model_name
        self.use_cbam = use_cbam

        if not USE_TIMM:
            raise ImportError("timm is required for this model. Please install with: pip install timm")

        # 支持的模型列表（按优先级排序）
        supported_models = [
            'tf_efficientnet_b4_ns',
            'efficientnet_b4',
            'efficientnet_b3',
            'efficientnet_b2',
            'efficientnet_b0'
        ]

        # 尝试加载指定模型，失败则尝试备选方案
        model_loaded = False
        for attempt_model in [model_name] + supported_models:
            try:
                self.backbone = timm.create_model(
                    attempt_model,
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=(1, 2, 3, 4)  # 获取4个不同尺度的特征
                )
                self.model_name = attempt_model
                logger.info(f"Successfully loaded model: {attempt_model}")
                model_loaded = True
                break
            except Exception as e:
                logger.warning(f"Failed to load {attempt_model}: {e}")
                continue

        if not model_loaded:
            raise RuntimeError("Failed to load any supported EfficientNet model")

        # 获取特征通道数
        self.feature_channels = self._get_feature_channels()
        logger.info(f"Feature channels: {self.feature_channels}")

        # 创建特征投影层
        self.projections = nn.ModuleList([
            FeatureProjection(ch, 512, use_gem=True, use_cbam=use_cbam)
            for ch in self.feature_channels
        ])

    def _get_feature_channels(self) -> List[int]:
        """安全地获取各层特征通道数"""
        # 方法1: 从 model.feature_info 获取（推荐）
        if hasattr(self.backbone, 'feature_info'):
            try:
                channels = [info['num_chs'] for info in self.backbone.feature_info]
                # 根据 out_indices 筛选
                if hasattr(self.backbone, 'out_indices'):
                    indices = self.backbone.out_indices
                    channels = [channels[i] for i in indices]
                logger.info(f"Feature channels from feature_info: {channels}")
                return channels
            except Exception as e:
                logger.warning(f"Failed to get feature_info: {e}")

        # 方法2: 使用 dummy 前向传播
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy_input)
                channels = [f.shape[1] for f in features]
                logger.info(f"Feature channels from forward: {channels}")
                return channels
        except Exception as e:
            logger.warning(f"Failed to detect feature channels via forward: {e}")

        # 方法3: 根据模型名称返回默认值（回退方案）
        default_map = {
            'b0': [16, 24, 40, 112],
            'b4': [24, 32, 56, 160],
        }
        for key, val in default_map.items():
            if key in self.model_name:
                logger.warning(f"Using default channels for {self.model_name}: {val}")
                return val
        return [24, 32, 56, 160]  # 通用默认值

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        try:
            raw_features = self.backbone(x)
            projected_features = {}
            for idx, (feat, proj) in enumerate(zip(raw_features, self.projections)):
                projected_features[f'scale_{idx}'] = proj(feat)
            return projected_features
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise


# -------------------- 多尺度融合 --------------------
class MultiScaleFusion(nn.Module):
    """改进的多尺度特征融合模块（动态尺度数）"""
    def __init__(self, feature_dim: int = 512, fusion_type: str = 'attention'):
        super().__init__()
        self.feature_dim = feature_dim
        self.fusion_type = fusion_type
        # 注意：num_scales 将在第一次前向时根据输入特征数量动态确定
        self.num_scales = None
        self.fusion_layers = None  # 将在第一次前向时构建

    def _build_fusion_layers(self, num_scales: int):
        """根据尺度数构建融合层"""
        if self.fusion_type == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=self.feature_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.norm = nn.LayerNorm(self.feature_dim)
            self.dropout = nn.Dropout(0.1)
        elif self.fusion_type == 'weighted':
            self.scale_weights = nn.Parameter(torch.ones(num_scales))
            self.norm = nn.LayerNorm(self.feature_dim)
        elif self.fusion_type == 'pyramid':
            self.pyramid_fusion = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.feature_dim * 2, self.feature_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1)
                ) for _ in range(num_scales - 1)
            ])
            self.norm = nn.LayerNorm(self.feature_dim)
        else:  # 'concat'
            self.fusion = nn.Sequential(
                nn.Linear(self.feature_dim * num_scales, self.feature_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(self.feature_dim * 2, self.feature_dim),
                nn.LayerNorm(self.feature_dim)
            )

    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        # 收集所有特征
        feature_list = []
        for i in range(len(features)):  # 根据实际特征数量动态确定
            key = f'scale_{i}'
            if key in features:
                feature_list.append(features[key])
            else:
                break  # 假设特征键是连续的 scale_0, scale_1, ...

        if not feature_list:
            raise ValueError("No valid features found")

        num_scales = len(feature_list)
        # 如果是第一次前向，构建融合层
        if self.num_scales is None:
            self.num_scales = num_scales
            self._build_fusion_layers(num_scales)
        elif self.num_scales != num_scales:
            raise RuntimeError(f"Number of scales changed from {self.num_scales} to {num_scales}")

        batch_size = feature_list[0].shape[0]
        device = feature_list[0].device

        if self.fusion_type == 'attention':
            stacked = torch.stack(feature_list, dim=1)  # [B, num_scales, feature_dim]
            fused, _ = self.attention(stacked, stacked, stacked)
            fused = self.dropout(fused)
            fused = self.norm(fused.mean(dim=1))
        elif self.fusion_type == 'weighted':
            weights = torch.softmax(self.scale_weights, dim=0)
            fused = sum(w * f for w, f in zip(weights, feature_list))
            fused = self.norm(fused)
        elif self.fusion_type == 'pyramid':
            fused = feature_list[-1]
            for i in range(len(feature_list) - 2, -1, -1):
                combined = torch.cat([fused, feature_list[i]], dim=1)
                fused = self.pyramid_fusion[i](combined)
            fused = self.norm(fused)
        else:  # 'concat'
            fused = torch.cat(feature_list, dim=1)
            fused = self.fusion(fused)

        return fused


# -------------------- 损失函数 --------------------
class FocalLoss(nn.Module):
    """Focal Loss，处理类别不平衡"""
    def __init__(self, alpha: Union[float, Tensor] = 0.25, gamma: float = 2.0,
                 reduction: str = 'mean', pos_weight: Optional[Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        probs = torch.sigmoid(inputs)
        probs = torch.clamp(probs, min=1e-8, max=1-1e-8)

        if self.pos_weight is not None:
            bce_loss = F.binary_cross_entropy(probs, targets, weight=self.pos_weight, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(probs, targets, reduction='none')

        pt = torch.where(targets > 0.5, probs, 1 - probs)  # 使用 >0.5 判断正负
        if isinstance(self.alpha, (float, int)):
            alpha_t = torch.where(targets > 0.5, self.alpha, 1 - self.alpha)
        else:
            # alpha 应为 (num_classes,) 的张量，利用广播
            alpha_t = torch.where(targets > 0.5, self.alpha, 1 - self.alpha)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingBCE(nn.Module):
    """标准标签平滑 BCE Loss"""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # 标准标签平滑：正样本目标 = 1 - smoothing，负样本目标 = smoothing
        smoothed_targets = targets * (1 - self.smoothing) + (1 - targets) * self.smoothing
        return F.binary_cross_entropy_with_logits(inputs, smoothed_targets)


# -------------------- 主模型 --------------------
class DeepDanbooruV3(nn.Module):
    """优化版多标签分类模型"""
    def __init__(self, num_classes: int, config: Optional[argparse.Namespace] = None):
        super().__init__()
        self.num_classes = num_classes

        if config is None:
            config = self._get_default_config()
        self.config = config

        # 主干网络
        self.backbone = EfficientNetBackbone(
            model_name=config.model_name,
            pretrained=config.pretrained,
            use_cbam=config.use_cbam
        )

        # 特征融合（动态尺度数）
        self.fusion = MultiScaleFusion(
            feature_dim=512,
            fusion_type=config.fusion_type
        )

        # 分类头
        classifier_dropout = config.classifier_dropout
        self.classifier = self._build_classifier(classifier_dropout)

        # 损失函数
        self._setup_loss_function()

        # 初始化新加层的权重
        self._init_weights()

    def _get_default_config(self) -> argparse.Namespace:
        config = argparse.Namespace()
        config.model_name = 'efficientnet_b4'
        config.pretrained = True
        config.use_focal_loss = False
        config.use_label_smoothing = False
        config.fusion_type = 'attention'
        config.use_cbam = True
        config.classifier_dropout = 0.5
        return config

    def _build_classifier(self, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.4),
            nn.Linear(512, self.num_classes)
        )

    def _setup_loss_function(self):
        if self.config.use_focal_loss:
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        elif self.config.use_label_smoothing:
            self.criterion = LabelSmoothingBCE(smoothing=0.1)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def _init_weights(self):
        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 只初始化新添加的层（融合和分类器），保持主干预训练权重
        self.fusion.apply(init_layer)
        self.classifier.apply(init_layer)

    def forward(self, x: Tensor, targets: Optional[Tensor] = None) -> Dict[str, Tensor]:
        try:
            features = self.backbone(x)
            fused_features = self.fusion(features)
            logits = self.classifier(fused_features)
            probabilities = torch.sigmoid(logits)

            output = {
                'logits': logits,
                'probabilities': probabilities,
                'features': fused_features
            }

            if targets is not None:
                if isinstance(self.criterion, FocalLoss):
                    loss = self.criterion(logits, targets)
                else:
                    loss = self.criterion(logits, targets)
                output['loss'] = loss

            return output
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise

    def predict(self, x: Tensor, threshold: float = 0.5, return_probs: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            probabilities = output['probabilities']
            predictions = (probabilities > threshold).float()
            if return_probs:
                return predictions, probabilities
            return predictions

    def predict_top_k(self, x: Tensor, k: int = 5) -> Tuple[Tensor, Tensor]:
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            probabilities = output['probabilities']
            top_probs, top_indices = torch.topk(probabilities, k, dim=1)
            return top_indices, top_probs

    def get_feature_importance(self, x: Tensor) -> Dict[str, Tensor]:
        self.eval()
        features = self.backbone(x)
        importance = {}
        for key, feat in features.items():
            importance[key] = feat.mean(dim=(2, 3))
        return importance


# -------------------- 模型创建与辅助函数 --------------------
def create_model(num_classes: int, **kwargs) -> DeepDanbooruV3:
    config = argparse.Namespace()
    # 设置默认值
    default_config = DeepDanbooruV3._get_default_config(None)
    for key, value in vars(default_config).items():
        setattr(config, key, value)
    # 覆盖用户提供的参数
    for key, value in kwargs.items():
        setattr(config, key, value)
    return DeepDanbooruV3(num_classes, config)


def model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)) -> Dict[str, Union[int, float]]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = total_params * 4 / (1024 ** 2)
    buffer_size = sum(b.numel() for b in model.buffers()) * 4 / (1024 ** 2)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': param_size,
        'buffer_size_mb': buffer_size,
        'total_size_mb': param_size + buffer_size,
        'input_size': input_size
    }


def test_model_compatibility():
    logger.info("Testing model compatibility...")
    configs = [
        {'model_name': 'efficientnet_b0', 'fusion_type': 'attention'},
        {'model_name': 'efficientnet_b4', 'fusion_type': 'weighted'},
        {'model_name': 'efficientnet_b4', 'fusion_type': 'pyramid'},
    ]
    for config in configs:
        try:
            model = create_model(num_classes=1000, **config)
            x = torch.randn(1, 3, 224, 224)
            output = model(x)
            logger.info(f"✓ Config {config} works")
        except Exception as e:
            logger.warning(f"✗ Config {config} failed: {e}")
    logger.info("Compatibility test completed")


# -------------------- 工具类（保持不变，仅修正导入） --------------------
class ModelEnsemble(nn.Module):
    def __init__(self, models: List[DeepDanbooruV3], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0] * len(models)
        self.weights = torch.tensor(self.weights) / sum(self.weights)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        predictions = []
        features_list = []
        for model in self.models:
            with torch.no_grad():
                output = model(x)
                predictions.append(output['probabilities'])
                features_list.append(output['features'])
        weighted_pred = sum(w * pred for w, pred in zip(self.weights, predictions))
        avg_features = torch.stack(features_list).mean(dim=0)
        return {
            'probabilities': weighted_pred,
            'features': avg_features,
            'individual_predictions': predictions
        }


class GradCAM:
    def __init__(self, model: DeepDanbooruV3, target_layer: str = None):
        self.model = model
        self.target_layer = target_layer or 'backbone.backbone.4'
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        def forward_hook(module, input, output):
            self.activations = output
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break

    def generate_cam(self, x: Tensor, target_class: int) -> Tensor:
        self.model.eval()
        output = self.model(x)
        logits = output['logits']
        self.model.zero_grad()
        target_score = logits[0, target_class]
        target_score.backward()
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * activations, dim=0)
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


class DataAugmentation:
    @staticmethod
    def get_train_transforms(input_size: int = 224):
        if not TORCHVISION_AVAILABLE:
            logger.warning("torchvision not available, returning None")
            return None
        return transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_val_transforms(input_size: int = 224):
        if not TORCHVISION_AVAILABLE:
            logger.warning("torchvision not available, returning None")
            return None
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class ModelTrainer:
    def __init__(self, model: DeepDanbooruV3, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)

    def train_epoch(self, dataloader, optimizer, scheduler=None):
        self.model.train()
        total_loss = 0
        num_batches = 0
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            output = self.model(data, targets)
            loss = output['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            if batch_idx % 100 == 0:
                logger.info(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        if scheduler:
            scheduler.step()
        return total_loss / num_batches

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                output = self.model(data, targets)
                loss = output['loss']
                predictions = output['probabilities']
                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self._compute_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / len(dataloader)
        return metrics

    def _compute_metrics(self, predictions: Tensor, targets: Tensor, threshold: float = 0.5):
        pred_binary = (predictions > threshold).float()
        tp = (pred_binary * targets).sum(dim=0)
        fp = (pred_binary * (1 - targets)).sum(dim=0)
        fn = ((1 - pred_binary) * targets).sum(dim=0)
        tn = ((1 - pred_binary) * (1 - targets)).sum(dim=0)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
            'f1': f1.mean().item(),
            'accuracy': ((pred_binary == targets).float().mean()).item()
        }


def save_model(model: DeepDanbooruV3, filepath: str, optimizer=None, epoch=None, metrics=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model.config,
        'num_classes': model.num_classes,
        'model_name': model.backbone.model_name,
    }
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch:
        checkpoint['epoch'] = epoch
    if metrics:
        checkpoint['metrics'] = metrics
    torch.save(checkpoint, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str, num_classes: int = None, device: str = 'cpu') -> DeepDanbooruV3:
    checkpoint = torch.load(filepath, map_location=device)
    config = checkpoint.get('model_config', argparse.Namespace())
    num_classes = num_classes or checkpoint.get('num_classes', 1000)
    model = DeepDanbooruV3(num_classes, config)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Model loaded from {filepath}")
    return model


def example_usage():
    logger.info("=== DeepDanbooru V3 Usage Example ===")
    model = create_model(
        num_classes=6000,
        model_name='efficientnet_b4',
        fusion_type='attention',
        use_focal_loss=True,
        use_cbam=True
    )
    stats = model_summary(model)
    logger.info(f"Model parameters: {stats['total_params']:,}")

    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    y = torch.randint(0, 2, (batch_size, 6000)).float()  # 二值标签

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    output = model(x, y)
    loss = output['loss']
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    logger.info(f"Training loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        predictions = model.predict(x, threshold=0.3)
        top_indices, top_probs = model.predict_top_k(x, k=10)
    logger.info(f"Prediction shape: {predictions.shape}")
    logger.info(f"Top-10 predictions shape: {top_indices.shape}")

    importance = model.get_feature_importance(x)
    logger.info(f"Feature importance scales: {list(importance.keys())}")
    logger.info("Example completed successfully!")


# -------------------- 主入口 --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepDanbooru V3 Model - Improved Version')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classes')
    parser.add_argument('--model_name', type=str, default='efficientnet_b4', help='Backbone model name')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss instead of BCE')
    parser.add_argument('--use_label_smoothing', action='store_true', help='Use label smoothing BCE loss')
    parser.add_argument('--fusion_type', type=str, default='attention', choices=['attention', 'weighted', 'pyramid', 'concat'], help='Feature fusion method')
    parser.add_argument('--use_cbam', action='store_true', default=True, help='Use CBAM attention mechanism')
    parser.add_argument('--classifier_dropout', type=float, default=0.5, help='Classifier dropout rate')
    parser.add_argument('--test', action='store_true', help='Run test')
    parser.add_argument('--compatibility_test', action='store_true', help='Run compatibility test')

    args = parser.parse_args()

    try:
        if args.compatibility_test:
            test_model_compatibility()
            sys.exit(0)

        model = create_model(num_classes=args.num_classes, **vars(args))
        stats = model_summary(model)
        logger.info("Model created successfully!")
        logger.info(f"Classes: {args.num_classes}")
        logger.info(f"Backbone: {args.model_name}")
        logger.info(f"Fusion type: {args.fusion_type}")
        logger.info(f"Total parameters: {stats['total_params']:,}")
        logger.info(f"Trainable parameters: {stats['trainable_params']:,}")
        logger.info(f"Model size: {stats['total_size_mb']:.2f} MB")

        if args.test:
            logger.info("Running comprehensive test...")
            batch_size = 2
            input_tensor = torch.randn(batch_size, 3, 224, 224)
            target_tensor = torch.randint(0, 2, (batch_size, args.num_classes)).float()  # 二值标签

            model.eval()
            with torch.no_grad():
                output = model(input_tensor, target_tensor)
                logger.info("✓ Forward pass successful!")
                for key, value in output.items():
                    if isinstance(value, torch.Tensor):
                        logger.info(f"  {key}: {value.shape}")
                    else:
                        logger.info(f"  {key}: {value}")

                predictions = model.predict(input_tensor, threshold=0.5)
                logger.info(f"✓ Predictions shape: {predictions.shape}")

                top_indices, top_probs = model.predict_top_k(input_tensor, k=5)
                logger.info(f"✓ Top-5 predictions: {top_indices.shape}, {top_probs.shape}")

                importance = model.get_feature_importance(input_tensor)
                logger.info(f"✓ Feature importance computed for {len(importance)} scales")

            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            optimizer.zero_grad()
            output = model(input_tensor, target_tensor)
            loss = output['loss']
            loss.backward()
            optimizer.step()
            logger.info(f"✓ Training step successful! Loss: {loss.item():.4f}")
            logger.info("All tests passed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
