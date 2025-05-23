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

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入timm，提供更好的错误处理
try:
    import timm
    USE_TIMM = True
    logger.info(f"Using timm version: {timm.__version__}")
except ImportError:
    USE_TIMM = False
    logger.warning("timm not found. Please install with: pip install timm")

class GeM(nn.Module):
    """广义均值池化 (Generalized Mean Pooling)
    
    相比传统平均池化，GeM可以更好地保留显著特征
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return self.gem(x, p=self.p, eps=self.eps)
    
    def gem(self, x: Tensor, p: float = 3.0, eps: float = 1e-6) -> Tensor:
        # 添加数值稳定性检查
        x_clamped = torch.clamp(x, min=eps)
        return F.avg_pool2d(x_clamped.pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class ChannelAttention(nn.Module):
    """通道注意力模块 (改进的CBAM风格)"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 确保reduction后的通道数至少为1
        reduced_channels = max(in_channels // reduction, 1)
        
        # 使用1x1卷积替代全连接，更高效
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
    """完整的CBAM注意力模块"""
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class FeatureProjection(nn.Module):
    """改进的特征投影模块"""
    def __init__(self, in_channels: int, out_channels: int = 512, use_gem: bool = True, use_cbam: bool = True):
        super().__init__()
        
        # 注意力机制选择
        if use_cbam:
            self.attention = CBAM(in_channels)
        else:
            self.attention = ChannelAttention(in_channels)
            
        # 池化方式选择
        if use_gem:
            self.pool = GeM(p=3.0)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
            
        # 特征投影
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

class EfficientNetBackbone(nn.Module):
    """改进的EfficientNet主干网络"""
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
            FeatureProjection(channels, 512, use_gem=True, use_cbam=use_cbam)
            for channels in self.feature_channels
        ])

    def _get_feature_channels(self) -> List[int]:
        """安全地获取各层特征通道数"""
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy_input)
                channels = [f.shape[1] for f in features]
                logger.info(f"Detected feature channels: {channels}")
                return channels
        except Exception as e:
            logger.warning(f"Failed to detect feature channels: {e}, using defaults")
            # 根据不同模型提供默认值
            if 'b0' in self.model_name:
                return [16, 24, 40, 112]
            elif 'b4' in self.model_name:
                return [24, 32, 56, 160]
            else:
                return [24, 32, 56, 160]  # 通用默认值

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        try:
            # 获取多尺度特征
            raw_features = self.backbone(x)
            
            # 投影到统一维度
            projected_features = {}
            for idx, (feat, proj) in enumerate(zip(raw_features, self.projections)):
                projected_features[f'scale_{idx}'] = proj(feat)
            
            return projected_features
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise

class MultiScaleFusion(nn.Module):
    """改进的多尺度特征融合模块"""
    def __init__(self, feature_dim: int = 512, num_scales: int = 4, fusion_type: str = 'attention'):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_scales = num_scales
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention':
            # 多头注意力融合
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.norm = nn.LayerNorm(feature_dim)
            self.dropout = nn.Dropout(0.1)
            
        elif fusion_type == 'weighted':
            # 学习权重融合
            self.scale_weights = nn.Parameter(torch.ones(num_scales))
            self.norm = nn.LayerNorm(feature_dim)
            
        elif fusion_type == 'pyramid':
            # 金字塔融合
            self.pyramid_fusion = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_dim * 2, feature_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1)
                ) for _ in range(num_scales - 1)
            ])
            self.norm = nn.LayerNorm(feature_dim)
            
        else:
            # 拼接融合（默认）
            self.fusion = nn.Sequential(
                nn.Linear(feature_dim * num_scales, feature_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(feature_dim * 2, feature_dim),
                nn.LayerNorm(feature_dim)
            )
        
    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        # 收集所有特征
        feature_list = []
        batch_size = None
        device = None
        
        for i in range(self.num_scales):
            key = f'scale_{i}'
            if key in features:
                feat = features[key]
                feature_list.append(feat)
                if batch_size is None:
                    batch_size = feat.shape[0]
                    device = feat.device
            else:
                # 用零张量填充缺失的尺度
                if batch_size is not None and device is not None:
                    feature_list.append(torch.zeros(
                        batch_size, self.feature_dim, device=device
                    ))
        
        if not feature_list:
            raise ValueError("No valid features found")
        
        if self.fusion_type == 'attention':
            # 注意力融合
            stacked = torch.stack(feature_list, dim=1)  # [B, num_scales, feature_dim]
            fused, _ = self.attention(stacked, stacked, stacked)
            fused = self.dropout(fused)
            fused = self.norm(fused.mean(dim=1))  # 全局平均池化
            
        elif self.fusion_type == 'weighted':
            # 加权融合
            weights = torch.softmax(self.scale_weights, dim=0)
            fused = sum(w * f for w, f in zip(weights, feature_list))
            fused = self.norm(fused)
            
        elif self.fusion_type == 'pyramid':
            # 金字塔融合（自下而上）
            fused = feature_list[-1]  # 从最高层开始
            for i in range(len(feature_list) - 2, -1, -1):
                combined = torch.cat([fused, feature_list[i]], dim=1)
                fused = self.pyramid_fusion[i](combined)
            fused = self.norm(fused)
            
        else:
            # 拼接融合
            fused = torch.cat(feature_list, dim=1)
            fused = self.fusion(fused)
        
        return fused

class FocalLoss(nn.Module):
    """改进的Focal Loss，处理类别不平衡"""
    def __init__(self, alpha: Union[float, Tensor] = 0.25, gamma: float = 2.0, 
                 reduction: str = 'mean', pos_weight: Optional[Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
        
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # 应用sigmoid并限制范围避免数值不稳定
        probs = torch.sigmoid(inputs)
        probs = torch.clamp(probs, min=1e-8, max=1-1e-8)
        
        # 计算BCE loss
        if self.pos_weight is not None:
            bce_loss = F.binary_cross_entropy(probs, targets, weight=self.pos_weight, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        
        # 计算pt
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # 计算alpha_t
        if isinstance(self.alpha, (float, int)):
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        else:
            alpha_t = self.alpha
        
        # 计算focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingBCE(nn.Module):
    """带标签平滑的BCE Loss"""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # 标签平滑
        smoothed_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(inputs, smoothed_targets)

class DeepDanbooruV3(nn.Module):
    """优化版多标签分类模型"""
    def __init__(self, num_classes: int, config: Optional[argparse.Namespace] = None):
        super().__init__()
        self.num_classes = num_classes
        
        # 使用默认配置如果未提供
        if config is None:
            config = self._get_default_config()
        
        self.config = config
        
        # 主干网络
        self.backbone = EfficientNetBackbone(
            model_name=getattr(config, 'model_name', 'efficientnet_b4'),
            pretrained=getattr(config, 'pretrained', True),
            use_cbam=getattr(config, 'use_cbam', True)
        )
        
        # 特征融合
        self.fusion = MultiScaleFusion(
            feature_dim=512,
            num_scales=4,
            fusion_type=getattr(config, 'fusion_type', 'attention')
        )
        
        # 分类头 - 使用更深的网络和残差连接
        classifier_dropout = getattr(config, 'classifier_dropout', 0.5)
        self.classifier = self._build_classifier(classifier_dropout)
        
        # 损失函数选择
        self._setup_loss_function()
        
        # 初始化权重
        self._init_weights()

    def _get_default_config(self) -> argparse.Namespace:
        """获取默认配置"""
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
        """构建分类器"""
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
        """设置损失函数"""
        if getattr(self.config, 'use_focal_loss', False):
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        elif getattr(self.config, 'use_label_smoothing', False):
            self.criterion = LabelSmoothingBCE(smoothing=0.1)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def _init_weights(self):
        """改进的权重初始化"""
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
        
        # 只初始化新添加的层，保持预训练权重
        self.fusion.apply(init_layer)
        self.classifier.apply(init_layer)

    def forward(self, x: Tensor, targets: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """前向传播"""
        try:
            # 特征提取
            features = self.backbone(x)
            
            # 多尺度融合
            fused_features = self.fusion(features)
            
            # 分类
            logits = self.classifier(fused_features)
            probabilities = torch.sigmoid(logits)
            
            # 输出字典
            output = {
                'logits': logits,
                'probabilities': probabilities,
                'features': fused_features  # 返回特征用于可视化或其他用途
            }
            
            # 计算损失
            if targets is not None:
                if isinstance(self.criterion, FocalLoss):
                    loss = self.criterion(logits, targets)  # Focal loss需要logits
                else:
                    loss = self.criterion(logits, targets)
                output['loss'] = loss
            
            return output
            
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise

    def predict(self, x: Tensor, threshold: float = 0.5, return_probs: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """预测函数"""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            probabilities = output['probabilities']
            predictions = (probabilities > threshold).float()
            
            if return_probs:
                return predictions, probabilities
            return predictions
    
    def predict_top_k(self, x: Tensor, k: int = 5) -> Tuple[Tensor, Tensor]:
        """预测前k个最可能的标签"""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            probabilities = output['probabilities']
            top_probs, top_indices = torch.topk(probabilities, k, dim=1)
            return top_indices, top_probs
    
    def get_feature_importance(self, x: Tensor) -> Dict[str, Tensor]:
        """获取特征重要性（用于可解释性）"""
        self.eval()
        features = self.backbone(x)
        
        # 计算每个尺度的平均激活
        importance = {}
        for key, feat in features.items():
            importance[key] = feat.mean(dim=(2, 3))  # 空间维度平均
        
        return importance

def create_model(num_classes: int, **kwargs) -> DeepDanbooruV3:
    """模型创建函数"""
    config = argparse.Namespace()
    
    # 设置配置
    config.model_name = kwargs.get('model_name', 'efficientnet_b4')
    config.pretrained = kwargs.get('pretrained', True)
    config.use_focal_loss = kwargs.get('use_focal_loss', False)
    config.use_label_smoothing = kwargs.get('use_label_smoothing', False)
    config.fusion_type = kwargs.get('fusion_type', 'attention')
    config.use_cbam = kwargs.get('use_cbam', True)
    config.classifier_dropout = kwargs.get('classifier_dropout', 0.5)
    
    return DeepDanbooruV3(num_classes, config)

def model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)) -> Dict[str, Union[int, float]]:
    """模型参数统计"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 估算模型大小（MB）
    param_size = total_params * 4 / (1024 ** 2)  # 假设float32
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
    """测试模型兼容性"""
    logger.info("Testing model compatibility...")
    
    try:
        # 测试不同配置
        configs = [
            {'model_name': 'efficientnet_b0', 'fusion_type': 'attention'},
            {'model_name': 'efficientnet_b4', 'fusion_type': 'weighted'},
            {'model_name': 'efficientnet_b4', 'fusion_type': 'pyramid'},
        ]
        
        for config in configs:
            try:
                model = create_model(num_classes=1000, **config)
                # 简单测试
                x = torch.randn(1, 3, 224, 224)
                output = model(x)
                logger.info(f"✓ Config {config} works")
            except Exception as e:
                logger.warning(f"✗ Config {config} failed: {e}")
        
        logger.info("Compatibility test completed")
        
    except Exception as e:
        logger.error(f"Compatibility test failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepDanbooru V3 Model - Improved Version')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classes')
    parser.add_argument('--model_name', type=str, default='efficientnet_b4', 
                       help='Backbone model name')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--use_focal_loss', action='store_true', 
                       help='Use focal loss instead of BCE')
    parser.add_argument('--use_label_smoothing', action='store_true',
                       help='Use label smoothing BCE loss')
    parser.add_argument('--fusion_type', type=str, default='attention',
                       choices=['attention', 'weighted', 'pyramid', 'concat'],
                       help='Feature fusion method')
    parser.add_argument('--use_cbam', action='store_true', default=True,
                       help='Use CBAM attention mechanism')
    parser.add_argument('--classifier_dropout', type=float, default=0.5,
                       help='Classifier dropout rate')
    parser.add_argument('--test', action='store_true', help='Run test')
    parser.add_argument('--compatibility_test', action='store_true', 
                       help='Run compatibility test')
    
    args = parser.parse_args()
    
    try:
        if args.compatibility_test:
            test_model_compatibility()
        
        # 创建模型
        model = create_model(num_classes=args.num_classes, **vars(args))
        
        # 模型统计
        stats = model_summary(model)
        logger.info("Model created successfully!")
        logger.info(f"Classes: {args.num_classes}")
        logger.info(f"Backbone: {args.model_name}")
        logger.info(f"Fusion type: {args.fusion_type}")
        logger.info(f"Total parameters: {stats['total_params']:,}")
        logger.info(f"Trainable parameters: {stats['trainable_params']:,}")
        logger.info(f"Model size: {stats['total_size_mb']:.2f} MB")
        
        if args.test:
            # 测试运行
            logger.info("Running comprehensive test...")
            batch_size = 2
            input_tensor = torch.randn(batch_size, 3, 224, 224)
            target_tensor = torch.rand(batch_size, args.num_classes)
            
            # 前向传播测试
            model.eval()
            with torch.no_grad():
                # 基本前向传播
                output = model(input_tensor, target_tensor)
                logger.info("✓ Forward pass successful!")
                
                for key, value in output.items():
                    if isinstance(value, torch.Tensor):
                        logger.info(f"  {key}: {value.shape}")
                    else:
                        logger.info(f"  {key}: {value}")
                
                # 预测测试
                predictions = model.predict(input_tensor, threshold=0.5)
                logger.info(f"✓ Predictions shape: {predictions.shape}")
                
                # Top-k预测测试
                top_indices, top_probs = model.predict_top_k(input_tensor, k=5)
                logger.info(f"✓ Top-5 predictions: {top_indices.shape}, {top_probs.shape}")
                
                # 特征重要性测试
                importance = model.get_feature_importance(input_tensor)
                logger.info(f"✓ Feature importance computed for {len(importance)} scales")
                
            # 训练模式测试
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            # 模拟训练步骤
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

# 额外的工具函数和类

class ModelEnsemble(nn.Module):
    """模型集成类，用于提升预测性能"""
    def __init__(self, models: List[DeepDanbooruV3], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0] * len(models)
        self.weights = torch.tensor(self.weights) / sum(self.weights)  # 归一化权重
        
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        predictions = []
        features_list = []
        
        for model in self.models:
            with torch.no_grad():
                output = model(x)
                predictions.append(output['probabilities'])
                features_list.append(output['features'])
        
        # 加权平均预测
        weighted_pred = sum(w * pred for w, pred in zip(self.weights, predictions))
        
        # 特征平均
        avg_features = torch.stack(features_list).mean(dim=0)
        
        return {
            'probabilities': weighted_pred,
            'features': avg_features,
            'individual_predictions': predictions
        }

class GradCAM:
    """Grad-CAM可视化工具，用于模型解释性"""
    def __init__(self, model: DeepDanbooruV3, target_layer: str = None):
        self.model = model
        self.target_layer = target_layer or 'backbone.backbone.4'  # 默认最后一个特征层
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self._register_hooks()
    
    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        # 根据目标层名称查找并注册钩子
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break
    
    def generate_cam(self, x: Tensor, target_class: int) -> Tensor:
        """生成类激活图"""
        self.model.eval()
        
        # 前向传播
        output = self.model(x)
        logits = output['logits']
        
        # 反向传播到目标类
        self.model.zero_grad()
        target_score = logits[0, target_class]
        target_score.backward()
        
        # 生成CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # 计算权重
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # 加权求和
        cam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * activations, dim=0)
        cam = F.relu(cam)  # 只保留正值
        
        # 归一化到0-1
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

class DataAugmentation:
    """数据增强工具类"""
    @staticmethod
    def get_train_transforms(input_size: int = 224):
        """获取训练时的数据变换"""
        try:
            import torchvision.transforms as transforms
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
        except ImportError:
            logger.warning("torchvision not available, using basic transforms")
            return None
    
    @staticmethod
    def get_val_transforms(input_size: int = 224):
        """获取验证时的数据变换"""
        try:
            import torchvision.transforms as transforms
            return transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except ImportError:
            logger.warning("torchvision not available, using basic transforms")
            return None

class ModelTrainer:
    """模型训练器"""
    def __init__(self, model: DeepDanbooruV3, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train_epoch(self, dataloader, optimizer, scheduler=None):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data, targets)
            loss = output['loss']
            loss.backward()
            
            # 梯度裁剪
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
        """验证模型"""
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
        
        # 计算指标
        metrics = self._compute_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics
    
    def _compute_metrics(self, predictions: Tensor, targets: Tensor, threshold: float = 0.5):
        """计算评估指标"""
        pred_binary = (predictions > threshold).float()
        
        # 计算各种指标
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
    """保存模型"""
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
    """加载模型"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # 获取模型配置
    config = checkpoint.get('model_config', argparse.Namespace())
    num_classes = num_classes or checkpoint.get('num_classes', 1000)
    
    # 创建模型
    model = DeepDanbooruV3(num_classes, config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Model loaded from {filepath}")
    return model

# 使用示例和测试代码
def example_usage():
    """使用示例"""
    logger.info("=== DeepDanbooru V3 Usage Example ===")
    
    # 1. 创建模型
    model = create_model(
        num_classes=6000,  # Danbooru通常有数千个标签
        model_name='efficientnet_b4',
        fusion_type='attention',
        use_focal_loss=True,
        use_cbam=True
    )
    
    # 2. 模型统计
    stats = model_summary(model)
    logger.info(f"Model parameters: {stats['total_params']:,}")
    
    # 3. 模拟数据
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    y = torch.randint(0, 2, (batch_size, 6000)).float()
    
    # 4. 训练步骤
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    output = model(x, y)
    loss = output['loss']
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    logger.info(f"Training loss: {loss.item():.4f}")
    
    # 5. 预测
    model.eval()
    with torch.no_grad():
        predictions = model.predict(x, threshold=0.3)
        top_indices, top_probs = model.predict_top_k(x, k=10)
        
    logger.info(f"Prediction shape: {predictions.shape}")
    logger.info(f"Top-10 predictions shape: {top_indices.shape}")
    
    # 6. 特征重要性
    importance = model.get_feature_importance(x)
    logger.info(f"Feature importance scales: {list(importance.keys())}")
    
    logger.info("Example completed successfully!")

if __name__ == "__main__":
    # 如果直接运行此脚本，先运行示例
    if len(sys.argv) == 1:
        example_usage()
