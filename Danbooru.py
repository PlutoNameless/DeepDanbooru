import argparse
from typing import Optional

import torch
from torch import nn, Tensor
from torchvision.ops import StochasticDepth
from efficientnet_pytorch import EfficientNet
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GeM(nn.Module):
    """广义均值池化"""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return self.gem(x, p=self.p, eps=self.eps)
    
    def gem(self, x: Tensor, p=3, eps=1e-6) -> Tensor:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class StochasticDepthBlock(nn.Module):
    """带随机深度的MBConv块"""
    def __init__(self, block, survival_prob=0.8):
        super().__init__()
        self.block = block
        self.survival_prob = survival_prob
        self.stochastic_depth = StochasticDepth(survival_prob, "row")

    def forward(self, x: Tensor) -> Tensor:
        return self.stochastic_depth(self.block(x))

class EfficientNetBackbone(nn.Module):
    """改进的EfficientNet主干网络"""
    def __init__(self, name: str = 'efficientnet-b4', pretrained: bool = True):
        super().__init__()
        model = EfficientNet.from_pretrained(name) if pretrained else EfficientNet.from_name(name)
        
        # 提取多尺度特征
        self.stem = nn.Sequential(
            model._conv_stem,
            model._bn0,
            model._swish
        )
        
        self.blocks = nn.ModuleList([
            nn.Sequential(*model._blocks[:2]),   # stride 2
            nn.Sequential(*model._blocks[2:4]),  # stride 4
            nn.Sequential(*model._blocks[4:6]),  # stride 8
            nn.Sequential(*model._blocks[6:])    # stride 16
        ])
        
        # 添加随机深度
        for i, block in enumerate(self.blocks):
            for j in range(len(block)):
                survival_prob = 1 - (i * len(block) + j) / (4 * len(block)) * 0.2
                block[j] = StochasticDepthBlock(block[j], survival_prob)
        
        # 特征投影
        self.projections = nn.ModuleList([
            nn.Sequential(
                GeM(),
                nn.Conv2d(32, 512, 1),
                nn.BatchNorm2d(512),
                nn.SiLU(inplace=True)
            ),
            nn.Sequential(
                GeM(),
                nn.Conv2d(56, 512, 1),
                nn.BatchNorm2d(512),
                nn.SiLU(inplace=True)
            ),
            nn.Sequential(
                GeM(),
                nn.Conv2d(160, 512, 1),
                nn.BatchNorm2d(512),
                nn.SiLU(inplace=True)
            ),
            nn.Sequential(
                GeM(),
                nn.Conv2d(448, 512, 1),
                nn.BatchNorm2d(512),
                nn.SiLU(inplace=True)
            )
        ])

    def forward(self, x: Tensor) -> Dict[int, Tensor]:
        features = {}
        x = self.stem(x)
        
        for idx, block in enumerate(self.blocks):
            x = block(x)
            features[idx+1] = self.projections[idx](x)
        
        return features

class CrossAttentionFusion(nn.Module):
    """高效跨尺度注意力融合"""
    def __init__(self, dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(4)])
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(3)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim*4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim*4, dim)
            )
            for _ in range(3)
        ])

    def forward(self, features: Dict[int, Tensor]) -> Tensor:
        # 按尺度排序特征 [B, C, H, W]
        keys = sorted(features.keys())
        feats = [features[k].flatten(2).permute(0, 2, 1) for k in keys]  # [B, N, C]
        
        # 自顶向下融合
        fused = feats[-1]
        for i in reversed(range(len(feats)-1)):
            fused = self.norm[i](fused)
            context, _ = self.attn[i](
                query=self.norm[i](feats[i]),
                key=fused,
                value=fused
            )
            fused = context + self.ffn[i](context)
        
        return fused.mean(dim=1)  # 全局平均

class DeepDanbooruV3(nn.Module):
    """优化版多标签分类模型"""
    def __init__(self, num_classes: int, config: argparse.Namespace):
        super().__init__()
        # 主干网络
        self.backbone = EfficientNetBackbone()
        
        # 特征融合
        self.fusion = CrossAttentionFusion()
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
        )
        
        # 初始化
        self._init_weights()
        self.config = config

    def _init_weights(self):
        # 使用自适应初始化
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            # 特征提取 [B, C, H, W]
            features = self.backbone(x)
            
            # 跨尺度融合
            fused = self.fusion(features)  # [B, 512]
            
            # 分类
            return torch.sigmoid(self.classifier(fused))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--use_amp', action='store_true')
    config = parser.parse_args()

    model = DeepDanbooruV3(num_classes=config.num_classes, config=config)
