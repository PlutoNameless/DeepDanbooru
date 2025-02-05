import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import List

# 添加位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# 改进的ResNetBlock
class ResNetBlock(nn.Module):
    expansion = 4  # 添加expansion属性
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        
        out += self.shortcut(identity)
        out = F.relu(out, inplace=True)
        return out

class ResNet(nn.Module):
    def __init__(self, block: nn.Module, num_blocks: List[int], num_classes: int = 1000):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: nn.Module, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class DeepDanbooruModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        
        # 特征提取
        self.resnet = ResNet(ResNetBlock, [3, 4, 23, 3], num_classes=2048)
        
        # 序列建模
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=args.lstm_hidden_size,
            num_layers=args.lstm_num_layers,
            batch_first=True,
            dropout=0.2 if args.lstm_num_layers > 1 else 0
        )
        
        # Transformer
        self.pos_encoder = PositionalEncoding(args.lstm_hidden_size)
        encoder_layer = TransformerEncoderLayer(
            d_model=args.lstm_hidden_size,
            nhead=args.transformer_nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, args.transformer_num_encoder_layers)
        
        # 分类头
        self.fc = nn.Linear(args.lstm_hidden_size, num_classes)
        self._init_weights()

    def _init_weights(self):
        init_range = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 特征提取 [B, 3, H, W] -> [B, 2048]
        x = self.resnet(x)
        
        # 添加序列维度 [B, 2048] -> [B, 1, 2048]
        x = x.unsqueeze(1)
        
        # LSTM处理 [B, 1, 2048] -> [B, 1, lstm_hidden_size]
        x, _ = self.lstm(x)
        
        # Transformer处理 [B, 1, H] -> [1, B, H]
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # 取最后一个时间步 [B, H]
        x = x[-1]  # 当序列长度为1时等价于 x.squeeze(0)
        
        # 分类输出
        x = self.fc(x)
        return torch.sigmoid(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lstm_hidden_size', default=512, type=int)
    parser.add_argument('--lstm_num_layers', default=2, type=int)  
    parser.add_argument('--transformer_nhead', default=12, type=int)
    parser.add_argument('--transformer_num_encoder_layers', default=12, type=int)
    parser.add_argument('--num_classes', required=True, type=int)  # 必须指定类别数
    args = parser.parse_args()

    model = DeepDanbooruModel(num_classes=args.num_classes)
