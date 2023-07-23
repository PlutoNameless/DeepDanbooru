import argparse
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

parser = argparse.ArgumentParser()
parser.add_argument('--lstm_hidden_size', default=512, type=int)
parser.add_argument('--lstm_num_layers', default=2, type=int)  
parser.add_argument('--transformer_nhead', default=12, type=int)
parser.add_argument('--transformer_num_encoder_layers', default=12, type=int)

args = parser.parse_args()

lstm_hidden_size = args.lstm_hidden_size 
lstm_num_layers = args.lstm_num_layers
transformer_nhead = args.transformer_nhead
transformer_num_encoder_layers = args.transformer_num_encoder_layers


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
        
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
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

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out
    
class DeepDanbooruModel(nn.Module):

    def __init__(self):
      
        super().__init__()
        
        self.resnet = ResNet(ResNetBlock, [3, 4, 23, 3])
        
        # LSTM
        self.lstm = nn.LSTM(2048, 
                            lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True)
                            
        # Transformer
        encoder_layer = TransformerEncoderLayer(lstm_hidden_size, 
                                               transformer_nhead)
                                               
        self.transformer_encoder = TransformerEncoder(encoder_layer,  
                                                     transformer_num_encoder_layers)
                                                     
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        x = self.resnet(x) 
        
        # LSTM
        x = x.squeeze(-1).squeeze(-1)
        x, _ = self.lstm(x)
        
        # Transformer 
        x = self.transformer_encoder(x)
        
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
       
model = DeepDanbooruModel()
