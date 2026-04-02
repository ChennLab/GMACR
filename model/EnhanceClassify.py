import math
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
from .CBAM import CBAM

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False, dropout_rate=0.2):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM(planes, 16)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.dropout(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out = out + residual
        out = self.relu(out)

        return out

class Classifier(nn.Module):
    def __init__(self, block, layers, num_classes, ch, att_type=None, dropout_rate=0.5):
        super(Classifier, self).__init__()
        self.inplanes = ch
        self.feature_dim = ch * 8
        self.num_classes = num_classes
        self.hyper = 0.5

        # 假设输入都是单通道，如果残差图是多通道需要修改 in_channels
        self.conv1_int = nn.Conv2d(1, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_int = nn.BatchNorm2d(ch)

        self.conv1_p = nn.Conv2d(1, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_p = nn.BatchNorm2d(ch)

        self.relu = nn.ReLU(inplace=True)

        # --- ResNet Layers ---
        # Layer 1 (输出通道 ch)
        self.layer1 = self._make_layer(block, ch, layers[0], att_type=att_type)

        # Layer 2 (输出通道 ch*2, stride=2 进行下采样)
        self.layer2 = self._make_layer(block, ch * 2, layers[1], stride=2, att_type=att_type)
        self.downsample_p_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.channel_adjust_p_2 = nn.Conv2d(ch, ch * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_adjust_p_2 = nn.BatchNorm2d(ch * 2)
        self.bn_int_2 = nn.BatchNorm2d(ch * 2)

        # Layer 3 (输出通道 ch*4, stride=2 进行下采样)
        self.layer3 = self._make_layer(block, ch * 4, layers[2], stride=2, att_type=att_type)
        self.downsample_p_3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.channel_adjust_p_3 = nn.Conv2d(ch * 2, ch * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_adjust_p_3 = nn.BatchNorm2d(ch * 4)
        self.bn_int_3 = nn.BatchNorm2d(ch * 4)

        # Layer 4 (输出通道 ch*8, stride=2 进行下采样)
        self.layer4 = self._make_layer(block, ch * 8, layers[3], stride=2, att_type=att_type)
        self.downsample_p_4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.channel_adjust_p_4 = nn.Conv2d(ch * 4, ch * 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_adjust_p_4 = nn.BatchNorm2d(ch * 8)
        self.bn_int_4 = nn.BatchNorm2d(ch * 8)

        # 1. 全局平均池化
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 投影头 (用于 SupCon Loss)
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # 在第一个线性层之后，激活函数之后
            nn.Linear(self.feature_dim // 2, 128)
        )

        # 分类头
        # self.classification_head = nn.Linear(self.feature_dim, self.num_classes)
        self.classification_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # 在第一个线性层之后，激活函数之后
            nn.Linear(self.feature_dim // 2, self.num_classes)
        )


        # --- 初始化权重 (可选，但推荐) ---
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None, block_dropout_rate=0.5):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type, dropout_rate=block_dropout_rate))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type == 'CBAM', dropout_rate=block_dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x, res):

        # --- Initial Convolution ---
        x_int = self.relu(self.bn1_int(self.conv1_int(x)))  # 原图的特征
        x_res = self.relu(self.bn1_p(self.conv1_p(res)))  # 反事实指导图的特征

        # --- Stage 1 ---
        x_int = self.layer1(x_int)

        # --- Stage 2 ---
        x_int = self.layer2(x_int)
        x_p = self.bn_adjust_p_2(self.channel_adjust_p_2(self.downsample_p_2(x_res)))
        # x_int = x_int * ((1 + self.hyper) * x_p)  # 乘法注入
        # x_int = x_int + ((1 + self.hyper) * x_p)  # 加法注入
        # x_int = x_int + x_p  # 加法注入
        # x_int = x_int + 0.5 * torch.sigmoid(x_p)  # 加法注入
        x_int = x_int * torch.sigmoid(x_p)  # 加法注入
        x_int = self.relu(self.bn_int_2(x_int))

        # --- Stage 3 ---
        x_int = self.layer3(x_int)
        x_p = self.bn_adjust_p_3(self.channel_adjust_p_3(self.downsample_p_3(x_p)))
        # x_int = x_int * ((1 + self.hyper) * x_p)  # 乘法注入
        # x_int = x_int + ((1 + self.hyper) * x_p)  # 加法注入
        # x_int = x_int + x_p  # 加法注入
        # x_int = x_int + 0.5 * torch.sigmoid(x_p)  # 加法注入
        x_int = x_int * torch.sigmoid(x_p)  # 加法注入
        x_int = self.relu(self.bn_int_3(x_int))

        # --- Stage 4 ---
        x_int = self.layer4(x_int)
        x_p = self.bn_adjust_p_4(self.channel_adjust_p_4(self.downsample_p_4(x_p)))
        # x_int = x_int * ((1 + self.hyper) * x_p)  # 乘法注入
        # x_int = x_int + ((1 + self.hyper) * x_p)  # 加法注入
        # x_int = x_int + x_p  # 加法注入
        # x_int = x_int + 0.5 * torch.sigmoid(x_p)  # 加法注入
        x_int = x_int * torch.sigmoid(x_p)  # 加法注入
        x_int = self.relu(self.bn_int_4(x_int))

        # --- Final Pooling and Heads ---
        # 使用最终融合后的 x_int
        z = self.pool(x_int)
        feature_pooled = torch.flatten(z, 1)

        # Classification score
        score = self.classification_head(feature_pooled)
        # score = F.softmax(score, dim=1) # Softmax 通常在计算损失函数时应用(如 CrossEntropyLoss)

        # Projection feature for SupCon
        feature_projected = self.projection_head(feature_pooled)
        feature_projected = F.normalize(feature_projected, dim=1)

        return score, feature_projected
