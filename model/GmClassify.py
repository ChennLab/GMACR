import math
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
from .CBAM_GM import CBAM

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM(planes, 16)
        else:
            self.cbam = None

    def forward(self, x, gm):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out, gm)

        out = out + residual
        out = self.relu(out)

        return out

class Classifier(nn.Module):
    def __init__(self, block, layers, num_classes, ch, att_type=None):
        super(Classifier, self).__init__()
        self.inplanes = ch
        self.feature_dim = ch * 8
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)

        # ------------------------------------------------------Layer 1
        self.layer1_blocks = nn.ModuleList()
        current_inplanes = ch
        current_outplanes = ch
        # Layer 1 stride is 1, no downsample needed for BasicBlock input/output planes match
        self.layer1_blocks.append(block(current_inplanes, current_outplanes, stride=1, downsample=None,
                                        use_cbam=att_type))  # 第一个块
        current_inplanes = current_outplanes * block.expansion  # 更新 inplanes for next block
        for i in range(1, layers[0]):
            self.layer1_blocks.append(block(current_inplanes, current_outplanes, stride=1, downsample=None,
                                            use_cbam=att_type))  # 剩余块

        # -------------------------------------------------------Layer 2
        self.layer2_blocks = nn.ModuleList()
        current_outplanes = ch * 2
        # Downsample layer 2 input
        self.layer2_downsample = nn.Sequential(
            nn.Conv2d(current_inplanes, current_outplanes * block.expansion, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(current_outplanes * block.expansion),
        )
        self.layer2_blocks.append(
            block(current_inplanes, current_outplanes, stride=2, downsample=self.layer2_downsample,
                  use_cbam=att_type))  # 第一个块
        current_inplanes = current_outplanes * block.expansion
        for i in range(1, layers[1]):
            self.layer2_blocks.append(block(current_inplanes, current_outplanes, stride=1, downsample=None,
                                            use_cbam=att_type))  # 剩余块

        # -------------------------------------------------------Layer 3
        self.layer3_blocks = nn.ModuleList()
        current_outplanes = ch * 4
        # Downsample layer 3 input
        self.layer3_downsample = nn.Sequential(
            nn.Conv2d(current_inplanes, current_outplanes * block.expansion, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(current_outplanes * block.expansion),
        )
        self.layer3_blocks.append(
            block(current_inplanes, current_outplanes, stride=2, downsample=self.layer3_downsample,
                  use_cbam=att_type))  # 第一个块
        current_inplanes = current_outplanes * block.expansion
        for i in range(1, layers[2]):
            self.layer3_blocks.append(block(current_inplanes, current_outplanes, stride=1, downsample=None,
                                            use_cbam=att_type))  # 剩余块

        # -------------------------------------------------------Layer 4
        self.layer4_blocks = nn.ModuleList()
        current_outplanes = ch * 8
        # Downsample layer 4 input
        self.layer4_downsample = nn.Sequential(
            nn.Conv2d(current_inplanes, current_outplanes * block.expansion, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(current_outplanes * block.expansion),
        )
        self.layer4_blocks.append(
            block(current_inplanes, current_outplanes, stride=2, downsample=self.layer4_downsample,
                  use_cbam=att_type))  # 第一个块
        current_inplanes = current_outplanes * block.expansion
        for i in range(1, layers[3]):
            self.layer4_blocks.append(block(current_inplanes, current_outplanes, stride=1, downsample=None,
                                            use_cbam=att_type))  # 剩余块

        # 1. 全局平均池化
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 投影头 (用于 SupCon Loss)
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, 128)
        )

        # 分类头
        self.classification_head = nn.Linear(self.feature_dim, self.num_classes)

    def forward(self, x, gm_mask):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        gm = gm_mask.float()  # 确保 GM 是 float
        if gm.ndim == 3:
            gm = gm.unsqueeze(1)  # Add channel dim if missing

        current_spatial_size = x.shape[-2:]
        gm_layer1 = F.interpolate(gm, size=current_spatial_size, mode='bilinear', align_corners=False)
        for block in self.layer1_blocks:
            # 将特征和下采样到当前层分辨率的 GM 掩码传递给块 (如果使用 GM 集成 CBAM)
            x = block(x, gm_layer1)

        current_spatial_size = (current_spatial_size[0] // 2, current_spatial_size[1] // 2)
        gm_layer2 = F.interpolate(gm, size=current_spatial_size, mode='bilinear', align_corners=False)
        for block in self.layer2_blocks:
            # 将特征和下采样到当前层分辨率的 GM 掩码传递给块 (如果使用 GM 集成 CBAM)
            x = block(x, gm_layer2)

        current_spatial_size = (current_spatial_size[0] // 2, current_spatial_size[1] // 2)
        gm_layer3 = F.interpolate(gm, size=current_spatial_size, mode='bilinear', align_corners=False)
        for block in self.layer3_blocks:
            # 将特征和下采样到当前层分辨率的 GM 掩码传递给块 (如果使用 GM 集成 CBAM)
            x = block(x, gm_layer3)

        current_spatial_size = (current_spatial_size[0] // 2, current_spatial_size[1] // 2)
        gm_layer4 = F.interpolate(gm, size=current_spatial_size, mode='bilinear', align_corners=False)
        for block in self.layer4_blocks:
            # 将特征和下采样到当前层分辨率的 GM 掩码传递给块 (如果使用 GM 集成 CBAM)
            x = block(x, gm_layer4)

        z = self.pool(x)  # (128, 512, 1, 1)
        feature = torch.flatten(z, 1)  # (128, 512)

        score = self.classification_head(feature)  # (128, 2)
        score = F.softmax(score, dim=1)  # (128, 2)

        feature = self.projection_head(feature)  # (128, 128)
        feature = F.normalize(feature, dim=1)  # (128, 128)

        return score, feature