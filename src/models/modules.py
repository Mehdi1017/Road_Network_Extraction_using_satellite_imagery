import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.encoders import get_encoder

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))

class D3S2PP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super().__init__()
        self.conv1x1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.ac1 = DepthwiseSeparableConv(in_channels, out_channels, dilation=atrous_rates[0], padding=atrous_rates[0])
        self.ac2 = DepthwiseSeparableConv(in_channels, out_channels, dilation=atrous_rates[1], padding=atrous_rates[1])
        self.ac3 = DepthwiseSeparableConv(in_channels, out_channels, dilation=atrous_rates[2], padding=atrous_rates[2])
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.project = nn.Sequential(nn.Conv2d(out_channels * 5, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout(0.5))

    def forward(self, x):
        res = [self.conv1x1(x), self.ac1(x), self.ac2(x), self.ac3(x)]
        global_feat = F.interpolate(self.global_avg_pool(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(global_feat)
        return self.project(torch.cat(res, dim=1))

class DeepLabV3PlusD3S2PP(SegmentationModel):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", classes=1):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=3, depth=5, weights=encoder_weights)
        encoder_channels = self.encoder.out_channels
        self.d3s2pp = D3S2PP(in_channels=encoder_channels[-1], out_channels=256)
        
        self.low_level_project = nn.Sequential(nn.Conv2d(encoder_channels[1], 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU())
        self.final_convs = nn.Sequential(nn.Conv2d(304, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.segmentation_head = nn.Sequential(nn.Conv2d(256, classes, kernel_size=1))
        self.classification_head = None

    def forward(self, x):
        features = self.encoder(x)
        x_high = F.interpolate(self.d3s2pp(features[-1]), scale_factor=4, mode='bilinear', align_corners=False)
        x_low = self.low_level_project(features[1])
        if x_high.shape[-2:] != x_low.shape[-2:]:
            x_high = F.interpolate(x_high, size=x_low.shape[-2:], mode='bilinear', align_corners=False)
        x_out = self.final_convs(torch.cat([x_high, x_low], dim=1))
        logits = self.segmentation_head(x_out)
        return F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)