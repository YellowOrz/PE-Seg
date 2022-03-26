import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
from typing import List


class Squeeze_Excite(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)      # 等价于Global Average Pooling
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.feature = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                     nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True),
                                     nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                     nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True),
                                     Squeeze_Excite(out_channels))

    def forward(self, x):
        return self.feature(x)


class Encoder1(nn.Module):  # 没有问题，by xzf
    def __init__(self, requires_grad=False, BN=False):
        super().__init__()
        if BN:
            self.feature_list = [5, 12, 25, 38]
            base_model = tv.models.vgg19_bn(pretrained=True).features
        else:
            self.feature_list = [3, 8, 17, 26]
            base_model = tv.models.vgg19(pretrained=True).features
        self.model = nn.Sequential(*list(base_model.children())[:len(base_model)-1])
        if not requires_grad:       # vgg无需梯度更新
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = []
        for i, layer in enumerate(list(self.model)):
            x = layer(x)
            if i in self.feature_list:
                features.append(x)
        return x, features


class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel, rate):
        super(ASPP, self).__init__()

        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Conv2d(in_channel, out_channel, 1),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, bias=False),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU(inplace=True)))
        for r in rate:
            self.blocks.append(nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=r, dilation=r, bias=False),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU(inplace=True)))
        self.project = nn.Sequential(nn.Conv2d(out_channel * (len(self.blocks)+1), out_channel, 1, bias=False),
                                     nn.BatchNorm2d(out_channel),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[2:]
        y = []
        y.append(nn.functional.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=False))
        for block in self.blocks:
            y.append(block(x))
        y = torch.cat(y, dim=1)
        return self.project(y)


# class ASPP(nn.Module):
#     # https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py
#     def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
#         super(ASPP, self).__init__()
#         modules = []
#         modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True)))
#
#         rates = tuple(atrous_rates)
#         for rate in rates:
#             modules.append(ASPPConv(in_channels, out_channels, rate))
#         modules.append(ASPPPooling(in_channels, out_channels))
#
#         self.convs = nn.ModuleList(modules)
#         self.project = nn.Sequential(nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True),nn.Dropout(0.5))
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         _res = []
#         for conv in self.convs:
#             _res.append(conv(x))
#         res = torch.cat(_res, dim=1)
#         return self.project(res)
#
#
# class ASPPConv(nn.Sequential):
#     def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
#         modules = [nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
#             nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True)]
#         super(ASPPConv, self).__init__(*modules)
#
#
# class ASPPPooling(nn.Sequential):
#     def __init__(self, in_channels: int, out_channels: int) -> None:
#         super(ASPPPooling, self).__init__(nn.AdaptiveAvgPool2d(1),nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         size = x.shape[-2:]
#         for mod in self:
#             x = mod(x)
#         return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class Decoder1(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        skp1_channels = [512, 256, 128, 64]
        num_filters = [256, 128, 64, 32]
        self.conv_block = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i, out_channels in enumerate(num_filters):
            self.upsample.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            self.conv_block.append(Conv_Block(in_channels + skp1_channels[i], out_channels))
            in_channels = out_channels

    def forward(self, x, skip):
        for i, s in enumerate(skip):
            x = self.upsample[i](x)
            x = torch.cat([x, s], dim=1)
            x = self.conv_block[i](x)
        return x


class Encoder2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        num_filters = [32, 64, 128, 256]
        self.conv_block = nn.ModuleList()
        self.maxpool = nn.ModuleList()
        for out_channels in num_filters:
            self.maxpool.append(nn.MaxPool2d(2))
            self.conv_block.append(Conv_Block(in_channels, out_channels))
            in_channels = out_channels

    # 根据Encoder1的另一种实现方式    # TODO:这里也要修改???
    def forward(self, x):
        features = []
        for i in range(len(self.conv_block)):
            x = self.conv_block[i](x)
            features.append(x)
            x = self.maxpool[i](x)
        return x, features


class Decoder2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        skp1_channels = [512, 256, 128, 64]
        skp2_channels = [256, 128, 64, 32]
        num_filters = [256, 128, 64, 32]
        self.conv_block = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i, out_channels in enumerate(num_filters):
            self.upsample.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            self.conv_block.append(Conv_Block(in_channels + skp1_channels[i] + skp2_channels[i], out_channels))
            in_channels = out_channels

    def forward(self, x, skip1, skip2):
        for i in range(len(skip2)):
            x = self.upsample[i](x)
            x = torch.cat([x, skip1[i], skip2[i]], dim=1)
            x = self.conv_block[i](x)
        return x


class DoubleUNet(nn.Module):
    def __init__(self, VGG_BN):
        super().__init__()
        self.encoder1 = Encoder1(requires_grad=True, BN=VGG_BN)
        self.aspp1 = ASPP(512, 64, [6,12,18])  # 512是VGG（encoder1）输出的通道数，64跟源码保持一致（应该为256比较妥当叭？）
        self.decoder1 = Decoder1(64)  # 输出通道数为
        self.encoder2 = Encoder2(3)
        self.aspp2 = ASPP(256, 64, [6,12,18])
        self.decoder2 = Decoder2(64)
        self.output_block1 = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())
        self.output_block2 = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())  # 可以共享一个output_block吗？

    def forward(self, inputs):
        # UNet1
        x = inputs
        x, skip_1 = self.encoder1(x)
        skip_1.reverse()
        x = self.aspp1(x)
        x = self.decoder1(x, skip_1)
        output1 = self.output_block1(x)
        # UNet2
        x = inputs * output1
        x, skip_2 = self.encoder2(x)
        skip_2.reverse()
        x = self.aspp2(x)
        x = self.decoder2(x, skip_1, skip_2)
        output2 = self.output_block2(x)

        return torch.cat([output1, output2], dim=1)


if __name__ == '__main__':
    x = torch.rand(1, 3, 192, 256)  # N*C*H*W
    net = DoubleUNet()
    output = net(x)
    print(output)


