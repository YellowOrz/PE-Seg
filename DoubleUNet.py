import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F


class Squeeze_Excite(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 等价于Global Average Pooling
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
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(inplace=True))
        self.SE = Squeeze_Excite(out_channels)

    def forward(self, x):
        return self.SE(self.feature(x))


# https://blog.csdn.net/free1993/article/details/108880686
# class Hooker:
#     features = None
#
#     def __init__(self, model, layer_num):
#         self.hook = model[layer_num].register_forward_hook(self.hook_fn)
#
#     def hook_fn(self, module, input, output):
#         self.features = output.cpu()
#
#     def remove(self):
#          self.hook.remove()

# skip_connections = []
# def get_output(module, inputs, outputs):
#     skip_connections.append(outputs)
# class Encoder1(nn.Module):  # 验证：输出大小正确
#     def __init__(self):
#         super().__init__()
#         self.base_model = tv.models.vgg19(pretrained=True, progress=True)
#         self.features = self.base_model.features
#
#     def forward(self, x):
#         skip_connections.clear()
#         layers = [3, 8, 17, 26, 35]  # tf中卷积自带激活函数
#         for l in layers:
#             self.features[l].register_forward_hook(get_output)
#             # conv_out = Hooker(self.features, i)
#             # skip_connections.append(conv_out.features)
#             # conv_out.remove()
#         self.features(x)
#         output = skip_connections.pop()  # features中最后一层为MaxPool，而output应该是最后的ReLU的输出
#         return output, skip_connections.copy()

# Encoder1的另一种实现方式
class Encoder1(nn.Module):  # 验证：输出大小正确
    def __init__(self):
        super().__init__()
        self.feature_list = [3, 8, 17, 26]
        base_model = tv.models.vgg19(pretrained=True, progress=True).features
        self.model = nn.Sequential(*list(base_model.children())[:len(base_model)-1])

    def forward(self, x):
        features = []
        for i, layer in enumerate(list(self.model)):
            x = layer(x)
            if i in self.feature_list:
                features.append(x)
        return x, features


class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ASPP, self).__init__()
        self.block1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Conv2d(in_channel, out_channel, 1),
                                    # nn.BatchNorm2d(out_channel),  # pytroch中单个通道只有一个元素的话不能用BN
                                    nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, bias=False),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU())
        self.block3 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=6, dilation=6, bias=False),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU())
        self.block4 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=12, dilation=12, bias=False),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU())
        self.block5 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=18, dilation=18, bias=False),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU())
        self.block6 = nn.Sequential(nn.Conv2d(out_channel * 5, out_channel, 1, bias=False),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU())

    def forward(self, x):
        size = x.shape[2:]
        y1 = self.block1(x)
        # y1 = nn.functional.upsample(y1, size=size, mode='bilinear') # nn.functional.upsample is deprecated. Use nn.functional.interpolate instead
        y1 = nn.functional.interpolate(y1, size=size, mode='bilinear', align_corners=False)
        y2 = self.block2(x)
        y3 = self.block3(x)
        y4 = self.block4(x)
        y5 = self.block5(x)
        y = torch.cat([y1, y2, y3, y4, y5], dim=1)
        output = self.block6(y)
        return output


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

    def forward(self, x, skip1):
        for i, skip in enumerate(skip1):
            x = self.upsample[i](x)
            x = torch.cat([x, skip], dim=1)
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

    # def forward(self, inputs):
    #     skip_connections.clear()
    #     x = inputs
    #     for i in range(len(self.conv_block)):
    #         x = self.conv_block[i](x)
    #         skip_connections.append(x)
    #         x = self.maxpool[i](x)
    #     return x, skip_connections.copy()

    # 根据Encoder1的另一种实现方式这里也要修改
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
    def __init__(self):
        super().__init__()
        self.encoder1 = Encoder1()
        self.aspp1 = ASPP(512, 64)  # 512是VGG（encoder1）输出的通道数，64跟源码保持一致（应该为256比较妥当叭？）
        self.decoder1 = Decoder1(64)  # 输出通道数为
        self.encoder2 = Encoder2(3)
        self.aspp2 = ASPP(256, 64)
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


