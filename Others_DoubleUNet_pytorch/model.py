import torch
import torch.nn as nn


class Squeeze_Excite(nn.Module):

    def __init__(self, channel, reduction):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class VGGBlock(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.SE = Squeeze_Excite(out_channels, 8)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out - self.relu(out)
        out = self.SE(out)

        return out


# U-Net
class UNet(nn.Module):

    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


# U-Net++
class NestedUNet(nn.Module):

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv3_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.cov0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.cov1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.cov0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


# DoubleU-Net
class DoubleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # VGGBlock = (Conv + bn + relu)* 2 + squeeze_excite
        self.conv1 = VGGBlock(3, 64, 64)
        self.conv2 = VGGBlock(64, 128, 128)
        self.conv3 = VGGBlock(128, 256, 256)
        self.conv4 = VGGBlock(256, 512, 512)
        self.conv5 = VGGBlock(512, 512, 512)

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.Vgg1 = VGGBlock(1024, 256, 256)
        self.Vgg2 = VGGBlock(512, 128, 128)
        self.Vgg3 = VGGBlock(256, 64, 64)
        self.Vgg4 = VGGBlock(128, 32, 32)

        self.out = output_block()

        self.conv11 = VGGBlock(6, 32, 32)
        self.conv12 = VGGBlock(32, 64, 64)
        self.conv13 = VGGBlock(64, 128, 128)
        self.conv14 = VGGBlock(128, 256, 256)

        self.Vgg5 = VGGBlock(1024, 256, 256)
        self.Vgg6 = VGGBlock(640, 128, 128)
        self.Vgg7 = VGGBlock(320, 64, 64)
        self.Vgg8 = VGGBlock(160, 32, 32)

        self.out1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1))

    def forward(self, x):
        # UNet1: encoder
        x1 = self.conv1(x)  # conv1-5 = VGGBlock
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))
        # ？？？这里应该有一个ASPP

        # UNet1: decoder
        x5 = self.up(x5)    # up = UPSample
        x5 = torch.cat([x5, x4], 1) # cat用于拼接
        # Vgg1-8 = VGGBlock
        x6 = self.Vgg1(x5)
        x6 = self.up(x6)
        x6 = torch.cat([x6, x3], 1)
        x7 = self.Vgg2(x6)
        x7 = self.up(x7)
        x7 = torch.cat([x7, x2], 1)
        x8 = self.Vgg3(x7)
        x8 = self.up(x8)
        x8 = torch.cat([x8, x1], 1)
        x9 = self.Vgg4(x8)

        output1 = self.out(x9)  # out = conv + Sigmoid
        output1 = x * output1   # MULTIPLY  # 这里不能叫output1！！！不然最后的CONCATENATE就用错了！！！

        x = torch.cat([x, output1], 1)  # 这里跟论文说的不一样啊。论文说的是把output1和x相乘后作为输入，没有cat这一步骤

        # UNet2: encoder
        x11 = self.conv11(x)
        x12 = self.conv12(self.pool(x11))
        x13 = self.conv13(self.pool(x12))
        x14 = self.conv14(self.pool(x13))
        y = self.pool(x14)

        # UNet2: decoder
        y = self.up(y)
        y = torch.cat([y, x14, x4], 1)
        y = self.Vgg5(y)
        y = self.up(y)
        y = torch.cat([y, x13, x3], 1)
        y = self.Vgg6(y)
        y = self.up(y)
        y = torch.cat([y, x12, x2], 1)
        y = self.Vgg7(y)
        y = self.up(y)
        y = torch.cat([y, x11, x1], 1)
        y = self.Vgg8(y)
        output2 = self.out1(y)
        return output2


def output_block():
    layer = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1)),
                          nn.Sigmoid())
    return layer


if __name__ == '__main__':
    model = DoubleUNet()
    x = torch.randn(1, 3, 512, 512)
    model(x)
