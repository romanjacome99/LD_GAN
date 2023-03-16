import torch
import torch.nn as nn


class CompactUnet(nn.Module):
    def __init__(self, bands):
        super(CompactUnet, self).__init__()

        # ---------------- Encoder ---------------- #

        self.conv1_0 = nn.Conv2d(3, bands, 3, padding=1)
        self.act1_0 = nn.ReLU()
        self.conv1_1 = nn.Conv2d(bands, bands, 3, padding=1)
        self.act1_1 = nn.ReLU()

        self.down_2 = nn.MaxPool2d(2, 2)
        self.conv2_0 = nn.Conv2d(bands, 2 * bands, 3, padding=1)
        self.act2_0 = nn.ReLU()
        self.conv2_1 = nn.Conv2d(2 * bands, 2 * bands, 3, padding=1)
        self.act2_1 = nn.ReLU()

        self.down_3 = nn.MaxPool2d(2, 2)
        self.conv3_0 = nn.Conv2d(2 * bands, 3 * bands, 3, padding=1)
        self.act3_0 = nn.ReLU()
        self.conv3_1 = nn.Conv2d(3 * bands, 3 * bands, 3, padding=1)
        self.act3_1 = nn.ReLU()

        self.down_4 = nn.MaxPool2d(2, 2)
        self.conv4_0 = nn.Conv2d(3 * bands, 4 * bands, 3, padding=1)
        self.act4_0 = nn.ReLU()
        self.conv4_1 = nn.Conv2d(4 * bands, 4 * bands, 3, padding=1)
        self.act4_1 = nn.ReLU()

        # ---------------- Decoder ---------------- #

        self.up_5 = nn.Upsample(scale_factor=2)
        self.conv5_0 = nn.Conv2d(4 * bands, 3 * bands, 2, padding=0)  # (0, 1))
        self.act5_0 = nn.ReLU()
        self.conv5_1 = nn.Conv2d(3 * bands * 2, 3 * bands, 3, padding=1)
        self.act5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(3 * bands, 3 * bands, 3, padding=1)
        self.act5_2 = nn.ReLU()

        self.up_6 = nn.Upsample(scale_factor=2)
        self.conv6_0 = nn.Conv2d(3 * bands, 2 * bands, 2, padding=0)
        self.act6_0 = nn.ReLU()
        self.conv6_1 = nn.Conv2d(2 * bands * 2, 2 * bands, 3, padding=1)
        self.act6_1 = nn.ReLU()
        self.conv6_2 = nn.Conv2d(2 * bands, 2 * bands, 3, padding=1)
        self.act6_2 = nn.ReLU()

        self.up_7 = nn.Upsample(scale_factor=2)
        self.conv7_0 = nn.Conv2d(2 * bands, bands, 2, padding=0)
        self.act7_0 = nn.ReLU()
        self.conv7_1 = nn.Conv2d(bands * 2, bands, 3, padding=1)
        self.act7_1 = nn.ReLU()
        self.conv7_2 = nn.Conv2d(bands, bands, 3, padding=1)
        self.act7_2 = nn.ReLU()

        self.conv10 = nn.Conv2d(bands, bands, 1, padding=0)
        self.act10 = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight.data)


    def forward(self, x):
        # ---------------- Encoder ---------------- #
        x1 = self.act1_0(self.conv1_0(x))
        x1 = self.act1_1(self.conv1_1(x1))

        x2 = self.down_2(x1)
        x2 = self.act2_0(self.conv2_0(x2))
        x2 = self.act2_1(self.conv2_1(x2))

        x3 = self.down_3(x2)
        x3 = self.act3_0(self.conv3_0(x3))
        x3 = self.act3_1(self.conv3_1(x3))

        x4 = self.down_4(x3)
        x4 = self.act4_0(self.conv4_0(x4))
        x4 = self.act4_1(self.conv4_1(x4))

        # ---------------- Decoder ---------------- #

        x5 = self.act5_0(self.conv5_0(self.up_5(x4)))
        x5 = nn.functional.pad(x5, (0, 1, 0, 1))
        x5 = torch.cat((x5, x3), dim=1)
        x5 = self.act5_1(self.conv5_1(x5))
        x5 = self.act5_2(self.conv5_2(x5))

        x6 = self.act6_0(self.conv6_0(self.up_6(x5)))
        x6 = nn.functional.pad(x6, (0, 1, 0, 1))
        x6 = torch.cat((x6, x2), dim=1)
        x6 = self.act6_1(self.conv6_1(x6))
        x6 = self.act6_2(self.conv6_2(x6))

        x7 = self.act7_0(self.conv7_0(self.up_7(x6)))
        x7 = nn.functional.pad(x7, (0, 1, 0, 1))
        x7 = torch.cat((x7, x1), dim=1)
        x7 = self.act7_1(self.conv7_1(x7))
        x7 = self.act7_2(self.conv7_2(x7))

        x8 = self.act10(self.conv10(x7))

        return x8


import torch
import torch.nn as nn


# UNet with PixelShuffle & MaxPool

class double_conv(nn.Module):
    def __init__(self, input_features, output_features, negative_slope=0, p_drop=0):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_features, output_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Dropout2d(p=p_drop),
            nn.Conv2d(output_features, output_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down_block(nn.Module):
    def __init__(self, input_features, output_features, negative_slope=0, p_drop=0):
        super(down_block, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=p_drop),
            double_conv(input_features, output_features, negative_slope, p_drop)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class up_block(nn.Module):
    def __init__(self, input_features, output_features, negative_slope=0, p_drop=0):
        super(up_block, self).__init__()
        self.up = nn.PixelShuffle(upscale_factor=2)
        self.conv = nn.Sequential(
            nn.Dropout2d(p=p_drop),
            double_conv(int(input_features / 4 + output_features), output_features, negative_slope, p_drop)
        )

    def forward(self, x, x_pre):
        x = self.up(x)
        x = torch.cat((x, x_pre), 1)
        x = self.conv(x)
        return x


class UnetModel(nn.Module):
    def __init__(self, input_features, output_features, negative_slope=0, p_drop=0):
        super(UnetModel, self).__init__()
        self.in_conv = double_conv(input_features, 64, negative_slope, p_drop)
        self.down1 = down_block(64, 128, negative_slope, p_drop)  # W/2
        self.down2 = down_block(128, 256, negative_slope, p_drop)  # W/4
        self.down3 = down_block(256, 512, negative_slope, p_drop)  # W/8
        self.bottleneck = down_block(512, 1024, negative_slope, p_drop)  # W/16
        self.up1 = up_block(1024, 512, negative_slope, p_drop)  # W/8
        self.up2 = up_block(512, 256, negative_slope, p_drop)  # W/4
        self.up3 = up_block(256, 128, negative_slope, p_drop)  # W/2
        self.out_conv = up_block(128, 64, negative_slope, p_drop)  # W
        self.out = nn.Conv2d(64, output_features, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        conv = self.in_conv(x)  # W 64
        down1 = self.down1(conv)  # W/2 128
        down2 = self.down2(down1)  # W/4 256
        down3 = self.down3(down2)  # W/8 512
        bottleneck = self.bottleneck(down3)  # W/16 1024
        up = self.up1(bottleneck, down3)  # W/8 512
        up = self.up2(up, down2)  # W/4 256
        up = self.up3(up, down1)  # W/2 128
        up = self.out_conv(up, conv)  # W 64
        return self.out(up)


class UnetCompiled(nn.Module):
    def __init__(self, bands):
        super(UnetCompiled, self).__init__()

        # ---------------- Encoder ---------------- #

        self.conv1_0 = nn.Conv2d(bands, 32, 3, padding=1)
        self.act1_0 = nn.ReLU()
        self.conv1_1 = nn.Conv2d(32, 32, 3, padding=1)
        self.act1_1 = nn.ReLU()
        self.bn_1 = nn.BatchNorm2d(32)

        self.down_2 = nn.MaxPool2d(2, 2)
        self.conv2_0 = nn.Conv2d(32, 64, 3, padding=1)
        self.act2_0 = nn.ReLU()
        self.conv2_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.act2_1 = nn.ReLU()
        self.bn_2 = nn.BatchNorm2d(64)

        self.down_3 = nn.MaxPool2d(2, 2)
        self.conv3_0 = nn.Conv2d(64, 128, 3, padding=1)
        self.act3_0 = nn.ReLU()
        self.conv3_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.act3_1 = nn.ReLU()
        self.bn_3 = nn.BatchNorm2d(128)

        self.down_4 = nn.MaxPool2d(2, 2)
        self.conv4_0 = nn.Conv2d(128, 256, 3, padding=1)
        self.act4_0 = nn.ReLU()
        self.conv4_1 = nn.Conv2d(256, 256, 3, padding=1)
        self.act4_1 = nn.ReLU()
        self.bn_4 = nn.BatchNorm2d(256)
        self.drop_4 = nn.Dropout2d(0.3)

        self.down_5 = nn.MaxPool2d(2, 2)
        self.conv5_0 = nn.Conv2d(256, 512, 3, padding=1)
        self.act5_0 = nn.ReLU()
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.act5_1 = nn.ReLU()
        self.bn_5 = nn.BatchNorm2d(512)
        self.drop_5 = nn.Dropout2d(0.3)

        # ---------------- Decoder ---------------- #

        self.convt6_0 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.conv6_0 = nn.Conv2d(512, 256, 3, padding=1)
        self.act6_0 = nn.ReLU()
        self.conv6_1 = nn.Conv2d(256, 256, 3, padding=1)
        self.act6_1 = nn.ReLU()

        self.convt7_0 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.conv7_0 = nn.Conv2d(256, 128, 3, padding=1)
        self.act7_0 = nn.ReLU()
        self.conv7_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.act7_1 = nn.ReLU()

        self.convt8_0 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv8_0 = nn.Conv2d(128, 64, 3, padding=1)
        self.act8_0 = nn.ReLU()
        self.conv8_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.act8_1 = nn.ReLU()

        self.convt9_0 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv9_0 = nn.Conv2d(64, 32, 3, padding=1)
        self.act9_0 = nn.ReLU()
        self.conv9_1 = nn.Conv2d(32, 32, 3, padding=1)
        self.act9_1 = nn.ReLU()

        self.conv10_0 = nn.Conv2d(32, bands, 3, padding=1)
        self.act10_0 = nn.ReLU()
        self.conv10_1 = nn.Conv2d(bands, bands, 3, padding=1)

    def forward(self, x):
        # ---------------- Encoder ---------------- #
        x1 = self.act1_0(self.conv1_0(x))
        x1 = self.act1_1(self.conv1_1(x1))
        x1 = self.bn_1(x1)

        x2 = self.down_2(x1)
        x2 = self.act2_0(self.conv2_0(x2))
        x2 = self.act2_1(self.conv2_1(x2))
        x2 = self.bn_2(x2)

        x3 = self.down_3(x2)
        x3 = self.act3_0(self.conv3_0(x3))
        x3 = self.act3_1(self.conv3_1(x3))
        x3 = self.bn_3(x3)

        x4 = self.down_4(x3)
        x4 = self.act4_0(self.conv4_0(x4))
        x4 = self.act4_1(self.conv4_1(x4))
        x4 = self.drop_4(self.bn_4(x4))

        x5 = self.down_5(x4)
        x5 = self.act5_0(self.conv5_0(x5))
        x5 = self.act5_1(self.conv5_1(x5))
        x5 = self.drop_5(self.bn_5(x5))

        # ---------------- Decoder ---------------- #

        x6 = self.convt6_0(x5)
        x6 = torch.cat((x6, x4), dim=1)
        x6 = self.act6_0(self.conv6_0(x6))
        x6 = self.act6_1(self.conv6_1(x6))

        x7 = self.convt7_0(x6)
        x7 = torch.cat((x7, x3), dim=1)
        x7 = self.act7_0(self.conv7_0(x7))
        x7 = self.act7_1(self.conv7_1(x7))

        x8 = self.convt8_0(x7)
        x8 = torch.cat((x8, x2), dim=1)
        x8 = self.act8_0(self.conv8_0(x8))
        x8 = self.act8_1(self.conv8_1(x8))

        x9 = self.convt9_0(x8)
        x9 = torch.cat((x9, x1), dim=1)
        x9 = self.act9_0(self.conv9_0(x9))
        x9 = self.act9_1(self.conv9_1(x9))

        x10 = self.act10_0(self.conv10_0(x9))
        x10 = self.conv10_1(x10)

        return x10
