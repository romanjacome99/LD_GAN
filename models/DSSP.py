import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, optical_encoder):
        super(Block, self).__init__()
        self.optical_encoder = optical_encoder

        self.eta_parameter = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.eps_parameter = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act_2 = nn.ReLU(inplace=True)
        self.conv_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)

    def get_h(self, x):
        x1 = self.conv_1(x)
        x2 = self.act_2(self.conv_2(x1))
        x3 = self.conv_3(x + x2)
        x4 = self.conv_3(x3)

        return x4

    def get_gradient(self, x, y):
        y1 = self.optical_encoder(x, only_measurement=True)
        return self.optical_encoder(y1 - y, only_transpose=True)

    def forward(self, x, y):
        xh = torch.clamp(self.eta_parameter, min=0) * (x - self.get_h(x))
        x1 = x - torch.clamp(self.eps_parameter, min=0) * (self.get_gradient(x, y) + xh)

        return x1


class DSSP(nn.Module):
    def __init__(self, in_channels, out_channels, stages, optical_encoder):
        super(DSSP, self).__init__()
        self.in_channnels = in_channels
        self.out_channels = out_channels

        self.stages = stages
        self.optical_encoder = optical_encoder

        self.blocks = self.build_blocks(stages)

    def build_blocks(self, stages):
        blocks = []
        for i in range(stages):
            blocks.append(Block(self.in_channnels, self.out_channels, self.optical_encoder))

        return nn.ModuleList(blocks)

    def forward(self, x, y):
        for k in range(self.stages):
            x = self.blocks[k](x, y)

        return x
