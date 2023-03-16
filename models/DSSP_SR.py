import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels,sdf,ldf):
        super(Block, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.L = in_channels
        self.sdf = sdf
        self.ldf = ldf
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
    def x2y(self, x):
        kernel = self.SpectralDegradationFilter(3, self.L, self.ldf).to(self.device)
        low_spectral = F.conv2d(x, kernel, padding=1)
        y =  nn.AvgPool2d(kernel_size=(int(self.sdf),int(self.sdf)))(low_spectral)
        return y

    def y2x(self, y):
        kernel = self.SpectralUpsamplingFilter(3, self.ldf, self.L).to(self.device)
        hs = F.interpolate(y,scale_factor=(self.sdf,self.sdf))
        x =   F.conv2d(hs,kernel,padding=1)
        return x

    def SpectralDegradationFilter(self, window_size, L, q):
        kernel = torch.zeros((L // q, L, window_size, window_size))
        for i in range(0, L // q):
            kernel[i, i * q:(i + 1) * (q), window_size // 2, window_size // 2] = 1 / q
        return kernel


    def SpectralUpsamplingFilter(self, window_size, q, L):
        kernel = torch.zeros((L, L // q, window_size, window_size))
        for i in range(0, L // q):
            for j in range(0, q):
                kernel[i * q + j, i, window_size // 2, window_size // 2] = 1
        return kernel


    def get_gradient(self, x, y):
        y1 = self.x2y(x)
        return self.y2x(y1 - y)

    def forward(self, x, y):
        xh = torch.clamp(self.eta_parameter, min=0) * (x - self.get_h(x))
        x1 = x - torch.clamp(self.eps_parameter, min=0) * (self.get_gradient(x, y) + xh)

        return x1


class DSSP_SR(nn.Module):
    def __init__(self, in_channels, out_channels, stages,sdf,ldf):
        super(DSSP_SR, self).__init__()
        self.in_channnels = in_channels
        self.out_channels = out_channels
        self.sdf = sdf
        self.ldf = ldf
        self.stages = stages

        self.blocks = self.build_blocks(stages)

    def build_blocks(self, stages):
        blocks = []
        for i in range(stages):
            blocks.append(Block(self.in_channnels, self.out_channels,self.sdf,self.ldf))

        return nn.ModuleList(blocks)

    def forward(self, x, y):
        for k in range(self.stages):
            x = self.blocks[k](x, y)

        return x
