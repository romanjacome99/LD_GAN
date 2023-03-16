import random
import scipy.io as sio

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

from einops import repeat


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# optical encoders
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#


class CASSI(nn.Module):
    def __init__(self, input_shape, stride, patch_size=None, mask_path=None, trainable_mask=False,
                 mask_seed=None, is_patch=True, y_norm=False):
        super(CASSI, self).__init__()
        self.stride = stride
        self.patch_size = patch_size
        self.mask_path = mask_path
        self.trainable_mask = trainable_mask
        self.mask_seed = mask_seed
        self.is_patch = is_patch
        self.y_norm = y_norm

        self.build(input_shape)

    def build(self, input_shape):
        M, N, self.L = input_shape
        M = M if M > self.patch_size else self.patch_size
        N = M

        if self.mask_path is None:

            if self.mask_seed is not None:
                torch.manual_seed(self.mask_seed)
                print('mask seed established: {}'.format(self.mask_seed))

            phi = (1 + torch.sign(torch.randn((M, N)))) / 2
            phi = repeat(phi, 'm n -> l m n', l=self.L)

        else:
            phi = sio.loadmat(f'datasets/masks/{self.mask_path}.mat')['CA']
            if phi.ndim == 2:
                phi = repeat(phi, 'm n -> l m n', l=self.L)

            phi = torch.from_numpy(phi.astype('float32'))

        self.phi = torch.nn.Parameter(phi, requires_grad=self.trainable_mask)

    def set_is_patch(self, is_patch):
        self.is_patch = is_patch

    def get_phi_patch(self, phi):
        phi_shape = phi.shape
        pxm = random.randint(0, phi_shape[1] - self.patch_size)
        pym = random.randint(0, phi_shape[2] - self.patch_size)

        return phi[:, pxm:pxm + self.patch_size:1, pym:pym + self.patch_size:1]

    def get_measurement(self, x, phi):
        b, L, M, N = x.shape
        y1 = torch.einsum('blmn,lmn->blmn', x, phi)

        # shift and sum
        y2 = torch.zeros((b, 1, M, N + self.stride * (L - 1)), device=x.device)
        for l in range(L):
            y2 += nn.functional.pad(y1[:, l, None], (self.stride * l, self.stride * (L - l - 1)))

        return y2 / L * self.stride if self.y_norm else y2

    def get_transpose(self, y, phi, mask_mul=True):
        x = torch.cat([y[..., self.stride * l:self.stride * l + self.patch_size] for l in range(self.L)], dim=1)
        x = torch.einsum('blmn,lmn->blmn', x, phi) if mask_mul else x
        return x

    def forward(self, x, only_measurement=False, only_transpose=False, mask_mul=True):
        phi = self.get_phi_patch(self.phi)

        if only_transpose:
            return self.get_transpose(x, phi, mask_mul=mask_mul)

        if only_measurement:
            return self.get_measurement(x, phi)

        return self.get_transpose(self.get_measurement(x, phi), phi, mask_mul=mask_mul)


class Downsampling_Layer(nn.Module):
    def __init__(self, input_shape, sdf=2, ldf=2):
        super(Downsampling_Layer, self).__init__()

        self.sdf = sdf
        self.ldf = ldf
        self.L = input_shape[-1]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def forward(self, x, only_measurement=False, only_transpose=False):

        if only_transpose:
            return self.get_transpose(x)

        if only_measurement:
            return self.get_measurement(x)

        return self.get_transpose(self.get_measurement(x))

    def get_measurement(self, x):
        kernel = self.SpectralDegradationFilter(3, self.L, self.ldf).to(self.device)
        low_spectral = F.conv2d(x, kernel, padding=1)
        y =  nn.AvgPool2d(kernel_size=(int(self.sdf),int(self.sdf)))(low_spectral)
        return y

    def get_transpose(self, y):
        kernel = self.SpectralUpsamplingFilter(3, self.ldf, self.L).to(self.device)
        hs = F.interpolate(y,scale_factor=(self.sdf,self.sdf))
        x =   F.conv2d(hs,kernel,padding=1)
        return x
    def SpectralDegradationFilter(self, window_size, L, q):
        kernel = torch.zeros((L // q, L, window_size, window_size))
        for i in range(0, L // q):
            kernel[i, i * q:(i + 1) * (q), window_size // 2, window_size // 2] = 1 / q
        return kernel

    def ProjectionFilter(self, window_size, L):
        kernel = torch.zeros((1, L, window_size, window_size))
        kernel[0, 1:L, window_size // 2, window_size // 2] = 1
        return kernel

    def SpectralUpsamplingFilter(self, window_size, q, L):
        kernel = torch.zeros((L, L // q, window_size, window_size))
        for i in range(0, L // q):
            for j in range(0, q):
                kernel[i * q + j, i, window_size // 2, window_size // 2] = 1
        return kernel
