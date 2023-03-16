import numpy as np
import scipy.io as sio
import torch

from scipy import interpolate


class SpectralSensitivity:
    def __init__(self, spec_sensitivity_name, bands, device='cuda'):
        spec_sensitivity = sio.loadmat(f'datasets/camSpecSensitivity/cmf_{spec_sensitivity_name}.mat')

        # interpolate

        x = np.linspace(400, 720, 33)

        r = spec_sensitivity['r'].T
        g = spec_sensitivity['g'].T
        b = spec_sensitivity['b'].T

        f_r = interpolate.interp1d(x, r, axis=0)
        f_g = interpolate.interp1d(x, g, axis=0)
        f_b = interpolate.interp1d(x, b, axis=0)

        x_new = np.linspace(400, 720, bands)

        r_new = torch.from_numpy(f_r(x_new)).float().to(device)
        g_new = torch.from_numpy(f_g(x_new)).float().to(device)
        b_new = torch.from_numpy(f_b(x_new)).float().to(device)

        # build rgb matrix

        self.rgb_matrix = torch.cat([r_new, g_new, b_new], dim=-1)
        norm_constant = torch.sum(self.rgb_matrix, dim=0, keepdim=True).max()

        self.rgb_matrix /= norm_constant

    def get_rgb(self, spec):  # for training: spec [-1, 1]
        spec = (spec - torch.min(spec)) / (torch.max(spec) - torch.min(spec))
        rgb_spec = torch.einsum('blmn,lc->bcmn', spec, self.rgb_matrix)
        return 2 * (rgb_spec - torch.min(rgb_spec)) / (torch.max(rgb_spec) - torch.min(rgb_spec)) - 1

    def get_rgb_01(self, spec):  # spec [0, 1]
        return torch.einsum('blmn,lc->bcmn', spec, self.rgb_matrix)



def g(x, alpha, mu, sigma1, sigma2):
    sigma = (x < mu) * sigma1 + (x >= mu) * sigma2
    return alpha * np.exp((x - mu) ** 2 / (-2 * (sigma ** 2)))


def component_x(x): return g(x, 1.056, 5998, 379, 310) + \
                           g(x, 0.362, 4420, 160, 267) + g(x, -0.065, 5011, 204, 262)


def component_y(x): return g(x, 0.821, 5688, 469, 405) + \
                           g(x, 0.286, 5309, 163, 311)


def component_z(x): return g(x, 1.217, 4370, 118, 360) + \
                           g(x, 0.681, 4590, 260, 138)


def xyz_from_xy(x, y):
    """Return the vector (x, y, 1-x-y)."""
    return np.array((x, y, 1 - x - y))


ILUMINANT = {
    'D65': xyz_from_xy(0.3127, 0.3291),
    'E': xyz_from_xy(1 / 3, 1 / 3),
}

COLOR_SPACE = {
    'sRGB': (xyz_from_xy(0.64, 0.33),
             xyz_from_xy(0.30, 0.60),
             xyz_from_xy(0.15, 0.06),
             ILUMINANT['D65']),

    'AdobeRGB': (xyz_from_xy(0.64, 0.33),
                 xyz_from_xy(0.21, 0.71),
                 xyz_from_xy(0.15, 0.06),
                 ILUMINANT['D65']),

    'AppleRGB': (xyz_from_xy(0.625, 0.34),
                 xyz_from_xy(0.28, 0.595),
                 xyz_from_xy(0.155, 0.07),
                 ILUMINANT['D65']),

    'UHDTV': (xyz_from_xy(0.708, 0.292),
              xyz_from_xy(0.170, 0.797),
              xyz_from_xy(0.131, 0.046),
              ILUMINANT['D65']),

    'CIERGB': (xyz_from_xy(0.7347, 0.2653),
               xyz_from_xy(0.2738, 0.7174),
               xyz_from_xy(0.1666, 0.0089),
               ILUMINANT['E']),
}


class ColourSystem:

    def __init__(self, start=380, end=750, num=100, cs='sRGB', device='cpu'):
        self.device = device

        # Chromaticities
        bands = np.linspace(start=start, stop=end, num=num) * 10

        self.cmf = np.array([component_x(bands),
                             component_y(bands),
                             component_z(bands)])

        self.red, self.green, self.blue, self.white = COLOR_SPACE[cs]

        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T
        self.MI = np.linalg.inv(self.M)

        # White scaling array
        self.wscale = self.MI.dot(self.white)

        # xyz -> rgb transformation matrix
        self.A = self.MI / self.wscale[:, np.newaxis]

    def get_transform_matrix(self):
        XYZ = self.cmf
        RGB = XYZ.T @ self.A.T
        RGB = RGB / np.sum(RGB, axis=0, keepdims=True)
        return RGB

    def spec_to_rgb(self, spec):
        """Convert a spectrum to an rgb value."""
        M = self.get_transform_matrix()
        rgb = spec @ M
        return rgb

    def spec_to_rgb_torch(self, spec):  # between [-1, 1]
        """Convert a spectrum to an rgb value."""
        M = torch.tensor(self.get_transform_matrix(), dtype=torch.float32).to(self.device)
        # spec = (spec - torch.min(spec)) / (torch.max(spec) - torch.min(spec))
        rgb = torch.einsum('blmn,lk->bkmn', spec, M)

        return rgb

        # return 2 * (rgb - torch.min(rgb)) / (torch.max(rgb) - torch.min(rgb)) - 1