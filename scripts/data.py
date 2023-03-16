import glob
import random

import h5py
import numpy as np
import scipy.io as sio

import torch
import torch.utils.data as data


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# dataset
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#


class Dataset:
    def __init__(self, dataset_path, batch_size, patch_size, workers):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.workers = workers

    def get_arad_dataset(self, gen_dataset_path=None, real=1., syn=1.):
        train_dataset = AradDataset(f'{self.dataset_path}/train', gen_dataset_path=gen_dataset_path,
                                    real=real, syn=syn, patch_size=self.patch_size, is_train=True)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.workers, pin_memory=True)

        test_dataset = AradDataset(f'{self.dataset_path}/val', patch_size=self.patch_size, is_train=False)
        test_loader = data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.workers, pin_memory=True)

        big_test_dataset = AradDataset(f'{self.dataset_path}/val', patch_size=256, is_train=False)
        big_test_loader = data.DataLoader(big_test_dataset, batch_size=self.batch_size, shuffle=False,
                                          num_workers=self.workers, pin_memory=True)

        return train_loader, (test_loader, big_test_loader)


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# arad
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#


class AradDataset(data.Dataset):
    # real: percentage of real data
    # syn: percentage of synthetic data
    def __init__(self, dataset_path, gen_dataset_path=None, real=1., syn=1., patch_size=128,
                 is_train=True, ae_var=None, gan_var=None):
        super(AradDataset, self).__init__()

        if ae_var is not None and gan_var is not None:
            self.dataset_path = f'{dataset_path}_full_{patch_size}x{patch_size}x31_ae_var_{ae_var}_gan_var_{gan_var}.h5'
            raise "Por aquí no es"
        else:
            self.dataset_path = f'{dataset_path}_full_{patch_size}x{patch_size}x31.h5'

        self.gen_dataset_path = gen_dataset_path

        with h5py.File(self.dataset_path, 'r') as hf:
            self.base_dataset_len = int(hf['spec'].shape[0])

        self.real_samples = int(real * self.base_dataset_len)
        self.dataset_len = self.real_samples

        self.syn_samples = 0
        if gen_dataset_path is not None:
            print(f'Load generated dataset: {gen_dataset_path}')
            with h5py.File(self.gen_dataset_path, 'r') as hf:
                self.gen_dataset_len = int(hf['spec'].shape[0])

            self.syn_samples = int(syn * self.gen_dataset_len)
            self.dataset_len += self.syn_samples

        self.is_train = is_train

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        if index < self.real_samples:  # self.base_dataset_len:
            with h5py.File(self.dataset_path, 'r') as hf:
                spec = hf['spec'][index]
                rgb = hf['rgb'][index]

                spec = spec.astype(np.float32) / 65535.0
                rgb = rgb.astype(np.float32) / 65535.0

        else:
            with h5py.File(self.gen_dataset_path, 'r') as hf:
                spec = hf['spec'][index - self.real_samples]
                rgb = hf['rgb'][index - self.real_samples]

                spec = spec.astype(np.float32) / 65535.0
                rgb = rgb.astype(np.float32) / 65535.0

        if self.is_train == True:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                spec = np.rot90(spec)
                rgb = np.rot90(rgb)

            # Random vertical Flip
            for j in range(vFlip):
                spec = spec[:, ::-1, :].copy()
                rgb = rgb[:, ::-1, :].copy()

            # Random horizontal Flip
            for j in range(hFlip):
                spec = spec[::-1, :, :].copy()
                rgb = rgb[::-1, :, :].copy()

        spec = torch.from_numpy(spec.copy()).permute(2, 0, 1)
        rgb = torch.from_numpy(rgb.copy()).permute(2, 0, 1)

        return rgb, spec
