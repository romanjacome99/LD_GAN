import os
import random

import h5py
import numpy as np
from PIL import Image

import torch
import torchvision as tv
import torch.utils.data as data

import torchvision.datasets as dset
import torchvision.transforms as transforms

import scipy.io as sio


def load_dataset(dataset_name, dataset_path, batch_size, spatial_size, workers, labels=False):
    if 'arad' in dataset_name:
        dataset_loader = load_arad_dataset(dataset_path, batch_size, spatial_size, workers, labels=labels)
    elif 'celeba' in dataset_name:
        dataset_loader = load_celeba_dataset(dataset_path, batch_size, workers)
    else:
        raise "Dataset was not found, please enter the dataset name correctly"

    print(f'{dataset_name} dataset loaded')

    return dataset_loader


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# cave
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#


# load HSIs
def prepare_data(dataset_path):
    filenames = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.mat')]
    filenames.sort()

    HR_HSI = []
    for filename in filenames:
        print('Loading HSI: ' + filename)
        data = sio.loadmat(filename)
        HR_HSI.append(data['data_slice'] / 65535.0)

    HR_HSI = np.stack(HR_HSI, axis=-1)
    HR_HSI[HR_HSI < 0.] = 0.
    HR_HSI[HR_HSI > 1.] = 1.

    HR_HSI = 2 * HR_HSI - 1

    return HR_HSI


class CaveDataset(data.Dataset):
    def __init__(self, HSI, num_samples=30, isTrain=True):
        super(CaveDataset, self).__init__()
        self.HSI = HSI
        self.dataset_len = int(HSI.shape[-1])

        self.num_samples = num_samples
        self.isTrain = isTrain

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        label = self.HSI[:, :, :, index]

        if self.isTrain == True:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                label = np.rot90(label)

            # Random vertical Flip
            for j in range(vFlip):
                label = label[:, ::-1, :].copy()

            # Random horizontal Flip
            for j in range(hFlip):
                label = label[::-1, :, :].copy()

        label = torch.FloatTensor(label.copy()).permute(2, 0, 1)

        return label


def load_cave_dataset(dataset_path, batch_size, workers):
    HR_HSI = prepare_data(dataset_path)

    train_dataset = CaveDataset(HR_HSI, num_samples=30, isTrain=True)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                   pin_memory=True)

    return train_loader


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# arad
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

class AradDataset(data.Dataset):
    def __init__(self, dataset_path, spatial_size=64, isTrain=True, labels=False):
        super(AradDataset, self).__init__()
        self.dataset_path = f'{dataset_path}_full_{spatial_size}x{spatial_size}x31.h5'
        self.labels = labels

        with h5py.File(self.dataset_path, 'r') as hf:
            self.dataset_len = int(hf['spec'].shape[0])

        self.isTrain = isTrain

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        with h5py.File(self.dataset_path, 'r') as hf:
            spec = hf['spec'][index]
            rgb = hf['rgb'][index]

            spec = spec.astype(np.float32) / 65535.0
            rgb = rgb.astype(np.float32) / 65535.0

        if self.isTrain == True:
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

        if self.labels:
            return spec, rgb, torch.Tensor([1])
        else:
            return spec, rgb


class AradMatDataset(data.Dataset):
    def __init__(self, dataset_path, isTrain=True):
        super(AradMatDataset, self).__init__()
        self.dataset_path = dataset_path

        self.filenames = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.mat')]
        self.filenames.sort()

        self.dataset_len = len(self.filenames)
        self.isTrain = isTrain

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        # load .mat file
        data = sio.loadmat(self.filenames[index])
        label = data['cube']

        label = torch.from_numpy(label).permute(2, 0, 1)

        return label


def load_arad_dataset(dataset_path, batch_size, spatial_size, workers, labels=False):
    train_dataset = AradDataset(f'{dataset_path}/train', spatial_size=spatial_size, isTrain=True, labels=labels)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                   pin_memory=True)

    val_dataset = AradDataset(f'{dataset_path}/val', spatial_size=spatial_size, isTrain=False, labels=labels)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                 pin_memory=True)

    test_dataset = AradDataset(f'{dataset_path}/val', spatial_size=512, isTrain=False, labels=labels)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                  pin_memory=True)

    return train_loader, val_loader, test_loader


def load_celeba_dataset(dataset_path, batch_size, workers):
    train_dataset = dset.ImageFolder(root=dataset_path,
                                     transform=transforms.Compose([
                                         transforms.Resize(64),
                                         transforms.CenterCrop(64),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     ]))
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                   pin_memory=True)

    return train_loader, train_loader, train_loader
