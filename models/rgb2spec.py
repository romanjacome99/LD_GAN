from pathlib import Path
import time

import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from torchmetrics import MeanSquaredError, MeanAbsoluteError, StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from models.DGSMP import DGSMP
from models.DSSP import DSSP
from models.Unet import UnetCompiled, CompactUnet, UnetModel
from models.optical_encoders import CASSI

from scripts.spec2rgb import SpectralSensitivity
from scripts.functions import AverageMeter


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# baseline
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

class RGB2SPEC:
    def __init__(self, model, input_shape, lr=5e-4, save_path=None, device=False):
        self.model = model

        self.writer = SummaryWriter(save_path)
        self.save_path = save_path
        self.device = device
        self.num_blocking = torch.cuda.is_available()

        self.MSE = MeanSquaredError().to(device)
        self.MAE = MeanAbsoluteError().to(device)
        self.SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.PSNR = PeakSignalNoiseRatio().to(device)

        self.spec_sensitivity = SpectralSensitivity('Canon60D', bands=input_shape[-1], device=device)

        self.save_image_path = f'{save_path}/images'
        Path(self.save_image_path).mkdir(parents=True, exist_ok=True)

        # Model

        if model == 'unet':
            L = input_shape[-1]
            self.rgb2spec_model = CompactUnet(bands=L)
            # self.rgb2spec_model = UnetModel(3, L, p_drop=0.2)

            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.rgb2spec_model.parameters(), lr=lr)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        else:
            raise 'Model not found'

        self.rgb2spec_model.to(device)
        self.criterion.to(device)

    def train(self, train_loader, init_epoch, epochs, val_loader=None):

        print("Beginning training")

        time_begin = time.time()
        for epoch in range(init_epoch, epochs):
            train_metrics = self.train_step(train_loader, epoch, epochs)

            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train_{key}', value, epoch)

            if val_loader is not None:
                test_metrics = self.test_step(val_loader[0], epoch, epochs)

                for key, value in test_metrics.items():
                    self.writer.add_scalar(f'test_{key}', value, epoch)

            # lr and reg scheduler
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            # if epoch == 0 or epoch % 20 == 0 or epoch == epochs - 1:
            # self.save_checkpoint(f'{self.save_path}/checkpoints', epoch)
            self.save_images('recons', val_loader[0], epoch)
            self.save_images('big_recons', val_loader[1], epoch)
            self.save_checkpoint(f'{self.save_path}/checkpoints', epoch)


        print("Ending training")

        total_mins = (time.time() - time_begin) / 60
        print(f'Script finished in {total_mins:.2f} minutes')

    def train_step(self, dataloader, epoch, max_epochs):
        self.rgb2spec_model.train()

        return self.forward(dataloader, epoch, max_epochs, kind='train', colour='red')

    def test_step(self, dataloader, epoch, max_epochs):
        with torch.no_grad():
            self.rgb2spec_model.eval()

            return self.forward(dataloader, epoch, max_epochs, kind='test', colour='green')

    def forward(self, dataloader, epoch, max_epochs, kind, colour):
        losses = AverageMeter()

        mse_x_losses = AverageMeter()
        mae_x_losses = AverageMeter()
        ssim_x_losses = AverageMeter()
        psnr_x_losses = AverageMeter()

        dict_metrics = dict()
        data_loop = tqdm(enumerate(dataloader), total=len(dataloader), colour=colour)
        for _, data in data_loop:
            _, spec = data

            # rgb = Variable(rgb.to(self.device, non_blocking=self.num_blocking))
            spec = Variable(spec.to(self.device, non_blocking=self.num_blocking))
            rgb = self.spec_sensitivity.get_rgb_01(spec)

            spec_hat = self.rgb2spec_model(rgb)

            loss = self.criterion(spec_hat, spec)

            losses.update(loss.item(), spec.size(0))

            mse_x = self.MSE(spec_hat, spec)
            mae_x = self.MAE(spec_hat, spec)
            ssim_x = self.SSIM(spec_hat, spec)
            psnr_x = self.PSNR(spec_hat, spec)

            mse_x_losses.update(mse_x.item(), spec.size(0))
            mae_x_losses.update(mae_x.item(), spec.size(0))
            ssim_x_losses.update(ssim_x.item(), spec.size(0))
            psnr_x_losses.update(psnr_x.item(), spec.size(0))

            self.MSE.reset()
            self.MAE.reset()
            self.SSIM.reset()
            self.PSNR.reset()

            dict_metrics = dict(loss=losses.avg,
                                mse_x=mse_x_losses.avg, mae_x=mae_x_losses.avg,
                                ssim_x=ssim_x_losses.avg, psnr_x=psnr_x_losses.avg)

            if kind == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            lr = format(self.optimizer.param_groups[0]['lr'], '.1e')
            data_loop.set_description(f'{kind.capitalize()}: Epoch [{epoch + 1} / {max_epochs}] lr: {lr}')
            data_loop.set_postfix(**dict_metrics)

        del loss, spec_hat

        return dict_metrics

    def save_images(self, name, val_loader, epoch, save=True, show=False):
        self.rgb2spec_model.eval()

        rgb, spec = next(iter(val_loader))
        rgb = rgb.to(self.device, non_blocking=self.num_blocking)
        spec = spec.to(self.device, non_blocking=self.num_blocking)

        with torch.no_grad():
            spec_hat = self.rgb2spec_model(rgb)

        rgb_spec_hat = self.spec_sensitivity.get_rgb_01(spec_hat).permute(0, 2, 3, 1).cpu().numpy()
        rgb = rgb.permute(0, 2, 3, 1).cpu().numpy()

        # save images

        indices = np.linspace(0, len(rgb) - 1, 8).astype(int)

        # if patches:
        #     indices = [10, 18, 45, 49]  # for patches

        plt.figure(figsize=(40, 40))

        count = 1
        for idx in indices:
            plt.subplot(4, 4, count)
            plt.imshow(rgb[idx], cmap='gray')
            plt.title('rgb GT')
            plt.axis('off')

            plt.subplot(4, 4, count + 1)
            plt.imshow(rgb_spec_hat[idx])
            plt.title(f'ssim: {self.SSIM(spec_hat[idx:idx + 1], spec[idx:idx + 1]).item():.4f},'
                      f'psnr: {self.PSNR(spec_hat[idx:idx + 1], spec[idx:idx + 1]).item():.4f}')
            plt.axis('off')

            count += 2

        if save:
            plt.savefig('{}/{}_{:03d}.png'.format(self.save_image_path, name, epoch))

        if show:
            plt.show()

        plt.close()

        return spec_hat[0, :]

    def save_checkpoint(self, path, epoch=None):
        if epoch is None:
            torch.save(self.rgb2spec_model.state_dict(), '{}/rgb2spec_model.pth'.format(path))
        else:
            torch.save(self.rgb2spec_model.state_dict(), '{}/rgb2spec_model_{}.pth'.format(path, epoch))

    def load_checkpoint(self, path, epoch=None):
        if epoch is None:
            self.rgb2spec_model.load_state_dict(torch.load('{}/rgb2spec_model.pth'.format(path)))

        else:
            self.rgb2spec_model.load_state_dict(torch.load('{}/rgb2spec_model_{}.pth'.format(path, epoch)))
