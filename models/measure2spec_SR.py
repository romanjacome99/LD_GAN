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
from models.DGSMP_SR import  DGSMP
# from models.DGSMP import DGSMP
from models.DSSP_SR import DSSP_SR
from models.e2e import UnetCompiled
# from models.UnetL import UnetCompiled
from models.optical_encoders import CASSI, Downsampling_Layer

from scripts.spec2rgb import SpectralSensitivity
from scripts.functions import AverageMeter


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# baseline
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

class M2S_SR:
    def __init__(self, model, input_shape, stride, patch_size, mask_path=None, trainable_mask=False,
                 mask_seed=None, sdf= 2, ldf=2,lr=1e-4, save_path=None, device=False):
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

        y_norm = True if model == 'dgsmp' else False

        self.optical_encoder = Downsampling_Layer(input_shape, sdf=sdf, ldf=ldf)

        if model == 'unetcompiled':
            L = input_shape[-1]
            self.computational_decoder = UnetCompiled(bands=L)

            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.computational_decoder.parameters(), lr=lr)
            self.scheduler = None
        elif model == 'dssp':
            L = input_shape[-1]
            self.computational_decoder = DSSP_SR(in_channels=L, out_channels=L, stages=8,sdf=sdf,ldf=ldf)

            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.SGD(self.computational_decoder.parameters(), lr=lr)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.85, patience=5)

        elif model == 'dgsmp':
            self.computational_decoder = DGSMP(Ch=input_shape[-1], stages=4, device=device)

            self.criterion = nn.L1Loss()
            self.optimizer = torch.optim.Adam(self.computational_decoder.parameters(), lr=lr, betas=(0.9, 0.999),
                                              eps=1e-8)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[], gamma=0.1)

        else:
            raise 'Model not found'

        self.optical_encoder.to(device)
        self.computational_decoder.to(device)
        self.criterion.to(device)

    def train(self, train_loader, init_epoch, epochs, val_loader=None):

        print("Beginning training")

        time_begin = time.time()
        for epoch in range(init_epoch, epochs):

            train_metrics = self.train_step(train_loader, epoch, epochs)

            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train_{key}', value, epoch)

            if val_loader is not None:
                test_metrics = self.test_step(val_loader, epoch, epochs)

                for key, value in test_metrics.items():
                    self.writer.add_scalar(f'test_{key}', value, epoch)

            # lr and reg scheduler
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], epoch)

            if self.scheduler is not None:
                self.scheduler.step(epoch)

            # if epoch == 0 or epoch % 20 == 0 or epoch == epochs - 1:
            # self.save_checkpoint(f'{self.save_path}/checkpoints', epoch)
            self.save_images(val_loader, epoch)
            self.save_checkpoint(f'{self.save_path}/checkpoints', epoch)



        print("Ending training")

        total_mins = (time.time() - time_begin) / 60
        print(f'Script finished in {total_mins:.2f} minutes')

    def train_step(self, dataloader, epoch, max_epochs):
        self.optical_encoder.train()
        self.computational_decoder.train()

        return self.forward(dataloader, epoch, max_epochs, kind='train', colour='red')

    def test_step(self, dataloader, epoch, max_epochs):
        with torch.no_grad():
            self.optical_encoder.eval()
            self.computational_decoder.eval()

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
            _, x = data

            x = Variable(x.to(self.device, non_blocking=self.num_blocking))
            y = self.optical_encoder(x, only_measurement=True)

            if self.model == 'unetcompiled':
                x0 = self.optical_encoder(x)
                x_hat = self.computational_decoder(x0)

            elif self.model == 'dgsmp':
                x_hat = self.computational_decoder(y)

            else:
                x0 = self.optical_encoder(x)
                x_hat = self.computational_decoder(x0, y)

            loss = self.criterion(x_hat, x)

            losses.update(loss.item(), x.size(0))

            mse_x = self.MSE(x_hat, x)
            mae_x = self.MAE(x_hat, x)
            ssim_x = self.SSIM(x_hat, x)
            psnr_x = self.PSNR(x_hat, x)

            mse_x_losses.update(mse_x.item(), x.size(0))
            mae_x_losses.update(mae_x.item(), x.size(0))
            ssim_x_losses.update(ssim_x.item(), x.size(0))
            psnr_x_losses.update(psnr_x.item(), x.size(0))

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

        del loss, x_hat

        return dict_metrics

    def save_images(self, val_loader, epoch, save=True, show=False, patches=False):
        self.optical_encoder.eval()
        self.computational_decoder.eval()

        # for i, data in enumerate(val_loader):
        #     _, spec = data
        #
        #     if i == 5:
        #         break

        _, spec = next(iter(val_loader))
        spec = spec.to(self.device, non_blocking=self.num_blocking)

        with torch.no_grad():
            y = self.optical_encoder(spec, only_measurement=True)

            if self.model == 'unetcompiled':
                x0 = self.optical_encoder(spec)
                reconstructed = self.computational_decoder(x0)

            elif self.model == 'dgsmp':
                reconstructed = self.computational_decoder(y)

            else:
                x0 = self.optical_encoder(spec)
                reconstructed = self.computational_decoder(x0, y)

        y = y.permute(0, 2, 3, 1).cpu().numpy()

        if spec.shape[1] == 3:
            rgb_reconstructed = reconstructed.permute(0, 2, 3, 1).cpu().numpy()
            rgb_spec = spec.permute(0, 2, 3, 1).cpu().numpy()

        else:
            rgb_reconstructed = self.spec_sensitivity.get_rgb_01(reconstructed).permute(0, 2, 3, 1).cpu().numpy()
            rgb_spec = self.spec_sensitivity.get_rgb_01(spec).permute(0, 2, 3, 1).cpu().numpy()

        # save images

        indices = np.linspace(0, len(rgb_spec) - 1, 4).astype(int)

        # if patches:
        #     indices = [10, 18, 45, 49]  # for patches

        plt.figure(figsize=(30, 40))

        count = 1
        for idx in indices:
            plt.subplot(4, 3, count)
            plt.imshow(y[idx,:,:,0], cmap='gray')
            plt.title('Measurement')
            plt.axis('off')

            plt.subplot(4, 3, count + 1)
            plt.imshow(np.clip(rgb_reconstructed[idx], a_min=0., a_max=1.) ** 0.7)
            plt.title(f'ssim: {self.SSIM(reconstructed[idx:idx + 1], spec[idx:idx + 1]).item():.4f},'
                      f'psnr: {self.PSNR(reconstructed[idx:idx + 1], spec[idx:idx + 1]).item():.4f}')
            plt.axis('off')

            plt.subplot(4, 3, count + 2)
            plt.imshow(rgb_spec[idx] ** 0.7)
            plt.title('GT')
            plt.axis('off')

            count += 3

        if save:
            plt.savefig('{}/recons_{:03d}.png'.format(self.save_image_path, epoch))

        if show:
            plt.show()

        plt.close()

        return reconstructed[0, :]

    def save_full_images(self, val_loader, epoch, save=True, show=False, full=False):
        self.optical_encoder.eval()
        self.computational_decoder.eval()

        _, spec = next(iter(val_loader))
        spec = spec.to(self.device, non_blocking=self.num_blocking)

        with torch.no_grad():
            y = self.optical_encoder(spec, only_measurement=True)

            if self.model == 'dgsmp':
                reconstructed = self.computational_decoder(y)

            else:
                x0 = self.optical_encoder(spec)
                reconstructed = self.computational_decoder(x0, y)

        y = y.permute(0, 2, 3, 1).cpu().numpy()

        if spec.shape[1] == 3:
            rgb_reconstructed = reconstructed.permute(0, 2, 3, 1).cpu().numpy()
            rgb_spec = spec.permute(0, 2, 3, 1).cpu().numpy()

        else:
            rgb_reconstructed = self.spec_sensitivity.get_rgb_01(reconstructed).permute(0, 2, 3, 1).cpu().numpy()
            rgb_spec = self.spec_sensitivity.get_rgb_01(spec).permute(0, 2, 3, 1).cpu().numpy()

        # save images

        indices = np.linspace(0, len(rgb_spec) - 1, 4).astype(int)

        if full:
            indices = [10, 18, 45, 49]  # for patches

        plt.figure(figsize=(30, 40))

        count = 1
        for idx in indices:
            plt.subplot(4, 3, count)
            plt.imshow(y[idx], cmap='gray')
            plt.title('Measurement')
            plt.axis('off')

            plt.subplot(4, 3, count + 1)
            plt.imshow(np.clip(rgb_reconstructed[idx], a_min=0., a_max=1.) / np.max(rgb_reconstructed[idx]))
            plt.title(f'ssim: {self.SSIM(reconstructed[idx:idx + 1], spec[idx:idx + 1]).item():.4f},'
                      f'psnr: {self.PSNR(reconstructed[idx:idx + 1], spec[idx:idx + 1]).item():.4f}')
            plt.axis('off')

            plt.subplot(4, 3, count + 2)
            plt.imshow(rgb_spec[idx] / np.max(rgb_spec[idx]))
            plt.title('GT')
            plt.axis('off')

            count += 3

        if save:
            plt.savefig('{}/big_recons_{:03d}.png'.format(self.save_image_path, epoch))

        if show:
            plt.show()

        plt.close()

        return reconstructed[10, :, 32, 32], spec[10, :, 32, 32]

    def save_checkpoint(self, path, epoch=None):
        if epoch is None:
            torch.save(self.optical_encoder.state_dict(), '{}/optical_encoder.pth'.format(path))
            torch.save(self.computational_decoder.state_dict(), '{}/computational_decoder.pth'.format(path))
        else:
            torch.save(self.optical_encoder.state_dict(), '{}/optical_encoder_{}.pth'.format(path, epoch))
            torch.save(self.computational_decoder.state_dict(), '{}/computational_decoder_{}.pth'.format(path, epoch))

    def load_checkpoint(self, path, epoch=None, is_encoder=True, is_decoder=True):
        if epoch is None:
            if is_encoder:
                self.optical_encoder.load_state_dict(torch.load('{}/optical_encoder.pth'.format(path)))

            if is_decoder:
                self.computational_decoder.load_state_dict(torch.load('{}/computational_decoder.pth'.format(path)))

        else:
            if is_encoder:
                self.optical_encoder.load_state_dict(torch.load('{}/optical_encoder_{}.pth'.format(path, epoch)))

            if is_decoder:
                self.computational_decoder.load_state_dict(
                    torch.load('{}/computational_decoder_{}.pth'.format(path, epoch)))
