from pathlib import Path

from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio

from scripts.functions import AverageMeter
from scripts.spec2rgb import ColourSystem


# create an autoencoder model for hyperspectral images using torch

def rgb_scheduler(param, epoch):
    if epoch != 0 and epoch % 15 == 0:
        return param * 1.2

    return param


class HSAE:
    def __init__(self, feature=64, num_ch=31, lr=4e-3, save_path=None, device='cuda'):
        self.num_ch = num_ch
        self.writer = SummaryWriter(save_path)
        self.save_path = save_path
        self.device = device

        self.encoder = self.build_encoder(feature=feature, num_ch=num_ch)
        self.decoder = self.build_decoder(feature=feature, num_ch=num_ch)
        self.autoencoder = nn.Sequential(self.encoder, self.decoder)

        self.SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.PSNR = PeakSignalNoiseRatio().to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.9, patience=15, verbose=True)

        self.cs = ColourSystem(cs='sRGB', start=400, end=720, num=num_ch, device=self.device)

        if device == 'cuda' and torch.cuda.is_available():
            self.encoder.to(device)
            self.decoder.to(device)
            self.autoencoder.to(device)
            self.criterion = self.criterion.to(device)
            print('Using GPU')

    def build_encoder(self, num_ch=31, feature=64):
        return nn.Sequential(
            nn.Conv2d(num_ch, feature, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feature, feature, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feature, feature // 2, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feature // 2, feature // 2, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feature // 2, feature // 4, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feature // 4, feature // 8, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feature // 8, feature // 16, 3, padding=1),
            nn.Tanh(),
        )

    def build_decoder(self, feature=64, num_ch=31):
        return nn.Sequential(
            nn.Conv2d(feature // 16, feature // 8, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feature // 8, feature // 4, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feature // 4, feature // 2, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feature // 2, feature // 2, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feature // 2, feature, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feature, feature, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feature, num_ch, 3, padding=1),
            nn.ReLU(True)
        )

    def train(self, data_loader, epochs, val_loader=None):

        for epoch in range(epochs):

            # ---------------- Train ---------------- #

            self.encoder.train()
            self.decoder.train()
            self.autoencoder.train()

            accuracy_metrics = {}
            losses = AverageMeter()
            reg_losses = AverageMeter()
            ssim_losses = AverageMeter()
            psnr_losses = AverageMeter()

            iter = 0

            with tqdm(data_loader, unit='batch') as loop:
                for data in loop:
                    loop.set_description(f'Train: Epoch [{epoch + 1}/{epochs}]')
                    x, y = data

                    if self.device == 'cuda' and torch.cuda.is_available():
                        x = x.to(self.device, non_blocking=True)
                        y = y.to(self.device, non_blocking=True)

                    x = Variable(x)
                    y = Variable(y)

                    metrics = self.train_step(x, y)

                    losses.update(metrics['loss'], x.size(0))
                    reg_losses.update(metrics['reg_loss'], x.size(0))
                    ssim_losses.update(metrics['ssim'], x.size(0))
                    psnr_losses.update(metrics['psnr'], x.size(0))

                    metrics['loss'] = losses.avg
                    metrics['reg_loss'] = reg_losses.avg
                    metrics['ssim'] = ssim_losses.avg
                    metrics['psnr'] = psnr_losses.avg

                    for k, v in metrics.items():
                        if k not in accuracy_metrics:
                            accuracy_metrics[k] = 0
                        accuracy_metrics[k] += v

                    iter += 1
                    loop.set_postfix(**metrics)

            self.scheduler.step(metrics['loss'])

            # ---------------- Log metrics ---------------- #

            if self.writer:
                for k, v in accuracy_metrics.items():
                    self.writer.add_scalar('train_{}'.format(k), v / iter, epoch)

            # ---------------- Validation ---------------- #

            if val_loader is not None:
                with torch.no_grad():
                    self.encoder.eval()
                    self.decoder.eval()
                    self.autoencoder.eval()

                    accuracy_metrics = {}
                    losses = AverageMeter()
                    ssim_losses = AverageMeter()
                    psnr_losses = AverageMeter()

                    iter = 0

                    with tqdm(val_loader, unit='batch') as loop:
                        for data in loop:
                            loop.set_description(f'Test: Epoch [{epoch + 1}/{epochs}]')
                            x, y = data

                            if self.device == 'cuda' and torch.cuda.is_available():
                                x = x.to(self.device, non_blocking=True)
                                y = y.to(self.device, non_blocking=True)

                            x = Variable(x)
                            y = Variable(y)

                            metrics = self.test_step(x, y)

                            losses.update(metrics['loss'], x.size(0))
                            ssim_losses.update(metrics['ssim'], x.size(0))
                            psnr_losses.update(metrics['psnr'], x.size(0))

                            metrics['loss'] = losses.avg
                            metrics['ssim'] = ssim_losses.avg
                            metrics['psnr'] = psnr_losses.avg

                            for k, v in metrics.items():
                                if k not in accuracy_metrics:
                                    accuracy_metrics[k] = 0
                                accuracy_metrics[k] += v

                            iter += 1
                            loop.set_postfix(**metrics)

                    # Save reconstructed images
                    self.save_images(val_loader, epoch)

            # ---------------- Log metrics ---------------- #

            if self.writer:
                for k, v in accuracy_metrics.items():
                    self.writer.add_scalar('test_{}'.format(k), v / iter, epoch)

            # ---------------- Save model ---------------- #

            if int(epoch) % 25 == 0:
                self.save_checkpoint(self.save_path, epoch=epoch)
                print(f'saved checkpoint at epoch {epoch}')

    def evaluate(self, data_loader, indices=[0, 4]):

        # ---------------- Validation ---------------- #

        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            self.autoencoder.eval()

            accuracy_metrics = {}
            losses = AverageMeter()
            ssim_losses = AverageMeter()
            psnr_losses = AverageMeter()

            iter = 0

            with tqdm(data_loader, unit='batch') as loop:
                for data in loop:
                    loop.set_description(f'Evaluating')
                    x, y = data

                    if self.device == 'cuda' and torch.cuda.is_available():
                        x = x.to(self.device, non_blocking=True)
                        y = y.to(self.device, non_blocking=True)

                    x = Variable(x)
                    y = Variable(y)

                    metrics = self.test_step(x, y)

                    losses.update(metrics['loss'], x.size(0))
                    ssim_losses.update(metrics['ssim'], x.size(0))
                    psnr_losses.update(metrics['psnr'], x.size(0))

                    metrics['loss'] = losses.avg
                    metrics['ssim'] = ssim_losses.avg
                    metrics['psnr'] = psnr_losses.avg

                    for k, v in metrics.items():
                        if k not in accuracy_metrics:
                            accuracy_metrics[k] = 0
                        accuracy_metrics[k] += v

                    iter += 1
                    loop.set_postfix(**metrics)

            # Save reconstructed images
            self.save_images(data_loader, 9999, indices=indices)

    def train_step(self, x, y):
        self.optimizer.zero_grad()

        # reconstruct images
        embedded = self.encoder(x)
        reconstructed = self.decoder(embedded)

        # calculate loss
        loss = self.criterion(reconstructed, x)

        reg_loss = 0
        for param in self.encoder.parameters():
            reg_loss += torch.norm(param)

        for param in self.decoder.parameters():
            reg_loss += torch.norm(param)

        loss += 1e-8 * reg_loss

        # backprop
        loss.backward()
        self.optimizer.step()

        ssim = self.SSIM(reconstructed, x)
        psnr = self.PSNR(reconstructed, x)

        return dict(loss=loss.item(), reg_loss=reg_loss.item() if reg_loss != 0 else reg_loss,
                    ssim=ssim.item(), psnr=psnr.item())

    def test_step(self, x, y):

        # reconstruct images
        embedded = self.encoder(x)
        reconstructed = self.decoder(embedded)

        # calculate loss
        loss = self.criterion(reconstructed, x)

        ssim = self.SSIM(reconstructed, x)
        psnr = self.PSNR(reconstructed, x)

        return dict(loss=loss.item(), ssim=ssim.item(), psnr=psnr.item())

    def save_images(self, val_loader, epoch, indices=[4, 57]):  # indices=[98, 950]):
        image_save_path = f'{self.save_path}/images'
        try:
            Path(image_save_path).mkdir(parents=True, exist_ok=False)
        except:
            pass

        self.encoder.eval()
        self.decoder.eval()
        self.autoencoder.eval()

        x, y = next(iter(val_loader))

        if self.device == 'cuda' and torch.cuda.is_available():
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

        with torch.no_grad():
            embedded = self.encoder(x)
            reconstructed = self.decoder(embedded)

        rgb_x = self.cs.spec_to_rgb_torch(x).permute(0, 2, 3, 1).cpu().numpy()
        rgb_reconstructed = self.cs.spec_to_rgb_torch(reconstructed).permute(0, 2, 3, 1).cpu().numpy()
        embedded = embedded.permute(0, 2, 3, 1).cpu().numpy()

        # save images

        plt.figure(figsize=(20, 20))

        idx1 = indices[0]

        plt.subplot(2, 2, 1)
        plt.imshow(rgb_x[idx1])
        plt.title('Original')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(np.clip(rgb_reconstructed[idx1], a_min=0., a_max=1.))
        plt.title(f'ssim: {self.SSIM(reconstructed[idx1:idx1 + 1], x[idx1:idx1 + 1]).item():.4f},'
                  f'psnr: {self.PSNR(reconstructed[idx1:idx1 + 1], x[idx1:idx1 + 1]).item():.4f}')
        plt.axis('off')

        idx2 = indices[1]

        plt.subplot(2, 2, 3)
        plt.imshow(rgb_x[idx2])
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(np.clip(rgb_reconstructed[idx2], a_min=0., a_max=1.))
        plt.title(f'ssim: {self.SSIM(reconstructed[idx2:idx2 + 1], x[idx2:idx2 + 1]).item():.4f},'
                  f'psnr: {self.PSNR(reconstructed[idx2:idx2 + 1], x[idx2:idx2 + 1]).item():.4f}')
        plt.axis('off')

        plt.savefig('{}/recons_{:03d}.png'.format(image_save_path, epoch))
        plt.close()

        plt.figure(figsize=(20, 20))

        plt.subplot(2, 2, 1)
        plt.imshow(rgb_x[idx1])
        plt.title('Original')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(np.clip(embedded[idx1], a_min=0., a_max=1.))
        plt.title(f'Encoded. min {np.min(embedded[idx1]):.4f}, max {np.max(embedded[idx1]):.4f}')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(rgb_x[idx2])
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(np.clip(embedded[idx2], a_min=0., a_max=1.))
        plt.title(f'Encoded. min {np.min(embedded[idx2]):.4f}, max {np.max(embedded[idx2]):.4f}')

        plt.savefig('{}/encoded_{:03d}.png'.format(image_save_path, epoch))
        plt.close()

    def save_checkpoint(self, path, epoch=None):
        if epoch is None:
            torch.save(self.encoder.state_dict(), '{}/encoder.pth'.format(path))
            torch.save(self.decoder.state_dict(), '{}/decoder.pth'.format(path))
        else:
            torch.save(self.encoder.state_dict(), '{}/encoder_{}.pth'.format(path, epoch))
            torch.save(self.decoder.state_dict(), '{}/decoder_{}.pth'.format(path, epoch))

    def load_checkpoint(self, path, epoch=None):
        if epoch is None:
            self.encoder.load_state_dict(torch.load('{}/encoder.pth'.format(path)))
            self.decoder.load_state_dict(torch.load('{}/decoder.pth'.format(path)))
        else:
            self.encoder.load_state_dict(torch.load('{}/encoder_{}.pth'.format(path, epoch)))
            self.decoder.load_state_dict(torch.load('{}/decoder_{}.pth'.format(path, epoch)))
