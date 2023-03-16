from pathlib import Path

from colorama import Fore
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr

from models.schedulers import SigmoidScheduler, ExponentialScheduler
from scripts.functions import AverageMeter
from scripts.spec2rgb import ColourSystem


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# baseline
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#


class E2E:
    def __init__(self, batch_size, input_size, compression_rate, seed, noise, parameter, p_step, p_aum,
                 trainable, lr, save_path, samples, epochs, device):
        self.optical_encoder = self.build_optical_encoder(batch_size, input_size, compression_rate, seed,
                                                          noise, trainable, device)
        self.computational_decoder = self.build_computational_decoder(bands=input_size[-1])
        self.e2e_model = self.build_e2e_model()

        self.parameter = parameter
        self.p_step = p_step
        self.p_aum = p_aum
        self.lr = lr

        # self.param_scheduler = SigmoidScheduler(1e-5, epochs, alpha=0.1, param_min=parameter, inverse=True)
        self.param_scheduler = ExponentialScheduler(1e-3, epochs, param_min=parameter, inverse=True)

        self.writer = SummaryWriter(save_path)
        self.save_path = save_path
        self.samples = samples
        self.epochs = epochs
        self.device = device

        self.cs = ColourSystem(cs='sRGB', start=400, end=720, num=input_size[-1], device=device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.e2e_model.parameters(), lr=self.lr)

    def build_optical_encoder(self, batch_size, input_size, compression_rate, seed, noise, trainable, device):
        return SPC(batch_size, input_size, compression_rate, seed, noise, trainable, device)

    def build_computational_decoder(self, bands):
        return UnetCompiled(bands=bands)

    def build_e2e_model(self):
        return nn.Sequential(self.optical_encoder, self.computational_decoder)

    def save_images(self, epoch):
        if epoch == 0:
            Path(self.save_path + '/images').mkdir(parents=True, exist_ok=True)

        self.e2e_model.eval()
        predictions = self.e2e_model(self.samples).detach().cpu().numpy()
        samples = self.samples[:16].detach().cpu().numpy()
        predictions = predictions[:16]

        plt.figure(figsize=(8, 8))

        for i in range(0, predictions.shape[0], 2):
            sample = (np.transpose(samples[i], (1, 2, 0)) - np.min(samples[i])) / (
                    np.max(samples[i]) - np.min(samples[i]))
            prediction = (np.transpose(predictions[i], (1, 2, 0)) - np.min(predictions[i])) / (
                    np.max(predictions[i]) - np.min(predictions[i]))

            if sample.shape[0] > 3:
                sample = self.cs.spec_to_rgb(sample)

            if prediction.shape[0] > 3:
                prediction = self.cs.spec_to_rgb(prediction)

            plt.subplot(4, 4, i + 1)
            plt.title('Samples')
            plt.imshow(sample)
            plt.axis('off')
            plt.subplot(4, 4, i + 2)
            plt.title('Predictions')
            plt.imshow(prediction)
            plt.axis('off')

        plt.savefig('{}/images/image_{:04d}.png'.format(self.save_path, epoch))
        plt.close()

        # CA

        ca = self.e2e_model[0].phi[0].detach().cpu().numpy()
        ca = (ca - np.min(ca)) / (np.max(ca) - np.min(ca))

        plt.figure()
        plt.imshow(ca, cmap='gray')
        plt.axis('off')

        plt.savefig('{}/images/CA_{:04d}.png'.format(self.save_path, epoch))
        plt.close()

    def reg(self, parameter, phi):
        return parameter * torch.sum(torch.multiply(torch.square(1 + phi), torch.square(1 - phi)))

    # ---------------- Define training loop ---------------- #

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            print(f'Epoch {epoch}')
            self.e2e_model.train()
            train_metrics = self.forward_step(train_loader, epoch, training=True)
            with torch.no_grad():
                self.e2e_model.eval()
                val_metrics = self.forward_step(val_loader, epoch, training=False)

            # self.update_parameter(epoch)

            # ---------------- Log metrics ---------------- #

            self.save_images(epoch)

            if self.writer is not None:
                for key, value in train_metrics.items():
                    self.writer.add_scalar(f'train_{key}', value, epoch)

                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'val_{key}', value, epoch)

            if int(epoch) % 25 == 0:
                self.save_checkpoint(self.save_path, epoch=epoch)
                print(f'saved checkpoint at epoch {epoch}')

    def forward_step(self, data_loader, epoch, training=False):
        losses = AverageMeter()
        losses_reg = AverageMeter()
        losses_mae = AverageMeter()
        losses_ssim = AverageMeter()
        losses_psnr = AverageMeter()

        name_output = 'Train'
        color_output = 'green'
        fore_color = Fore.GREEN
        if not training:
            name_output = ' Test'
            color_output = 'red'
            fore_color = Fore.RED

        loop = tqdm(enumerate(data_loader), total=len(data_loader), colour=color_output)
        for iter, (images, target) in loop:
            images = images.to(self.device)
            target = target.to(self.device)

            # compute output
            recovered = self.e2e_model(images)
            loss = self.criterion(recovered, images)
            loss_reg = self.reg(self.parameter, self.e2e_model[0].phi)
            loss += loss_reg

            # measure accuracy and record loss

            loss_mae = torch.mean(torch.abs(recovered - images))
            loss_ssim = ssim(recovered, images)
            loss_psnr = psnr(recovered, images)

            losses.update(loss.data.item(), images.size(0))
            losses_reg.update(loss_reg.data.item(), images.size(0))
            losses_mae.update(loss_mae.data.item(), images.size(0))
            losses_ssim.update(loss_ssim.item(), images.size(0))
            losses_psnr.update(loss_psnr.item(), images.size(0))

            if training:
                # back-propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # update progress bar
            lr = format(self.optimizer.param_groups[0]['lr'], ".1e")
            output_description = f'{fore_color}{name_output}: Epoch [{epoch + 1}/{self.epochs}] lr: {lr}'

            loop.set_description(output_description)
            loop.set_postfix(loss=losses.avg, loss_reg=losses_reg.avg, mae=losses_mae.avg,
                             ssim=losses_ssim.avg, psnr=losses_psnr.avg, param=self.parameter)

        return {'loss': losses.avg, 'loss_reg': losses_reg.avg, 'mae': losses_mae.avg,
                'ssim': losses_ssim.avg, 'psnr': losses_psnr.avg, 'param': self.parameter}

    def update_parameter(self, epoch):
        self.parameter = self.param_scheduler.step(epoch)
        # if epoch % self.p_step == 0 and epoch > 50 and epoch < 170:
        #     parameter = self.parameter
        #     self.parameter *= self.p_aum
        #     print(f'Updated parameter from {parameter} to {self.parameter}')

    def save_checkpoint(self, path, epoch=None):
        if epoch is None:
            torch.save(self.e2e_model.state_dict(), '{}/e2e_model.pth'.format(path))
        else:
            torch.save(self.e2e_model.state_dict(), '{}/generator_{}.pth'.format(path, epoch))

    def load_checkpoint(self, path, epoch=None):
        if epoch is None:
            self.e2e_model.load_state_dict(torch.load('{}/e2e_model.pth'.format(path)))
        else:
            self.e2e_model.load_state_dict(torch.load('{}/e2e_model{}.pth'.format(path, epoch)))


# ---------------- Optical Encoder ---------------- #

class SPC(nn.Module):
    def __init__(self, batch_size, input_size, compression_rate=0.1, seed=0, noise=20, trainable=True, device='cpu'):
        super(SPC, self).__init__()
        self.batch_size = batch_size
        self.compression_rate = compression_rate
        self.seed = seed
        self.noise = noise
        self.trainable = trainable
        self.device = device

        self.build(input_size)

    def build(self, input_size):
        M, N, L = input_size
        self.shots = int(self.compression_rate * M * N * L)

        if self.seed is not None:
            torch.manual_seed(self.seed)

        data = torch.empty((self.shots, M, N)).normal_(mean=0.0, std=1.0) / torch.sqrt(torch.tensor(M * N))
        self.phi = torch.nn.Parameter(data, requires_grad=self.trainable)

    def get_measure(self, x):
        b, L, M, N = x.shape
        y = torch.einsum('blmn,kmn->bkl', x, self.phi)
        std = torch.sum(torch.pow(y, 2)) / ((b * M * (N + (L - 1))) * 10 ** (self.noise / 10))
        return y + torch.empty(b, 1, L, device=self.device).normal_(mean=0.0, std=float(torch.sqrt(std)))

    def get_transpose(self, y):
        x = torch.einsum('bkl,kmn->blmn', y, self.phi)
        return x / torch.max(x)

    def forward(self, x, only_measure=False, only_transpose=False):
        if only_transpose:
            return self.get_transpose(x)

        if only_measure:
            return self.get_measure(x)

        return self.get_transpose(self.get_measure(x))


# ---------------- Computational Decoder ---------------- #


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
