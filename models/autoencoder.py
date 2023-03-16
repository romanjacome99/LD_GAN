import time
from pathlib import Path

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import spectral_angle_mapper
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from torchmetrics import MeanSquaredError, MeanAbsoluteError, StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from scripts.functions import AverageMeter
from scripts.spec2rgb import ColourSystem


def rgb_scheduler(param, epoch):
    if epoch != 0 and epoch % 15 == 0:
        return param * 1.2

    return param


class HSAE:
    def __init__(self, feature=64, num_ch=31, lr=4e-3, save_path=None, device='cuda',reg_param=1e-3):
        self.num_ch = num_ch
        self.writer = SummaryWriter(save_path)
        self.save_path = save_path
        self.device = device
        self.num_blocking = torch.cuda.is_available()
        self.reg_param = reg_param
        self.encoder = self.build_encoder(feature=feature, num_ch=num_ch)
        self.decoder = self.build_decoder(feature=feature, num_ch=num_ch)

        self.MSE = MeanSquaredError().to(device)
        self.MAE = MeanAbsoluteError().to(device)
        self.SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.PSNR = PeakSignalNoiseRatio().to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.9, patience=15, verbose=True)

        self.cs = ColourSystem(cs='sRGB', start=400, end=720, num=num_ch, device=self.device)

        if device == 'cuda' and torch.cuda.is_available():
            self.encoder.to(device)
            self.decoder.to(device)
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

    def save_metrics(self, name, metrics, epoch):
        for key, value in metrics.items():
            self.writer.add_scalar(f'{name}_{key}', value, epoch)

    def train(self, train_loader, epochs, val_loader=None):

        print("Beginning training")

        time_begin = time.time()
        for epoch in range(epochs):
            train_metrics = self.train_step(train_loader, epoch, epochs)
            self.save_metrics('train', train_metrics, epoch)

            if val_loader is not None:
                test_metrics = self.test_step(val_loader[0], epoch, epochs)
                self.save_metrics('test', test_metrics, epoch)

                if epoch == 0 or epoch % 25 == 0:
                    self.save_images('recons', val_loader[0], epoch)
                    self.save_images('big_recons', val_loader[1], epoch)
            # lr and reg scheduler
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], epoch)

            self.scheduler.step(epoch)


            self.save_checkpoint(f'{self.save_path}/checkpoints', epoch)

        print("Ending training")

        total_mins = (time.time() - time_begin) / 60
        print(f'Script finished in {total_mins:.2f} minutes')

    def train_step(self, dataloader, epoch, max_epochs):
        self.encoder.train()
        self.decoder.train()

        return self.forward(dataloader, epoch, max_epochs, kind='train', colour='red')

    def test_step(self, dataloader, epoch, max_epochs):
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()

            return self.forward(dataloader, epoch, max_epochs, kind='test', colour='green')

    def apply_l2_regularizer(self):
        reg_loss = 0
        for param in self.encoder.parameters():
            reg_loss += torch.norm(param)

        for param in self.decoder.parameters():
            reg_loss += torch.norm(param)

        return 1e-8 * reg_loss

    def apply_variance_regularizers(self,z):
        std_val = torch.std(z, 0,  keepdim=False)
        return torch.mean(std_val)


    def forward(self, dataloader, epoch, max_epochs, kind, colour):
        losses = AverageMeter()
        reg_losses = AverageMeter()
        reg_var_losses = AverageMeter()
        sam_x_losses = AverageMeter()
        ssim_x_losses = AverageMeter()
        psnr_x_losses = AverageMeter()

        dict_metrics = dict()
        data_loop = tqdm(enumerate(dataloader), total=len(dataloader), colour=colour)
        for _, data in data_loop:
            _, spec = data

            spec = Variable(spec.to(self.device, non_blocking=self.num_blocking))

            embedded = self.encoder(spec)
            reconstructed = self.decoder(embedded)
            reg_var = self.apply_variance_regularizers(embedded)

            loss = self.criterion(reconstructed, spec)
            reg_loss = self.apply_l2_regularizer()
            loss += reg_loss# + reg_var*self.reg_param


            losses.update(loss.item(), spec.size(0))
            reg_losses.update(reg_loss.item(), spec.size(0))
            reg_var_losses.update(reg_var.item(), spec.size(0))


            sam_x = spectral_angle_mapper(reconstructed, spec)
            ssim_x = self.SSIM(reconstructed, spec)
            psnr_x = self.PSNR(reconstructed, spec)

            sam_x_losses.update(sam_x.item(), spec.size(0))
            ssim_x_losses.update(ssim_x.item(), spec.size(0))
            psnr_x_losses.update(psnr_x.item(), spec.size(0))

            dict_metrics = dict(loss=losses.avg, sam_x=sam_x_losses.avg, ssim_x=ssim_x_losses.avg,
                                psnr_x=psnr_x_losses.avg,reg_var=reg_var_losses.avg)

            if kind == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            lr = format(self.optimizer.param_groups[0]['lr'], '.1e')
            data_loop.set_description(f'{kind.capitalize()}: Epoch [{epoch + 1} / {max_epochs}] lr: {lr}')
            data_loop.set_postfix(**dict_metrics)

        return dict_metrics

    def save_images(self, save_name, val_loader, epoch):

        if val_loader is not None:
            image_save_path = f'{self.save_path}/images'
            try:
                Path(image_save_path).mkdir(parents=True, exist_ok=False)
            except:
                pass

            self.encoder.eval()
            self.decoder.eval()

            rgb, spec = next(iter(val_loader))
            spec = spec.to(self.device, non_blocking=self.num_blocking)

            with torch.no_grad():
                embedded = self.encoder(spec)
                reconstructed = self.decoder(embedded)

            rgb = rgb.cpu().numpy().transpose(0, 2, 3, 1)
            rgb_spec = self.cs.spec_to_rgb_torch(spec).permute(0, 2, 3, 1).cpu().numpy()
            embedded = embedded.permute(0, 2, 3, 1).cpu().numpy()
            rgb_reconstructed = self.cs.spec_to_rgb_torch(reconstructed).permute(0, 2, 3, 1).cpu().numpy()

            # save images

            indices = np.linspace(0, len(rgb_spec) - 1, 4).astype(int)

            plt.figure(figsize=(30, 40))

            count = 1
            for idx in indices:
                plt.subplot(4, 3, count)
                plt.imshow(rgb[idx] ** 0.5)
                plt.title('Original')
                plt.axis('off')

                plt.subplot(4, 3, count + 1)
                plt.imshow(np.clip(rgb_reconstructed[idx], a_min=0., a_max=1.) ** 0.5)
                plt.title(f'ssim: {self.SSIM(reconstructed[idx:idx + 1], spec[idx:idx + 1]).item():.4f},'
                          f'psnr: {self.PSNR(reconstructed[idx:idx + 1], spec[idx:idx + 1]).item():.4f}')
                plt.axis('off')

                plt.subplot(4, 3, count + 2)
                plt.imshow(((embedded[idx] + 1) / 2) ** 0.5)
                plt.title('Embedded')
                plt.axis('off')

                count += 3

            plt.savefig('{}/{}_{:03d}.png'.format(image_save_path, save_name, epoch))
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
