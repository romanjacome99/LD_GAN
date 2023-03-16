import time
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

# create an autoencoder model for hyperspectral images using torch
from scripts.functions import AverageMeter
from scripts.spec2rgb import ColourSystem


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGAN:
    def __init__(self, batch_size, bands=1, num_ch=31, nz=100, ngf=64, ndf=64, nc=3, lr=2e-4, reg_param=None,
                 is_autoencoder=False, save_path=None, device='cuda'):
        self.batch_size = batch_size
        self.num_ch = num_ch
        self.writer = SummaryWriter(save_path)
        self.is_autoencoder = is_autoencoder
        self.save_path = save_path
        self.device = device
        self.num_blocking = torch.cuda.is_available()

        self.nz = nz
        self.real_label = 1.
        self.fake_label = 0.

        if is_autoencoder:
            self.encoder = self.build_encoder(bands=bands, num_ch=num_ch)
            self.decoder = self.build_decoder(bands=bands, num_ch=num_ch)

            for param in self.encoder.parameters():
                param.requires_grad = False

            for param in self.decoder.parameters():
                param.requires_grad = False

        self.generator = self.build_generator(nz=nz, ngf=ngf, bands=bands if is_autoencoder else num_ch)
        self.discriminator = self.build_discriminator(ndf=ndf, bands=bands if is_autoencoder else num_ch)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.criterion = nn.BCELoss()
        self.reg_param = reg_param

        self.fixed_noise = torch.randn(16, nz, 1, 1, device=self.device)
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))

        self.cs = ColourSystem(cs='sRGB', start=400, end=720, num=num_ch, device=self.device)

        if device == 'cuda' and torch.cuda.is_available():
            if is_autoencoder:
                self.encoder.to(device)
                self.decoder.to(device)

            self.generator.to(device)
            self.discriminator.to(device)

            self.criterion = self.criterion.to(device)
            print('Using GPU')

    def build_encoder(self, num_ch=31, bands=3):
        return nn.Sequential(
            nn.Conv2d(num_ch, 16 * bands, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16 * bands, 16 * bands, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16 * bands, 8 * bands, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8 * bands, 8 * bands, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8 * bands, 4 * bands, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(4 * bands, 2 * bands, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(2 * bands, bands, 3, padding=1),
            nn.Tanh(),
        )

    def build_decoder(self, bands=3, num_ch=31):
        return nn.Sequential(
            nn.Conv2d(bands, 2 * bands, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(2 * bands, 4 * bands, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(4 * bands, 8 * bands, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8 * bands, 8 * bands, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8 * bands, 16 * bands, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16 * bands, 16 * bands, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16 * bands, num_ch, 3, padding=1),
            nn.ReLU(True)
        )

    def build_generator(self, nz, ngf, bands):
        return nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            ##########################################################
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            ##########################################################
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, bands, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def build_discriminator(self, ndf, bands):
        return nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(bands, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #####################################################
            # input is (nc) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nc) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #####################################################
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def save_metrics(self, name, metrics, epoch):
        for key, value in metrics.items():
            self.writer.add_scalar(f'{name}_{key}', value, epoch)

    def apply_variance_regularizers(self, z):
        std_val = torch.std(z, 0, keepdim=False)
        return torch.mean(std_val)

    def train(self, train_loader, epochs, val_loader=None):

        print("Beginning training")

        time_begin = time.time()
        for epoch in range(epochs):
            train_metrics = self.train_step(train_loader, epoch, epochs)
            self.save_metrics('train', train_metrics, epoch)

            if val_loader is not None:
                test_metrics = self.test_step(val_loader[0], epoch, epochs)

                for key, value in test_metrics.items():
                    self.writer.add_scalar(f'test_{key}', value, epoch)

                if epoch == 0 or epoch % 10 == 0 or epoch == epochs - 1:
                    self.save_images(val_loader[0], epoch)

            # lr and reg scheduler
            self.writer.add_scalar("lrG", self.optimizerG.param_groups[0]['lr'], epoch)
            self.writer.add_scalar("lrD", self.optimizerD.param_groups[0]['lr'], epoch)

            # if epoch == 0 or epoch % 20 == 0 or epoch == epochs - 1:
            self.save_checkpoint(f'{self.save_path}/checkpoints', epoch)

        print("Ending training")

        total_mins = (time.time() - time_begin) / 60
        print(f'Script finished in {total_mins:.2f} minutes')

    def train_step(self, dataloader, epoch, max_epochs):
        self.generator.train()
        self.discriminator.train()

        return self.forward(dataloader, epoch, max_epochs, kind='train', colour='red')

    def test_step(self, dataloader, epoch, max_epochs):
        with torch.no_grad():
            self.generator.eval()
            self.discriminator.eval()

            return self.forward(dataloader, epoch, max_epochs, kind='test', colour='green')

    def forward(self, dataloader, epoch, max_epochs, kind, colour):
        errD_losses = AverageMeter()
        errG_losses = AverageMeter()
        reg_var_losses = AverageMeter()
        D_x_losses = AverageMeter()
        D_G_z1_losses = AverageMeter()
        D_G_z2_losses = AverageMeter()

        dict_metrics = dict()
        data_loop = tqdm(enumerate(dataloader), total=len(dataloader), colour=colour)
        for _, data in data_loop:
            _, spec = data  # real spec
            # spec, _ = data  # rgb

            spec = Variable(spec.to(self.device, non_blocking=self.num_blocking))

            if self.is_autoencoder:
                with torch.no_grad():
                    embedded = self.encoder(spec)

            else:
                embedded = 2 * spec - 1

            b_size = embedded.size(0)
            y_real = torch.full((b_size,), self.real_label, dtype=torch.float).to(self.device,
                                                                                  non_blocking=self.num_blocking)

            noise = torch.randn(b_size, self.nz, 1, 1).to(self.device, non_blocking=self.num_blocking)
            y_fake = torch.full((b_size,), self.fake_label, dtype=torch.float).to(self.device,
                                                                                  non_blocking=self.num_blocking)

            if kind == 'train':
                self.generator.zero_grad()
                self.optimizerD.zero_grad()

            ################################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ################################################################

            # Train with all-real batch

            predicted = self.discriminator(embedded).view(-1)
            errD_real = self.criterion(predicted, y_real)

            # backprop
            if kind == 'train':
                errD_real.backward()

            D_x = predicted.mean().item()

            # Train with all-fake batch
            fake = self.generator(noise)

            if self.reg_param is not None:
                reg_var = self.apply_variance_regularizers(fake)

            predicted = self.discriminator(fake.detach()).view(-1)
            errD_fake = self.criterion(predicted, y_fake)

            # backprop
            if kind == 'train':
                errD_fake.backward()

            D_G_z1 = predicted.mean().item()

            errD = errD_real + errD_fake

            if kind == 'train':
                self.optimizerD.step()
                self.generator.zero_grad()
                self.optimizerG.zero_grad()

            ################################################################
            # (2) Update G network: maximize log(D(G(z)))
            ################################################################

            predicted = self.discriminator(fake).view(-1)
            errG = self.criterion(predicted, y_real)

            if self.reg_param is not None:
                errG -= reg_var * self.reg_param
                reg_var_losses.update(reg_var.item(), b_size)

            # backprop
            if kind == 'train':
                errG.backward()
                self.optimizerG.step()

            D_G_z2 = predicted.mean().item()

            errD_losses.update(errD.item(), b_size)
            errG_losses.update(errG.item(), b_size)
            D_x_losses.update(D_x, b_size)
            D_G_z1_losses.update(D_G_z1, b_size)
            D_G_z2_losses.update(D_G_z2, b_size)

            dict_metrics = dict(errD=errD_losses.avg, errG=errG_losses.avg, reg_var=reg_var_losses.avg,
                                D_x=D_x_losses.avg, D_G_z1=D_G_z1_losses.avg, D_G_z2=D_G_z2_losses.avg)

            ########################################################################################
            ########################################################################################

            lr = format(self.optimizerG.param_groups[0]['lr'], '.1e')
            data_loop.set_description(f'{kind.capitalize()}: Epoch [{epoch + 1} / {max_epochs}] lr: {lr}')
            data_loop.set_postfix(**dict_metrics)

        return dict_metrics

    def save_images(self, val_loader, epoch, save=True, show=False):
        image_save_path = f'{self.save_path}/images'
        try:
            Path(image_save_path).mkdir(parents=True, exist_ok=False)
        except:
            pass

        self.generator.eval()

        rgb, _ = next(iter(val_loader))
        rgb = rgb.to(self.device, non_blocking=self.num_blocking)

        with torch.no_grad():
            generated_embedded = self.generator(self.fixed_noise)

            if self.is_autoencoder:
                self.decoder.eval()
                generated_spec = self.cs.spec_to_rgb_torch(self.decoder(generated_embedded))
                generated_embedded = (generated_embedded + 1) / 2

            else:
                generated_embedded = (generated_embedded + 1) / 2
                generated_spec = self.cs.spec_to_rgb_torch(generated_embedded)

        rgb = rgb.permute(0, 2, 3, 1).cpu().numpy()
        generated_embedded = generated_embedded.permute(0, 2, 3, 1).cpu().numpy()
        generated_spec = generated_spec.permute(0, 2, 3, 1).cpu().numpy()

        # save images

        indices = np.linspace(0, len(self.fixed_noise) - 1, 16).astype(int)
        rgb_indices = np.linspace(0, len(rgb) - 1, 16).astype(int)

        count = 1

        if self.is_autoencoder:
            plt.figure(figsize=(60, 20))
            plt.suptitle('gen embedded - gen spec - gt rgb')

            for i in indices:
                plt.subplot(4, 12, count)
                plt.imshow(np.clip(generated_embedded[i, ..., :3], 0, 1) ** 0.5)
                plt.axis('off')

                plt.subplot(4, 12, count + 1)
                plt.imshow(np.clip(generated_spec[i], 0, 1) ** 0.5)
                plt.axis('off')

                plt.subplot(4, 12, count + 2)
                plt.imshow(rgb[rgb_indices[i]] ** 0.5)
                plt.axis('off')

                count += 3

        else:
            plt.figure(figsize=(40, 20))
            plt.suptitle('gen spec - gt rgb')

            for i in indices:
                plt.subplot(4, 8, count)
                plt.imshow(np.clip(generated_spec[i], 0, 1) ** 0.5)
                plt.axis('off')

                plt.subplot(4, 8, count + 1)
                plt.imshow(rgb[rgb_indices[i]] ** 0.5)
                plt.axis('off')

                count += 2

        if save:
            plt.savefig('{}/generated_{:03d}.png'.format(image_save_path, epoch))

        if show:
            plt.show()

        plt.close()

    def save_checkpoint(self, path, epoch=None):
        if epoch is None:
            if self.is_autoencoder:
                torch.save(self.encoder.state_dict(), '{}/encoder.pth'.format(path))
                torch.save(self.decoder.state_dict(), '{}/decoder.pth'.format(path))

            torch.save(self.generator.state_dict(), '{}/generator.pth'.format(path))
            torch.save(self.discriminator.state_dict(), '{}/discriminator.pth'.format(path))
        else:
            if self.is_autoencoder:
                torch.save(self.encoder.state_dict(), '{}/encoder_{}.pth'.format(path, epoch))
                torch.save(self.decoder.state_dict(), '{}/decoder_{}.pth'.format(path, epoch))

            torch.save(self.generator.state_dict(), '{}/generator_{}.pth'.format(path, epoch))
            torch.save(self.discriminator.state_dict(), '{}/discriminator_{}.pth'.format(path, epoch))

    def load_checkpoint(self, path, epoch=None, is_autoencoder=False, is_gan=False):
        if epoch is None:
            if is_autoencoder:
                self.encoder.load_state_dict(torch.load('{}/encoder.pth'.format(path)))
                self.decoder.load_state_dict(torch.load('{}/decoder.pth'.format(path)))
                print('loaded autoencoder checkpoint')

            if is_gan:
                self.generator.load_state_dict(torch.load('{}/generator.pth'.format(path)))
                self.discriminator.load_state_dict(torch.load('{}/discriminator.pth'.format(path)))
                print('loaded gan checkpoint')

        else:
            if is_autoencoder:
                self.encoder.load_state_dict(torch.load('{}/encoder_{}.pth'.format(path, epoch)))
                self.decoder.load_state_dict(torch.load('{}/decoder_{}.pth'.format(path, epoch)))
                print('loaded autoencoder checkpoint')

            if is_gan:
                self.generator.load_state_dict(torch.load('{}/generator_{}.pth'.format(path, epoch)))
                self.discriminator.load_state_dict(torch.load('{}/discriminator_{}.pth'.format(path, epoch)))
                print('loaded gan checkpoint')
