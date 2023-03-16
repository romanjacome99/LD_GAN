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
    def __init__(self, batch_size, feature=64, num_ch=31, nz=100, ngf=64, ndf=64, nc=3, lr=2e-4, is_autoencoder=False,
                 save_path=None, device='cuda'):
        self.batch_size = batch_size
        self.num_ch = num_ch
        self.writer = SummaryWriter(save_path)
        self.is_autoencoder = is_autoencoder
        self.save_path = save_path
        self.device = device

        self.nz = nz
        self.real_label = 1.
        self.fake_label = 0.

        if is_autoencoder:
            self.encoder = self.build_encoder(feature=feature, num_ch=num_ch)
            self.decoder = self.build_decoder(feature=feature, num_ch=num_ch)

            for param in self.encoder.parameters():
                param.requires_grad = False

            for param in self.decoder.parameters():
                param.requires_grad = False

        self.generator = self.build_generator(nz=nz, ngf=ngf, nc=nc)
        self.discriminator = self.build_discriminator(ndf=ndf, nc=nc)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.PSNR = PeakSignalNoiseRatio().to(self.device)

        self.criterion = nn.BCELoss()
        self.fixed_noise = torch.randn(16, nz, 1, 1, device=self.device)
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))

        self.cs = ColourSystem(cs='sRGB', start=400, end=720, num=num_ch, device=self.device)

        if device == 'cuda' and torch.cuda.is_available():
            if autoencoder:
                self.encoder.to(device)
                self.decoder.to(device)

            self.generator.to(device)
            self.discriminator.to(device)

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

    def build_generator(self, nz, ngf, nc):
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
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def build_discriminator(self, ndf, nc):
        return nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
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

    def train(self, data_loader, epochs, val_loader=None):

        for epoch in range(epochs):

            # ---------------- Train ---------------- #

            self.discriminator.zero_grad()

            accuracy_metrics = {}
            errD_losses = AverageMeter()
            errG_losses = AverageMeter()
            D_x_losses = AverageMeter()
            D_G_z1_losses = AverageMeter()
            D_G_z2_losses = AverageMeter()

            iter = 0

            with tqdm(data_loader, unit='batch') as loop:
                for data in loop:
                    loop.set_description(f'Train: Epoch [{epoch + 1}/{epochs}]')

                    x = data[0]  # data[1] for RGB arad, data[0] for celeba and spectral arad
                    b_size = x.size(0)
                    y_real = torch.full((b_size,), self.real_label, dtype=torch.float)

                    noise = torch.randn(b_size, self.nz, 1, 1)
                    y_fake = torch.full((b_size,), self.fake_label, dtype=torch.float)

                    if self.device == 'cuda' and torch.cuda.is_available():
                        x = x.to(self.device, non_blocking=True)
                        y_real = y_real.to(self.device, non_blocking=True)

                        noise = noise.to(self.device, non_blocking=True)
                        y_fake = y_fake.to(self.device, non_blocking=True)

                    x = Variable(x)
                    y_real = Variable(y_real)

                    noise = Variable(noise)
                    y_fake = Variable(y_fake)

                    if self.is_autoencoder:
                        with torch.no_grad():
                            x = self.encoder(x)

                    # x = self.cs.spec_to_rgb_torch(x)
                    # x = torch.clamp(x, 0, 1)
                    # x = 2 * x - 1  # solo para rgb arad

                    metrics = self.train_step(x, y_real, noise, y_fake)

                    errD_losses.update(metrics['errD'], b_size)
                    errG_losses.update(metrics['errG'], b_size)
                    D_x_losses.update(metrics['D_x'], b_size)
                    D_G_z1_losses.update(metrics['D_G_z1'], b_size)
                    D_G_z2_losses.update(metrics['D_G_z2'], b_size)

                    metrics['errD'] = errD_losses.avg
                    metrics['errG'] = errG_losses.avg
                    metrics['D_x'] = D_x_losses.avg
                    metrics['D_G_z1'] = D_G_z1_losses.avg
                    metrics['D_G_z2'] = D_G_z2_losses.avg

                    for k, v in metrics.items():
                        if k not in accuracy_metrics:
                            accuracy_metrics[k] = 0
                        accuracy_metrics[k] += v

                    iter += 1
                    loop.set_postfix(**metrics)

            # ---------------- Log metrics ---------------- #

            if self.writer:
                for k, v in accuracy_metrics.items():
                    self.writer.add_scalar('train_{}'.format(k), v / iter, epoch)

            self.save_images(epoch)

            # ---------------- Save model ---------------- #

            if int(epoch) % 25 == 0:
                self.save_checkpoint(self.save_path, epoch=epoch)
                print(f'saved checkpoint at epoch {epoch}')

    def train_step(self, x, y_real, noise, y_fake):
        self.generator.zero_grad()
        self.optimizerD.zero_grad()

        ################################################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ################################################################

        # Train with all-real batch

        predicted = self.discriminator(x).view(-1)
        errD_real = self.criterion(predicted, y_real)

        # backprop
        errD_real.backward()
        D_x = predicted.mean().item()

        # Train with all-fake batch
        fake = self.generator(noise)
        predicted = self.discriminator(fake.detach()).view(-1)
        errD_fake = self.criterion(predicted, y_fake)

        # backprop
        errD_fake.backward()
        D_G_z1 = predicted.mean().item()

        errD = errD_real + errD_fake
        self.optimizerD.step()

        ################################################################
        # (2) Update G network: maximize log(D(G(z)))
        ################################################################

        self.generator.zero_grad()
        self.optimizerG.zero_grad()

        predicted = self.discriminator(fake).view(-1)
        errG = self.criterion(predicted, y_real)

        # backprop
        errG.backward()
        D_G_z2 = predicted.mean().item()

        self.optimizerG.step()

        return dict(errD=errD.item(), errG=errG.item(), D_x=D_x, D_G_z1=D_G_z1, D_G_z2=D_G_z2)

    def save_images(self, epoch):
        image_save_path = f'{self.save_path}/images'
        try:
            Path(image_save_path).mkdir(parents=True, exist_ok=False)
        except:
            pass

        self.generator.eval()

        with torch.no_grad():
            generated_images = self.generator(self.fixed_noise)
            if self.is_autoencoder:
                generated_images = self.decoder(generated_images)

        if self.is_autoencoder:
            generated_images = self.cs.spec_to_rgb_torch(generated_images)
        else:
            generated_images = (generated_images + 1) / 2

        generated_images = generated_images.permute(0, 2, 3, 1).cpu().numpy()

        # save images

        plt.figure(figsize=(20, 20))

        plt.subplot(2, 2, 1)
        plt.imshow(generated_images[0])
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(generated_images[1])
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(generated_images[2])
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(generated_images[3])
        plt.axis('off')

        plt.savefig('{}/gen_{:03d}.png'.format(image_save_path, epoch))
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
