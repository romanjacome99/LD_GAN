#!/usr/bin/env python

import os
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch.optim
import torch.utils.data

from models.dcgan import DCGAN
from scripts.data import Dataset
from scripts.functions import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def init_parser():
    parser = argparse.ArgumentParser(description='Training script')

    # data args

    parser.add_argument('--result-path', default='results',
                        help='path to save results')
    parser.add_argument('--save-path', default='test_embedded_gan',
                        help='path to save specific experiment (This will be stored in result_path folder)')
    parser.add_argument('--load-path', default='results/Checkpoint_gan_spectral',
                        help='path to load checkpoint (from the root path)')

    # datasets and model

    parser.add_argument('--dataset-path', default=r'C:\Users\EMMANUELMARTINEZ\Documents\Datasets\ARAD',
                        help='path to dataset files')
    parser.add_argument('--dataset', type=str.lower, default='arad',
                        choices=['cave', 'kaist', 'arad', 'celeba'],
                        help='dataset name to be trained')
    parser.add_argument('--feature', default=48, type=int,  # 64: 4 channels, 48: 3 channels
                        help='number of feature maps')
    parser.add_argument('--patch-size', default=64, type=int,
                        help='spatial size of the input')
    parser.add_argument('--num-ch', default=31, type=int,
                        help='number of channels')

    # hyper parameters

    parser.add_argument('--init-epoch', default=0, type=int,
                        help='number of the initial epoch')
    parser.add_argument('--max-epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        help='mini-batch size (default: 128)')

    parser.add_argument('--optimizer', default='adam', type=str.lower, choices=['adam', 'adam_w', 'sgd'],
                        help='type of optimizer')
    parser.add_argument('--lr', default=2e-4, type=float,
                        help='learning rate for discriminator')
    parser.add_argument('--betas-D', default=(0.5, 0.999), type=tuple,
                        help='betas for adam optimizer for discriminator')
    parser.add_argument('--betas-G', default=(0.5, 0.999), type=tuple,
                        help='betas for adam optimizer for generator')

    # gpu config

    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # path configurations
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    save_path = f'{args.result_path}/{args.save_path}_{args.dataset}'
    save_path += f'_bs{args.batch_size}_lr{args.lr}_epochs{args.max_epochs}'
    checkpoint_path = f'{save_path}/checkpoints'
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    save_config(save_path, os.path.basename(__file__), args)  # Save the experiment config in a .txt file

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # load dataset
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    dataset = Dataset(args.dataset_path, args.batch_size, args.patch_size, args.workers)
    train_loader, val_loader = dataset.get_arad_dataset()

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # load model and hyperparams
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DCGAN(args.batch_size, args.feature, args.num_ch, lr=args.lr, is_autoencoder=False,
                  save_path=save_path, device=device)

    # summary model

    # encoder_num_parameters = sum([l.nelement() for l in model.encoder.parameters()])
    # decoder_num_parameters = sum([l.nelement() for l in model.decoder.parameters()])
    #
    # print(f'Encoder parameters: {encoder_num_parameters}')
    # print(f'Decoder parameters: {decoder_num_parameters}')
    # print(f'Total autoencoder parameters: {encoder_num_parameters + decoder_num_parameters}')

    generator_num_parameters = sum([l.nelement() for l in model.generator.parameters()])
    discriminator_num_parameters = sum([l.nelement() for l in model.discriminator.parameters()])

    print(f'Generator parameters: {generator_num_parameters}')
    print(f'Discriminator parameters: {discriminator_num_parameters}')
    print(f'Total GAN parameters: {generator_num_parameters + discriminator_num_parameters}')

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # load checkpoint
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    try:
        model.load_checkpoint(args.load_path, is_autoencoder=False, is_gan=True)
        print('¡Autoencoder and GAN model checkpoints loaded correctly!')
    except:
        raise 'error loading checkpoints'

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # create dataset
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    # num_patches = 57536 // 2
    num_patches = 57536
    # num_patches = 57536 * 2

    dataset_path = Path(r'C:\Users\EMMANUELMARTINEZ\Documents\Datasets\ARAD')
    initial_dataset_path = dataset_path / f'gen_spec_train_64x64x31_num_samples_{num_patches}.h5'

    with h5py.File(initial_dataset_path, 'w') as d:
        # Init data, save as uint16
        data = d.create_dataset('spec', (num_patches, args.patch_size, args.patch_size, args.num_ch), dtype=np.uint16)
        rgb_data = d.create_dataset('rgb', (num_patches, args.patch_size, args.patch_size, 3), dtype=np.uint16)

        model.generator.eval()

        torch.manual_seed(0)

        for i in range(num_patches):
            noise = torch.randn(1, 100, 1, 1).to(device, non_blocking=torch.cuda.is_available())

            generated_embedded = model.generator(noise)
            generated_spec = (generated_embedded + 1) / 2
            generated_rgb = model.cs.spec_to_rgb_torch(generated_spec)

            generated_spec = np.clip(generated_spec.permute(0, 2, 3, 1).cpu().detach().numpy().squeeze(), 0, 1)
            generated_rgb = np.clip(generated_rgb.permute(0, 2, 3, 1).cpu().detach().numpy().squeeze(), 0, 1)

            # # Convert the data to uint16 ndarray
            generated_spec = ((2 ** 16 - 1) * np.asarray(generated_spec)).astype(np.uint16)
            generated_rgb = ((2 ** 16 - 1) * np.asarray(generated_rgb)).astype(np.uint16)

            data[i] = generated_spec
            rgb_data[i] = generated_rgb

            # Print progress
            print(f"Processed {i} samples.")

    print('Fin')


if __name__ == '__main__':
    main()
