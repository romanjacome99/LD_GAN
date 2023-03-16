#!/usr/bin/env python

import os
import argparse
from pathlib import Path

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
    parser.add_argument('--load-path', default='results/Checkpoint_gan',
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
    model = DCGAN(args.batch_size, args.feature, args.num_ch, lr=args.lr, is_autoencoder=True,
                  save_path=save_path, device=device)

    # summary model

    encoder_num_parameters = sum([l.nelement() for l in model.encoder.parameters()])
    decoder_num_parameters = sum([l.nelement() for l in model.decoder.parameters()])

    print(f'Encoder parameters: {encoder_num_parameters}')
    print(f'Decoder parameters: {decoder_num_parameters}')
    print(f'Total autoencoder parameters: {encoder_num_parameters + decoder_num_parameters}')

    generator_num_parameters = sum([l.nelement() for l in model.generator.parameters()])
    discriminator_num_parameters = sum([l.nelement() for l in model.discriminator.parameters()])

    print(f'Generator parameters: {generator_num_parameters}')
    print(f'Discriminator parameters: {discriminator_num_parameters}')
    print(f'Total GAN parameters: {generator_num_parameters + discriminator_num_parameters}')

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # load checkpoint
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    try:
        model.load_checkpoint(args.load_path, is_autoencoder=True, is_gan=True)
        print('¡Autoencoder and GAN model checkpoints loaded correctly!')
    except:
        raise 'error loading checkpoints'

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # test
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    test_metrics = model.test_step(val_loader[0], args.max_epochs - 1, args.max_epochs)

    print('Test metrics:')
    for key, value in test_metrics.items():
        print(f'{key}: {value}')

    model.save_images(val_loader[0], args.max_epochs, save=True, show=True)

    print('Fin')


if __name__ == '__main__':
    main()
