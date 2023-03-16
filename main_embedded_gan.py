#!/usr/bin/env python

import os
import argparse
from itertools import product
from pathlib import Path

import torch.optim
import torch.utils.data

from models.dcgan_256 import DCGAN
from scripts.data import Dataset
from scripts.functions import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def init_parser():
    parser = argparse.ArgumentParser(description='Training script')

    # data args

    parser.add_argument('--result-path', default='results_cvpr_2',
                        help='path to save results')
    parser.add_argument('--save-path', default='embedded_gan',
                        help='path to save specific experiment (This will be stored in result_path folder)')
    parser.add_argument('--load-path',
                        # default='results/autoencoder_ps256_arad_bandsx_lr0.001_reg_varNone_bs8_epochs300/checkpoints',
                        default='results/autoencoder_ps64_arad_bandsx_lr0.001_reg_varNone_bs64_epochs300/checkpoints',
                        help='path to load checkpoint (from the root path)')

    # datasets and model

    parser.add_argument('--dataset-path',
                        default=r'C:\Roman\datasets\ARAD',
                        help='path to dataset files')
    parser.add_argument('--dataset', type=str.lower, default='arad',
                        choices=['cave', 'kaist', 'arad', 'celeba'],
                        help='dataset name to be trained')
    parser.add_argument('--bands', default=3, type=int,
                        help='number of bands for the latent space')
    parser.add_argument('--patch-size', default=256, type=int,
                        help='spatial size of the input')
    parser.add_argument('--num-ch', default=31, type=int,
                        help='number of channels')

    # hyper parameters

    parser.add_argument('--init-epoch', default=0, type=int,
                        help='number of the initial epoch')
    parser.add_argument('--max-epochs', default=200, type=int,
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        help='mini-batch size (default: 8)')

    parser.add_argument('--optimizer', default='adam', type=str.lower, choices=['adam', 'adam_w', 'sgd'],
                        help='type of optimizer')
    parser.add_argument('--lr', default=2e-4, type=float,
                        help='learning rate for discriminator')
    parser.add_argument('--betas-D', default=(0.5, 0.999), type=tuple,
                        help='betas for adam optimizer for discriminator')
    parser.add_argument('--betas-G', default=(0.5, 0.999), type=tuple,
                        help='betas for adam optimizer for generator')
    parser.add_argument('--reg-var', default=None, type=float,
                        help='regularization variance')
    parser.add_argument('--nz', default=100, type=int)
    parser.add_argument('--ngf', default=48, type=int)
    parser.add_argument('--ndf', default=48, type=int)

    # gpu config

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    return parser


def main(reg_param_ae,reg_param):  # num_run, params):
    parser = init_parser()
    args = parser.parse_args()
    args.reg_param_ae = reg_param_ae
    args.reg_var = reg_param


    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # path configurations
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    save_path = f'{args.result_path}/{args.save_path}_{args.dataset}_bands{args.bands}' \
                f'_nz{args.nz}_ngf{args.ngf}_ndf{args.ndf}'
    save_path += f'_bs{args.batch_size}_lr{args.lr}_reg_var{args.reg_var}_epochs{args.max_epochs}'
    checkpoint_path = f'{save_path}/checkpoints'
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    save_config(save_path, os.path.basename(__file__), args)  # Save the experiment config in a .txt file

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # load dataset
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    dataset = Dataset(args.dataset_path, args.batch_size, args.patch_size, args.workers)
    train_loader, val_loader = dataset.get_arad_dataset(real=1.0, syn=0.0)

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # load model and hyperparams
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DCGAN(args.batch_size, args.bands, args.num_ch, nz=args.nz, ngf=args.ngf, ndf=args.ndf,
                  lr=args.lr, reg_param=args.reg_var, is_autoencoder=True,
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
        args.load_path = f'results_cvpr_2/autoencoder_64_arad_lr0.001_epochs300_max_reg_param_{reg_param_ae}/checkpoints'
        model.load_checkpoint(args.load_path, is_autoencoder=True)
        print('¡Autoencoder model checkpoint loaded correctly!')
    except:
        raise 'error loading checkpoints'

    # test autoencoder

    # model.evaluate_autoencoder(val_loader[0])
    # model.evaluate_autoencoder(val_loader[1])

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # train
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    model.train(train_loader, args.max_epochs, val_loader=val_loader)
    model.save_checkpoint(checkpoint_path)

    print('Fin')


if __name__ == '__main__':
    for reg_param_ae in [1e-3]:
        for reg_param in [1e-5, 1e-3]:
            main(reg_param_ae,reg_param)
