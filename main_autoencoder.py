#!/usr/bin/env python

import os
import argparse
from pathlib import Path
from models.autoencoder import HSAE

import torch.optim
import torch.utils.data

from scripts.data import Dataset
from scripts.functions import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def init_parser():
    parser = argparse.ArgumentParser(description='Training script')

    # data args

    parser.add_argument('--result-path', default='results_cvpr_2',
                        help='path to save results')
    parser.add_argument('--save-path', default='autoencoder',
                        help='path to save specific experiment (This will be stored in result_path folder)')
    parser.add_argument('--load-path', default=None,
                        help='path to load checkpoint (from the root path)')

    # datasets and model

    parser.add_argument('--dataset-path', default=r'C:\Roman\datasets\ARAD',
                        help='path to dataset files')
    parser.add_argument('--dataset', type=str.lower, default='arad',
                        help='dataset name to be trained')
    parser.add_argument('--feature', default=48, type=int,  # 64: 4 channels, 48: 3 channels
                        help='number of feature maps')
    parser.add_argument('--patch-size', default=256, type=int,
                        help='spatial size of the input')
    parser.add_argument('--num-ch', default=31, type=int,
                        help='number of channels')

    # hyper parameters

    parser.add_argument('--init-epoch', default=0, type=int,
                        help='number of the initial epoch')
    parser.add_argument('--max-epochs', default=300, type=int,
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        help='mini-batch size (default: 64)')

    parser.add_argument('--optimizer', default='adam', type=str.lower, choices=['adam'],
                        help='type of optimizer')
    parser.add_argument('--init-lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--reg_param', default=1e-3, type=float,
                        help='initial learning rate')
    # gpu config

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()


    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # path configurations
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    args.feature = 16
    save_path = f'{args.result_path}/{args.save_path}_{args.patch_size}_{args.dataset}' \
                f'_lr{args.init_lr}_epochs{args.max_epochs}_n_{args.feature}'
    checkpoint_path = f'{save_path}/checkpoints'
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    save_config(save_path, os.path.basename(__file__), args)  # Save the experiment config in a .txt file

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # load dataset
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    dataset = Dataset(args.dataset_path, args.batch_size, args.patch_size, args.workers,da=None)
    train_loader, val_loader = dataset.get_arad_dataset()

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # load model and hyperparams
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HSAE(args.feature, args.num_ch, args.init_lr, save_path, device,reg_param=args.reg_param)

    # summary model

    encoder_num_parameters = sum([l.nelement() for l in model.encoder.parameters()])
    decoder_num_parameters = sum([l.nelement() for l in model.decoder.parameters()])

    print(f'Encoder parameters: {encoder_num_parameters}')
    print(f'Decoder parameters: {decoder_num_parameters}')
    print(f'Total parameters: {encoder_num_parameters + decoder_num_parameters}')

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # load checkpoint
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    if args.load_path:
        try:
            model.load_checkpoint(args.load_path)
            print('¡Model checkpoint loaded correctly!')
        except:
            raise 'error loading checkpoints'

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # train
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    model.train(train_loader, args.max_epochs, val_loader=val_loader)
    model.save_checkpoint(checkpoint_path)

    print('Fin')


if __name__ == '__main__':

    main()
