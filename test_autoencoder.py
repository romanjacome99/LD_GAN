#!/usr/bin/env python

import os
import argparse
from itertools import product
from pathlib import Path
import time
from models.autoencoder import HSAE

import torch.optim
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from colorama import Fore
from tqdm import tqdm

from models.losses import EMA_losses
from scripts.data import load_dataset
from scripts.functions import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def init_parser():
    parser = argparse.ArgumentParser(description='Training script')

    # data args

    parser.add_argument('--result-path', default='results',
                        help='path to save results')
    parser.add_argument('--save-path', default='gan_spectral_cifar10',
                        help='path to save specific experiment (This will be stored in result_path folder)')
    parser.add_argument('--load-checkpoint',
                        default='results/autoencoder_tanh_64_arad_lr0.001_epochs300',
                        help='path to load checkpoint (from the root path)')
    parser.add_argument('--use-tensorboard', default=True,
                        help='use tensorboard to save training in save_path folder')
    parser.add_argument('--training', default=True,
                        help='train the model')
    parser.add_argument('--seed', default=0,
                        help='seed for training')

    # datasets and model

    parser.add_argument('--dataset-path',
                        default=r'File_path',
                        help='path to dataset files')
    parser.add_argument('--dataset', type=str.lower, default='arad',
                        choices=['cave', 'kaist', 'arad'],
                        help='dataset name to be trained')
    parser.add_argument('--feature', default=48, type=int,  # 64: 4 channels, 48: 3 channels
                        help='number of feature maps')
    parser.add_argument('--spatial-size', default=64, type=int,
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
    parser.add_argument('--init-lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=None, type=float,
                        help='weight decay')

    # gpu config

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--is-cuda', default=True, type=bool,
                        help='use cuda or not')

    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # load dataset
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    train_loader, val_loader, test_loader = load_dataset(args.dataset, args.dataset_path, args.batch_size,
                                                         args.spatial_size, args.workers)

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # load model and hyperparams
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    device = 'cuda' if args.is_cuda else 'cpu'
    model = HSAE(args.feature, args.num_ch, args.init_lr, args.load_checkpoint, device)

    # summary model

    encoder_num_parameters = sum([l.nelement() for l in model.encoder.parameters()])
    decoder_num_parameters = sum([l.nelement() for l in model.decoder.parameters()])

    print(f'Encoder parameters: {encoder_num_parameters}')
    print(f'Decoder parameters: {decoder_num_parameters}')
    print(f'Total parameters: {encoder_num_parameters + decoder_num_parameters}')

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # load checkpoint
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    if args.seed:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if args.load_checkpoint:
        model.load_checkpoint(args.load_checkpoint)
        print('¡Model checkpoint loaded correctly!')

    else:
        raise ValueError('No checkpoint to load')

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # train
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    model.evaluate(test_loader)

    print('Fin')


if __name__ == '__main__':
    main()
