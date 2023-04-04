#!/usr/bin/env python
import os
import argparse
from itertools import product
from pathlib import Path

import torch.optim
import torch.utils.data

from models.measure2spec_SR import M2S_SR
from scripts.data import Dataset
from scripts.functions import *

# activate gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def init_parser():
    parser = argparse.ArgumentParser(description='Training script')

    # data args

    parser.add_argument('--result-path', default='results_cvpr',
                        help='path to save results')
    parser.add_argument('--save-path', default='ssr',
                        help='path to save specific experiment (This will be stored in result_path folder)')
    parser.add_argument('--load-path', default=None,
                        help='path to load checkpoint (from the root path)')

    # datasets and model

    parser.add_argument('--dataset-path',
                        default=r'File_path',
                        help='path to dataset files')
    parser.add_argument('--gen-dataset-path',
                        default='File_path',
                        # default=None,
                        help='path to generated dataset files')
    parser.add_argument('--dataset', type=str.lower, default='arad',
                        choices=['arad', 'cifar10', 'cifar100', 'imagenet'],
                        help='dataset name to be trained')
    parser.add_argument('--input-shape', type=tuple, default=(256, 256, 31),  # default=(64, 64, 31),
                        help='size of the patches')
    parser.add_argument('--patch-size', type=int, default=128,
                        help='size of the patches')
    parser.add_argument('--bands', default=3, type=int,
                        help='number of bands for the latent space')
    parser.add_argument('--num-samples', default=899,
                        help='num of generated samples')
    parser.add_argument('--real-samples', default=1.0,
                        help='num of real samples to be trained')
    parser.add_argument('--syn-samples', default=1.0,
                        help='num of synthetic samples to be trained')

    parser.add_argument('--mask-name', type=str.lower, default=None,  # 'rand_mask_128',
                        choices=['mask', 'bn_mask_128', 'rand_mask_128'],
                        help='mask name to be used')
    parser.add_argument('--trainable-mask', type=bool, default=False,
                        help='is the mask trainable?')
    parser.add_argument('--sdf', type=int, default=2,
                        help='spatial decimator factor')
    parser.add_argument('--ldf', type=int, default=4,
                        help='spectral decimator factor')
    parser.add_argument('--mask-seed', type=int, default=0,
                        help='seed for the mask')
    parser.add_argument('-m', '--model', type=str.lower,
                        choices=['unetcompiled', 'dssp', 'dgsmp'], default='unetcompiled',
                        help='model name to be trained (model will be selected according to dataset name)'
                             'dssp: (cassi + hqs)'
                             'dgsmp: (cassi + bayesian gsm)')

    # hyper parameters

    parser.add_argument('--init-epoch', default=0, type=int,
                        help='initial epoch')
    parser.add_argument('--max-epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        help='mini-batch size', dest='batch_size')
    parser.add_argument('--init-lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--ae-reg-param', default=None, type=float,
                        help='regularization parameter for the autoencoder')
    parser.add_argument('--gan-reg-param', default=None, type=float,
                        help='regularization parameter for the gan')

    # gpu config

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    return parser


def main(reg_param_ae,reg_param):
    parser = init_parser()
    args = parser.parse_args()

    args.reg_param_ae = reg_param_ae
    args.reg_param = reg_param

    if args.model == 'dssp':
        args.stride = 1
    else:
        args.stride = 2

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # path configurations
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    save_path = f'{args.result_path}/{args.save_path}_{args.model}_no_noise_{args.dataset}_mask{args.mask_name}' \
                f'_bands{args.bands}_emb_full_DA_{args.num_samples}_real{args.real_samples}_syn{args.syn_samples}_sdf_{args.sdf}_ldf_{args.ldf}_reg_param_gan_{reg_param}_reg_param_ae_{reg_param_ae}' \
                f'_lr{args.init_lr}_bs{args.batch_size}_epochs{args.max_epochs}'

    print(f'Experiment will be saved in {save_path}')

    checkpoint_path = f'{save_path}/checkpoints'
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    save_config(save_path, os.path.basename(__file__), args)  # Save the experiment config in a .txt file

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # load dataset
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    dataset = Dataset(args.dataset_path, args.batch_size, args.patch_size, args.workers)

    if args.num_samples > 0:
        args.gen_dataset_path = f'generated_data/gen_reg_max_var_param_gan_{reg_param}_reg_param_ae{reg_param_ae}/gen_emb_train_128x128x31_num_samples_3596.h5'
    else:
        args.gen_dataset_path = None

    train_loader, val_loader = dataset.get_arad_dataset(gen_dataset_path=args.gen_dataset_path,
                                                        real=args.real_samples, syn=args.syn_samples)

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # load model and hyperparams
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m2s_model = M2S_SR(args.model, args.input_shape, args.stride, args.patch_size, args.mask_name,
                       args.trainable_mask, args.mask_seed, args.sdf, args.ldf, args.init_lr, save_path=save_path,
                       device=device)

    # summary model

    num_parameters = sum([l.nelement() for l in m2s_model.computational_decoder.parameters()])
    print(f'Number of parameters: {num_parameters}')

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # load checkpoint
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    if args.load_path:
        m2s_model.load_checkpoint(args.load_path)
        print('¡Model checkpoint loaded correctly!')
        raise "Por aquí no es xd"

    else:
        pass  # raise ValueError('No checkpoint path provided')

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # train
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    m2s_model.train(train_loader, args.init_epoch, args.max_epochs, val_loader=val_loader[0])
    m2s_model.save_checkpoint(checkpoint_path)


if __name__ == '__main__':
    reg_param_aes = [0.0, 1e-5, 1e-3]
    reg_params = [0.0, 1e-5, 1e-3]


    for reg_param in reg_params:
        for reg_param_ae in reg_param_aes:
            main(reg_param_ae,reg_param)
