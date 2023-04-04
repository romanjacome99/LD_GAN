#!/usr/bin/env python
import os
import argparse
from itertools import product
from pathlib import Path

import torch.optim
import torch.utils.data

from models.measure2spec import M2S
from models.rgb2spec import RGB2SPEC
from scripts.data import Dataset
from scripts.functions import *

# activate gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def init_parser():
    parser = argparse.ArgumentParser(description='Training script')

    # data args

    parser.add_argument('--result-path', default='results',
                        help='path to save results')
    parser.add_argument('--save-path', default='rgb2spec',
                        help='path to save specific experiment (This will be stored in result_path folder)')
    parser.add_argument('--load-path', default=None,
                        help='path to load checkpoint (from the root path)')

    # datasets and model

    parser.add_argument('--dataset-path',

                        default=r'File_path',
                        help='path to dataset files')
    parser.add_argument('--gen-dataset-path',
                        default=r'File_path',
                        # default=None,
                        help='path to generated dataset files')
    parser.add_argument('--dataset', type=str.lower, default='arad',
                        choices=['arad', 'cifar10', 'cifar100', 'imagenet'],
                        help='dataset name to be trained')
    parser.add_argument('--input-shape', type=tuple, default=(128,128, 31),  # default=(64, 64, 31),
                        help='size of the patches')
    parser.add_argument('--patch-size', type=int, default=128,
                        help='size of the patches')
    parser.add_argument('--bands', default=1, type=int,
                        help='number of bands for the latent space')
    parser.add_argument('--num-samples', default=899,  # 3596,
                        help='num of generated samples')
    parser.add_argument('--real-samples', default=1.0,
                        help='num of real samples to be trained')
    parser.add_argument('--syn-samples', default=1.0,
                        help='num of synthetic samples to be trained')

    parser.add_argument('-m', '--model', type=str.lower,
                        choices=['unet', 'dssp', 'dgsmp'], default='unet',
                        help='model name to be trained (model will be selected according to dataset name)'
                             'dssp: (cassi + hqs)'
                             'dgsmp: (cassi + bayesian gsm)')

    # hyper parameters

    parser.add_argument('--init-epoch', default=0, type=int,
                        help='initial epoch')
    parser.add_argument('--max-epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
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
    # args.bands = params[0]
    # args.num_samples = int(params[1] * args.num_samples)

    # args.batch_size = params[0]
    # args.init_lr = params[1]

    if args.model == 'dssp':
        args.stride = 1
    else:
        args.stride = 2

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # path configurations
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    save_path = f'{args.result_path}/{args.save_path}_{args.model}_{args.dataset}_ps{args.patch_size}' \
                f'_bands{args.bands}_emb_full_DA_{args.num_samples}_reg_param_gan_{reg_param}_reg_param_ae_{reg_param_ae}_real{args.real_samples}_syn{args.syn_samples}' \
                f'_lr{args.init_lr}_bs{args.batch_size}_epochs{args.max_epochs}_explr0.99'

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
    model = RGB2SPEC(args.model, args.input_shape, args.init_lr, save_path=save_path, device=device)

    # summary model

    num_parameters = sum([l.nelement() for l in model.rgb2spec_model.parameters()])
    print(f'Number of parameters: {num_parameters}')

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # load checkpoint
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    if args.load_path:
        model.load_checkpoint(args.load_path)
        print('¡Model checkpoint loaded correctly!')
        raise "Por aquí no es xd"

    else:
        pass  # raise ValueError('No checkpoint path provided')

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # train
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    model.train(train_loader, args.init_epoch, args.max_epochs, val_loader=val_loader)
    model.save_checkpoint(checkpoint_path)



if __name__ == '__main__':
    reg_param_aes = [0.0]
    reg_params = [0.0, 1e-5, 1e-3]


    for reg_param in reg_params:
        for reg_param_ae in reg_param_aes:
            main(reg_param_ae,reg_param)

    # # bands = [1]
    # # num_samples = [1.0]
    #
    # batch_sizes = [8, 16, 32]
    # lrs = [1e-3, 5e-4, 1e-4]
    #
    # param_list = [batch_sizes, lrs]
    #
    # num_run = 0
    # for params in product(*param_list):
    #     main(num_run, params)
    #     num_run += 1
