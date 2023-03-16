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
    parser.add_argument('--patch-size', default=128, type=int,
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

    parser.add_argument('--reg_param_ae',default=1e-3,type=float)

    parser.add_argument('--reg_param', default=1e-3, type=float,
                        help='initial learning rate')
    # gpu config
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
    #checkpoint_path = f'{save_path}/checkpoints'
    #Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    for reg_param_ae, reg_param_gan in [[0.001,0.001]]:
        # for reg_param_ae in [0.001]:
            args.load_path = f'results/embedded_gan_arad_bs8_lr0.0002_epochs50_max_var_reg_param_{reg_param_gan}_reg_param_ae{reg_param_ae}/checkpoints'
            save_config(save_path, os.path.basename(__file__), args)  # Save the experiment config in a .txt file

            # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
            # load dataset
            # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

            #dataset = Dataset(args.dataset_path, args.batch_size, args.patch_size, args.workers)
            #train_loader, val_loader = dataset.get_arad_dataset()

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

            # # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
            # # test
            # # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
            #
            # test_metrics = model.test_step(val_loader[0], args.max_epochs - 1, args.max_epochs)
            #
            # print('Test metrics:')
            # for key, value in test_metrics.items():
            #     print(f'{key}: {value}')
            #
            # model.save_images(val_loader[0], args.max_epochs, save=True, show=True)

            # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
            # create dataset
            # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

            # num_patches = 57536 // 2
            num_patches = 57536
            # num_patches = 57536 * 2

            dataset_path = f'generated_data/gen_reg_max_var_param_gan_{reg_param_gan}_reg_param_ae{reg_param_ae}'
            Path(dataset_path).mkdir(parents=True, exist_ok=True)
            dataset_path = Path(dataset_path)
            print(f'Generating dataset with reg gan param {reg_param_gan} and reg ae param {reg_param_ae}')
            for num_patches in [3596]:
                initial_dataset_path = dataset_path / f'gen_emb_train_128x128x31_num_samples_{num_patches}.h5'
                


                with h5py.File(initial_dataset_path, 'w') as d:
                    # Init data, save as uint16
                    data = d.create_dataset('spec', (num_patches, args.patch_size, args.patch_size, args.num_ch), dtype=np.uint16)
                    embed = d.create_dataset('embd', (num_patches, args.patch_size, args.patch_size, 3), dtype=np.uint16)

                    rgb_data = d.create_dataset('rgb', (num_patches, args.patch_size, args.patch_size, 3), dtype=np.uint16)

                    model.generator.eval()
                    model.decoder.eval()

                    torch.manual_seed(0)

                    for i in range(num_patches):
                        noise = torch.randn(1, 100, 1, 1).to(device, non_blocking=torch.cuda.is_available())

                        generated_embedded = model.generator(noise)
                        generated_spec = model.decoder(generated_embedded)
                        generated_rgb = model.cs.spec_to_rgb_torch(generated_spec)
                        generated_embedded = np.clip(generated_embedded.permute(0, 2, 3, 1).cpu().detach().numpy().squeeze(), 0, 1)
                        generated_spec = np.clip(generated_spec.permute(0, 2, 3, 1).cpu().detach().numpy().squeeze(), 0, 1)
                        generated_rgb = np.clip(generated_rgb.permute(0, 2, 3, 1).cpu().detach().numpy().squeeze(), 0, 1)
                        generated_embedded = ((2 ** 16 - 1) * np.asarray(generated_embedded)).astype(np.uint16)
                        # # Convert the data to uint16 ndarray
                        generated_spec = ((2 ** 16 - 1) * np.asarray(generated_spec)).astype(np.uint16)
                        generated_rgb = ((2 ** 16 - 1) * np.asarray(generated_rgb)).astype(np.uint16)
                        embed[i] = generated_embedded
                        data[i] = generated_spec
                        rgb_data[i] = generated_rgb
                        generated_embedded
                        # Print progress
                        print(f"Processed {i} samples.")

    print('Fin')


if __name__ == '__main__':
    main()
