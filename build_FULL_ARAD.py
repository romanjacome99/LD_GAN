"""A script to chop a set of datas and disparities into the h5 format.
Optionally converts multispectral datas to RGB.

Note that here only the central disparity view is saved as a label.

You may need to install the plenpy library to handle the datas.

    pip install plenpy

See: https://gitlab.com/iiit-public/plenpy
"""

import random
from pathlib import Path
from itertools import product

import h5py
import numpy as np

import mat73
import scipy.io as sio

###############################################################################
# SCRIPT SETUP
# Update to your needs
###############################################################################

# Set path to original data data folder
data_path = Path(r'Resize_Train_spectral_256')
rgb_data_path = Path(r'Resize_Train_RGB_256')

# data_path = Path(r'C:\Users\EMMANUELMARTINEZ\Documents\Datasets\RAW\ARAD\Resize_Valid_spectral')
# rgb_data_path = Path(r'C:\Users\EMMANUELMARTINEZ\Documents\Datasets\RAW\ARAD\Resize_Valid_RGB')

# Path and name of the target .h5 dataset
dataset_path = Path(r'dataset_da')

# # DA
#
# is_rot = False
# is_vflip = False
# is_hflip = True

for iter, (is_rot, is_vflip, is_hflip) in enumerate(product([False, True], [False, True], [False, True])):

    if iter == 0:
        continue

    print(is_rot, is_vflip, is_hflip)

    DA_name = ''

    if is_rot:
        DA_name += '_rot'

    if is_vflip:
        DA_name += '_vflip'

    if is_hflip:
        DA_name += '_hflip'

    initial_dataset_path = dataset_path / f'train_full_128x128x31_DA_{DA_name}.h5'

    # If output .h5 file already exists, overwrite?
    overwrite = True

    # Format of input datas
    input_shape = (256, 256, 31)
    # input_shape = (64, 64, 31)
    w, h, num_ch = input_shape
    num_ch_rgb = 3

    # Format of chopped data
    # Number of channels is unchanged or converted to RGB
    output_shape = (128, 128)
    (w_c, h_c) = output_shape

    # Whether to convert multispectral to RGB via CIE standard
    conv_to_RGB = False

    ###############################################################################
    # SCRIPT START
    ###############################################################################

    if not all(x <= y for x, y in zip(output_shape, input_shape)):
        raise ValueError(f"Incompatible output shape {output_shape} "
                         f"and input shape {input_shape}.")

    if not conv_to_RGB:
        num_ch_c = num_ch
    else:
        num_ch_c = 3

    # Create dataset folder if it does not exist
    dataset_path.mkdir(exist_ok=True)
    if initial_dataset_path.exists() and not overwrite:
        raise FileExistsError(f"Datasetfile {initial_dataset_path} already exists. "
                              f"Either delete, rename or specify overwrite = True.")

    # Calculate number of data patches per data
    w_range, h_range = w // w_c, h // h_c
    num_patches_per_data = w_range * h_range

    if conv_to_RGB:
        print(f"Converting Spectrum to RGB.")

    # Get all the datas contained in the path.
    data_files = [x for x in data_path.glob('*.mat')]
    rgb_data_files = [x for x in rgb_data_path.glob('*.mat')]

    # When no datas have been found, raise error
    if not data_files:
        raise ValueError(f"Folder {data_path} does not contain any datas in the MAT format.")

    if not rgb_data_files:
        raise ValueError(f"Folder {rgb_data_files} does not contain any datas in the MAT format.")

    # Number of files found
    num_data_files = len(data_files)
    num_patches = num_data_files * num_patches_per_data

    # Print information to data usage
    print(f"{num_patches_per_data} data patches will be generated per single data.")
    print(f"There are {num_data_files} files in the given path.")
    print(f"{num_patches} patches will be created in total.")

    # DA

    augment_samples = is_rot + is_hflip + is_vflip + 1
    num_patches = augment_samples * num_patches

    # Counter for the scenes
    patch_counter = 0

    # Open the dataset_path and create the initial full dataset
    # (Do not grow .h5 file since it leads to chunking which slows down reading)
    with h5py.File(initial_dataset_path, 'w') as d:
        # Init data, save as uint16
        data = d.create_dataset('spec', (num_patches, w_c, h_c, num_ch_c), dtype=np.uint16)
        rgb_data = d.create_dataset('rgb', (num_patches, w_c, h_c, num_ch_rgb), dtype=np.uint16)

        # Load files and patch
        counter = 0
        for i, (file, rgb_file) in enumerate(zip(data_files, rgb_data_files)):
            print(f"Processing data {i + 1} of {len(data_files)}.")

            # Load the multispectral data

            try:
                spectral_image = sio.loadmat(str(file))['cube']
                rgb_image = sio.loadmat(str(rgb_file))['img']
            except:
                spectral_image = mat73.loadmat(str(file))['cube']
                rgb_image = mat73.loadmat(str(rgb_file))['img']

            # data augmentation

            if is_rot:
                rotTimes = random.randint(1, 3)
                spec_rot = spectral_image
                rgb_rot = rgb_image

                for j in range(rotTimes):
                    spec_rot = np.rot90(spec_rot)
                    rgb_rot = np.rot90(rgb_rot)

                # Convert the data to uint16 ndarray
                spec_rot = ((2 ** 16 - 1) * np.asarray(spec_rot)).astype(np.uint16)
                rgb_rot = ((2 ** 16 - 1) * np.asarray(rgb_rot)).astype(np.uint16)

            if is_vflip:
                spec_vflip = spectral_image[::-1, :, :].copy()
                rgb_vflip = rgb_image[::-1, :, :].copy()

                # Convert the data to uint16 ndarray
                spec_vflip = ((2 ** 16 - 1) * np.asarray(spec_vflip)).astype(np.uint16)
                rgb_vflip = ((2 ** 16 - 1) * np.asarray(rgb_vflip)).astype(np.uint16)

            if is_hflip:
                # horizontal Flip
                spec_hflip = spectral_image[:, ::-1, :].copy()
                rgb_hflip = rgb_image[:, ::-1, :].copy()

                # Convert the data to uint16 ndarray
                spec_hflip = ((2 ** 16 - 1) * np.asarray(spec_hflip)).astype(np.uint16)
                rgb_hflip = ((2 ** 16 - 1) * np.asarray(rgb_hflip)).astype(np.uint16)

            # # Convert the data to uint16 ndarray
            spectral_image = ((2 ** 16 - 1) * np.asarray(spectral_image)).astype(np.uint16)
            rgb_image = ((2 ** 16 - 1) * np.asarray(rgb_image)).astype(np.uint16)

            # Create patches from current data
            for w_iter, h_iter in product(range(w_range), range(h_range)):
                # Calculate start and end values
                w_min = w_iter * w_c
                w_max = (w_iter + 1) * h_c

                h_min = h_iter * h_c
                h_max = (h_iter + 1) * h_c

                # Extract data patch
                data[counter] = spectral_image[w_min:w_max, h_min:h_max, :]
                rgb_data[counter] = rgb_image[w_min:w_max, h_min:h_max, :]

                counter += 1

                if is_rot:
                    data[counter] = spec_rot[w_min:w_max, h_min:h_max, :]
                    rgb_data[counter] = rgb_rot[w_min:w_max, h_min:h_max, :]

                    counter += 1

                if is_vflip:
                    data[counter] = spec_vflip[w_min:w_max, h_min:h_max, :]
                    rgb_data[counter] = rgb_vflip[w_min:w_max, h_min:h_max, :]

                    counter += 1

                if is_hflip:
                    data[counter] = spec_hflip[w_min:w_max, h_min:h_max, :]
                    rgb_data[counter] = rgb_hflip[w_min:w_max, h_min:h_max, :]

                    counter += 1

                # Print progress
                print(f"Processed {counter} patches.")

        if counter == num_patches:
            print(f"Processed {DA_name} dataset")
        else:
            raise "The number of patches is not the same that the number of patches in the dataset," \
                  f"please check the code for {DA_name}"

# Done
print('The data was patched.')
