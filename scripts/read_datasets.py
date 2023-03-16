from os.path import join
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from os import listdir
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os.path import isfile


class DataGenerator(Sequence):

    def __init__(self, list_IDs, flag_tr, D, im_path, dic_img,
                 batch_size=5, dim=(400, 400, 120), shuffle=True, augment=False):

        # Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.flag_tr = flag_tr
        self.D = D
        self.im_path = im_path
        self.dic_img = dic_img
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        # number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        if self.flag_tr:
            X, Y, W = self.__data_generation(list_IDs_temp)
            return X, Y, W
        else:
            X, Y = self.__data_generation(list_IDs_temp)
            return X, Y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty([self.batch_size, *self.dim])
        W = np.empty([self.batch_size])

        # Generate data
        count = 0

        for i, ID in enumerate(list_IDs_temp):
            img_X = sio.loadmat(join(self.im_path, ID))[self.dic_img]
            img_X = img_X / np.max(tf.reshape(img_X, -1))

            if self.augment:
                transform_params = self.D.get_random_transform(self.dim)
                img_X = self.D.apply_transform(x=np.array(img_X), transform_parameters=transform_params)
            img_X_p = tf.image.central_crop(img_X, self.dim[0] / img_X.shape[0])

            X[i,] = img_X_p

        if self.flag_tr:
            return X, X, W
        else:
            return X, X


def generate_dataset(B, M, L, train_path, val_path, test_path, dict, perc_data):
    D = ImageDataGenerator(rotation_range=180, width_shift_range=0.2, height_shift_range=0.2,
                           horizontal_flip=True)

    params_d = {'dim': (M, M, L),
                'batch_size': B,
                'im_path': train_path,
                'dic_img': dict,
                'shuffle': True,
                'augment': False}  # Augmented only for training

    HyperFiles = [fn for fn in listdir(train_path) if isfile(join(train_path, fn)) and fn.lower().endswith('.mat')]

    data = round(len(HyperFiles) * perc_data)
    a = np.random.permutation(data)
    HyperFiles = list(np.array(HyperFiles)[a.astype(int)])

    print('Training images: ', len(HyperFiles))

    flag_tr = False
    train_gen = DataGenerator(HyperFiles, flag_tr, D, **params_d)

    params_d = {'dim': (M, M, L),
                'batch_size': B,
                'im_path': val_path,
                'dic_img': dict,
                'shuffle': True,
                'augment': False}

    HyperFiles = [fn for fn in listdir(val_path) if isfile(join(val_path, fn)) and fn.lower().endswith('.mat')]

    print('Validation images: ', len(HyperFiles))

    flag_tr = False
    val_gen = DataGenerator(HyperFiles, flag_tr, D, **params_d)

    params_d = {'dim': (M, M, L),
                'batch_size': B,
                'im_path': test_path,
                'dic_img': dict,
                'shuffle': True,
                'augment': False}

    HyperFiles = [fn for fn in listdir(test_path) if isfile(join(test_path, fn)) and fn.lower().endswith('.mat')]

    print('Test images: ', len(HyperFiles))

    flag_tr = False
    test_gen = DataGenerator(HyperFiles, flag_tr, D, **params_d)

    return train_gen, val_gen, test_gen
