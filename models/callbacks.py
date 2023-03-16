import io

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback

from scripts.functions import psnr


class BSCallback(Callback):
    def __init__(self, optical_encoder, train_batch_size, test_batch_size=5):
        super(BSCallback, self).__init__()
        self.optical_encoder = optical_encoder
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    def on_epoch_begin(self, epoch, logs=None):
        self.optical_encoder.layers[-1].set_batch_size(self.train_batch_size)

    def on_test_begin(self, logs=None):
        self.optical_encoder.layers[-1].set_batch_size(self.test_batch_size)


class UpdateBinaryParameter(Callback):
    def __init__(self, optical_encoder, p_aum, p_step):
        super().__init__()
        self.optical_encoder = optical_encoder
        self.p_aum = p_aum
        self.p_step = p_step
        self.optical_encoder = optical_encoder

    def on_epoch_end(self, epoch, logs=None):
        parameter = tf.keras.backend.get_value(self.optical_encoder.layers[-1].parameter)
        print('\nRegularizator: ' + str(parameter))

        if epoch % self.p_step == 0 and epoch > 50:
            new_parameter = parameter * self.p_aum
            self.optical_encoder.layers[-1].set_regularizer_parameter(new_parameter)
            print('\nRegularizator updated to ' + str(new_parameter))


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


class CASSIUnetLCallback(Callback):

    def __init__(self, optical_encoder, writer):
        super(CASSIUnetLCallback, self).__init__()
        self.optical_encoder = optical_encoder
        self.writer = writer

    def on_epoch_end(self, epoch, logs=None):
        if int(epoch + 1) % 10 == 0:
            with self.writer.as_default():
                H = self.optical_encoder.layers[-1].H[tf.newaxis, :32, :32, :1]
                tf.summary.image('Coded Aperture', H, step=epoch)


class CodedApertureCallback(Callback):

    def __init__(self, writer, sensing_name):
        super(CodedApertureCallback, self).__init__()
        self.writer = writer
        self.sensing_name = sensing_name

    def on_train_begin(self, logs=None):
        # verify the model contains CA
        _ = self.get_sensing_model()

    def get_sensing_model(self):
        try:
            sensing_model = self.model.get_layer(self.sensing_name)
            return sensing_model

        except:
            raise 'sensing was not in the model, please check the code implementation'

    def on_epoch_end(self, epoch, logs=None):
        if int(epoch + 1) % 50 == 0:
            sensing_model = self.get_sensing_model()

            with self.writer.as_default():
                H = tf.reshape(sensing_model.H, (*sensing_model.H.shape[:2], -1))[..., 0][tf.newaxis, ..., tf.newaxis]
                tf.summary.image('Coded Aperture', H, step=epoch)


class HSICallback(Callback):

    def __init__(self, writer, network_name, scene_rgb, x0=None, y0=None, spectral_scene=None):
        super(HSICallback, self).__init__()
        self.writer = writer
        self.network_name = network_name
        self.scene_rgb = scene_rgb
        self.x0 = x0
        self.y0 = y0
        self.spectral_scene = spectral_scene

    def on_train_begin(self, logs=None):
        # verify the model contains CA
        _ = self.get_network()

    def get_network(self):
        try:
            output_model = self.model.get_layer(self.network_name)
            network = Model(self.model.input, output_model.output)
            return network

        except:
            raise 'network was not in the model, please check the code implementation'

    def on_epoch_end(self, epoch, logs=None):

        if int(epoch + 1) % 50 == 0:
            network = self.get_network()
            x0 = [self.x0, self.y0] if self.network_name == 'hir_dssp' else self.x0
            reconstructed_hsi = network(x0)

            fig, axs = plt.subplots(1, 2, figsize=(16, 8))

            axs[0].imshow(self.scene_rgb[0])
            axs[0].set_title('Original')
            axs[0].axis('off')

            axs[1].imshow(reconstructed_hsi.numpy()[0][..., (25, 22, 11)])
            axs[1].set_title(f'Reconstructed image - PSNR: {psnr(self.spectral_scene[0], reconstructed_hsi[0]):4f}')
            axs[1].axis('off')

            with self.writer.as_default():
                tf.summary.image('Reconstruction', plot_to_image(fig), step=epoch)

        plt.close('all')
