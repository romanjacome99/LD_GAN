# LD-GAN: Low-Dimensional Generative Adversarial Network for Spectral Image Generation with Variance Regularization

## Abstract

Deep learning methods are state-of-the-art for spectral image (SI) computational tasks. However, these methods are constrained in their performance since SI datasets are limited due to the highly expensive and long acquisition time. Usually, data augmentation techniques are employed to mitigate the lack of data. Surpassing classical augmentation methods, such as geometric transformation, GANs enable diverse augmentation by learning and sampling from the data distribution. Nevertheless, GAN-based SI generation is challenging since the high-dimensionality nature of this kind of data hinders the convergence of the GAN training yielding suboptimal generation. To surmount this limitation, we propose low-dimensional GAN (LD-GAN), where we train the GAN employing a low-dimensional representation of the SI with the latent space of an autoencoder network (AE). Thus, we generate new low-dimensional samples which then are mapped to the SI dimension with the pre-trained decoder network. Besides, we propose a statistical regularization to control the low-dimensional representation variance for the AE training and to achieve high diversity of samples generated with the GAN. We validate our SI generation method as data augmentation for compressive SI, SI spatial-spectral super-resolution, and RBG to spectral tasks with improvements varying from 0.5 to 1 [dB] in each task respectively with the proposed method compared against the non-data augmentation training, traditional DA, and with a GAN trained to generate the full spectral image.


## Dataset

The employed dataset in this work can be downloaded in:

https://drive.google.com/file/d/1FQBfDd248dCKClR-BpX5V2drSbeyhKcq/view
https://drive.google.com/file/d/1A4GUXhVc5k5d_79gNvokEtVPG290qVkd/view
https://drive.google.com/file/d/12QY8LHab3gzljZc3V6UyHgBee48wh9un/view
https://drive.google.com/file/d/19vBR_8Il1qcaEZsK42aGfvg5lCuvLh1A/view 

## Training Autoencoder

Learn the LD representation of the SI dataset with ´main_autoencoder.py´
