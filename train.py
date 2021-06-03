import wandb
import numpy as np
import os
from wandb.keras import WandbCallback

from tensorflow.keras.datasets import fashion_mnist
from wandb.util import generate_id

import utils
import log_functions
from discriminator import create_discriminator
from generator import create_generator

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# re-scale dataset to be bounded between -1.0 -> 1.0
x_train = x_train / 255.0 * 2.0 - 1.0
x_test = x_test / 255.0 * 2.0 - 1.0
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print(x_train.shape, y_train.shape)



def train():
    wandb.init()
    config = wandb.config

    sample_noise = utils.generator_inputs(10, config)
    samples = [[] for _ in range(10)]

    discriminator = create_discriminator(config)
    generator = create_generator(config)

    joint_model = utils.create_joint_model(generator, discriminator)
    
    # generator.summary()
    # discriminator.summary()
    # joint_model.summary()
    log_functions.log_model_images(generator)
    log_functions.log_model_images(discriminator)
    log_functions.log_model_images(joint_model)

    for i in range(config.adversarial_epochs):
        print('=====================================================================')
        print('Adversarian Epoch: {}/{}'.format(i+1, config.adversarial_epochs))
        print('=====================================================================')
        utils.train_discriminator(generator, discriminator, x_train, y_train, x_test, y_test, config)
        utils.train_generator(generator, discriminator, joint_model, config)
        log_functions.sample_images(generator, sample_noise, samples)