import wandb
import numpy as np
import os
from wandb.keras import WandbCallback

from tensorflow.keras.datasets import fashion_mnist, mnist
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

    wandb_logger = log_functions.WandbLogger()

    num_samples = 20
    sample_noise = utils.generator_inputs(num_samples, config, types=np.array([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]))
    samples = [[] for _ in range(num_samples)]

    discriminator = create_discriminator(config)
    generator = create_generator(config)

    joint_model = utils.create_joint_model(generator, discriminator)
    
    wandb_logger.log_model_images(generator)
    wandb_logger.log_model_images(discriminator)
    wandb_logger.log_model_images(joint_model)

    for i in range(config.adversarial_epochs):
        print('=====================================================================')
        print('Adversarian Epoch: {}/{}'.format(i+1, config.adversarial_epochs))
        print('=====================================================================')
        utils.train_discriminator(generator, discriminator, x_train, y_train, x_test, y_test, config)
        utils.train_generator(generator, discriminator, joint_model, config)
        wandb_logger.sample_images(generator, sample_noise, samples)
        wandb_logger.push_logs()