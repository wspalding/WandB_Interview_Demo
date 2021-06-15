import wandb
import numpy as np
import os
from wandb.keras import WandbCallback

from tensorflow.keras.datasets import fashion_mnist, mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives

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
sample_types = np.array([0,1,2,3,4,0,1,2,3,4,5,6,7,8,9,5,6,7,8,9])


def train(config=None):
    wandb.init(config=config)
    config = wandb.config

    wandb_logger = log_functions.WandbLogger()

    num_samples = 20
    sample_noise = utils.generator_inputs(num_samples, config, types=sample_types)
    samples = [[] for _ in range(num_samples)]

    discriminator = create_discriminator(config)
    discriminator_optimizer = Adam(config.discriminator_learning_rate, beta_1=config.discriminator_learning_rate_decay)
    discriminator.compile(optimizer=discriminator_optimizer, loss=BinaryCrossentropy(),
        metrics=['acc'])

    generator = create_generator(config)
    generator_optimizer = Adam(config.generator_learning_rate, beta_1=config.generator_learning_rate_decay)
    generator.compile(loss=BinaryCrossentropy(), optimizer=generator_optimizer)

    joint_model = utils.create_joint_model(generator, discriminator)
    joint_model.compile(optimizer='adam', loss=BinaryCrossentropy(),
        metrics=['acc'])

    
    wandb_logger.log_model_images(generator)
    wandb_logger.log_model_images(discriminator)
    wandb_logger.log_model_images(joint_model)

    for i in range(config.adversarial_epochs):
        print('=====================================================================')
        print('Adversarian Epoch: {}/{}'.format(i+1, config.adversarial_epochs))
        print('=====================================================================')
        utils.train_discriminator(generator, discriminator, x_train, y_train, x_test, y_test, config, wandb_logger)
        utils.train_generator(generator, discriminator, joint_model, config, wandb_logger)
        wandb_logger.sample_images(generator, sample_noise, samples)
        wandb_logger.push_logs()

if __name__ == '__main__':
    config = {
        'image_shape': (28, 28, 1),
        'generator_seed_dim': 100,
        'adversarial_epochs': 50,
        'discriminator_examples': 60000,
        'generator_examples': 60000,
        'generator_epochs': 1,
        'discriminator_epochs': 1,
        'batch_size': 128,
        'generator_learning_rate': 1e-4,
        'discriminator_learning_rate': 1e-4,
        'generator_learning_rate_decay': 0.9,
        'discriminator_learning_rate_decay': 0.9
    }
    train(config)