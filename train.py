import wandb
import numpy as np
import os
from wandb.keras import WandbCallback

from tensorflow.keras.datasets import fashion_mnist
from wandb.util import generate_id

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

    discriminator = create_discriminator(config)
    generator = create_generator(config)

    