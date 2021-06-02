import imp

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, \
    Conv2DTranspose, Reshape, AveragePooling2D, UpSampling2D, LeakyReLU, BatchNormalization
from tensorflow.keras import initializers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import optimizers


def create_generator(config):
    random_dim = config.generator_seed_dim

    # generator = Sequential()
    # generator.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    # generator.add(LeakyReLU(0.2))
    # generator.add(Dense(512))
    # generator.add(LeakyReLU(0.2))
    # generator.add(Dense(1024))
    # generator.add(LeakyReLU(0.2))
    # generator.add(Dense(784, activation='tanh'))
    # generator.add(Reshape(config.image_shape))
    # generator.compile(loss='categorical_crossentropy', optimizer='adam')

    # return generator

    model = Sequential()
    model.add(Dense(7*7*256, use_bias=False, input_shape=(random_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    generator_optimizer = Adam(config.generator_learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=generator_optimizer)

    return model

# def generator_loss(fake_output):
#     cross_entropy = BinaryCrossentropy(from_logits=True)
#     return cross_entropy(tf.ones_like(fake_output), fake_output)