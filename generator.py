
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, \
    Conv2DTranspose, Reshape, AveragePooling2D, UpSampling2D, LeakyReLU, \
         BatchNormalization, Embedding, Concatenate, Input, Reshape
from tensorflow.keras import initializers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import optimizers


def create_generator(config):
    random_dim = config.generator_seed_dim

    noise_input = Input(shape=(random_dim,), name='noise input')

    embedding_input = Input(shape=(1,), name='type input')
    embedding = Embedding(10, random_dim)(embedding_input)
    embedding = Reshape((random_dim,))(embedding)
    concat = Concatenate()([noise_input, embedding])
    l1 = Dense(7*7*256, use_bias=False, input_shape=(random_dim,))(concat)
    l1 = BatchNormalization()(l1)
    l1 = LeakyReLU()(l1)

    reshape = Reshape((7, 7, 256))(l1)

    c1 = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(reshape)
    c1 = BatchNormalization()(c1)
    c1 = LeakyReLU()(c1)

    c2 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(c1)
    c2 = BatchNormalization()(c2)
    c2 = LeakyReLU()(c2)

    c3 = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh', name="generator_output")(c2)

    model = Model([noise_input, embedding_input], c3, name='generator')

    return model


def generator_loss(fake_output):
    cross_entropy = BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)