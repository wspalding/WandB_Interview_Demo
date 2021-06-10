
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, \
    Conv2DTranspose, Reshape, AveragePooling2D, UpSampling2D, LeakyReLU, \
         BatchNormalization, Embedding, Concatenate, Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import activations
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.ops.gen_math_ops import Mod


def create_discriminator(config):

    img_input = Input(shape=config.image_shape, name='image input')

    embedding_input = Input(shape=(1,), name='type input')
    embedding = Embedding(10, 28*28*1)(embedding_input)
    embedding = Reshape((28, 28, 1))(embedding)

    concat = Concatenate()([img_input, embedding])

    c1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(concat)
    c1 = LeakyReLU()(c1)
    c1 = Dropout(0.3)(c1)

    c2 = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(c1)
    c2 = LeakyReLU()(c2)
    c2 = Dropout(0.3)(c2)

    flatten = Flatten()(c2)
    output = Dense(2, activation='softmax', name="discriminator_output")(flatten)

    model = Model([img_input, embedding_input], output, name='discriminator')

    # discriminator_optimizer = Adam(config.discriminator_learning_rate, beta_1=config.discriminator_learning_rate_decay)
    # model.compile(optimizer=discriminator_optimizer, loss='categorical_crossentropy',
    #     metrics=['acc'])

    return model

