
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, \
    Conv2DTranspose, Reshape, AveragePooling2D, UpSampling2D, LeakyReLU
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import activations


def create_discriminator(config):

    # discriminator = Sequential()
    # discriminator.add(Flatten(input_shape=config.image_shape))
    # discriminator.add(Dense(1024, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    # discriminator.add(LeakyReLU(0.2))
    # discriminator.add(Dropout(0.3))
    # discriminator.add(Dense(512))
    # discriminator.add(LeakyReLU(0.2))
    # discriminator.add(Dropout(0.3))
    # discriminator.add(Dense(256))
    # discriminator.add(LeakyReLU(0.2))
    # discriminator.add(Dropout(0.3))
    # discriminator.add(Dense(2, activation='sigmoid'))
    # discriminator.compile(optimizer='sgd', loss='categorical_crossentropy',
    #     metrics=['acc'])
    # return discriminator

    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))

    discriminator_optimizer = Adam(config.discriminator_learning_rate)

    model.compile(optimizer=discriminator_optimizer, loss='categorical_crossentropy',
        metrics=['acc'])

    return model

# def discriminator_loss(real_output, fake_output):
#     cross_entropy = BinaryCrossentropy(from_logits=True)
#     real_loss = cross_entropy(tf.ones_like(real_output), real_output)
#     fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
#     total_loss = real_loss + fake_loss
#     return total_loss