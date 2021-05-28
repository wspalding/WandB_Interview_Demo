from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, \
    Conv2DTranspose, Reshape, AveragePooling2D, UpSampling2D, LeakyReLU
from tensorflow.keras import initializers



def create_discriminator(config):

    discriminator = Sequential()
    discriminator.add(Flatten(input_shape=config.image_shape))
    discriminator.add(Dense(1024, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(2, activation='sigmoid'))
    discriminator.compile(optimizer='sgd', loss='categorical_crossentropy',
        metrics=['acc'])
    return discriminator