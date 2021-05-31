import numpy as np
import os
import wandb
from wandb.keras import WandbCallback

from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.utils import to_categorical

import log_functions
from tensorflow.keras.models import Sequential

def create_joint_model(generator, discriminator):
    joint_model = Sequential()
    joint_model.add(generator)
    joint_model.add(discriminator)

    discriminator.trainable = False

    joint_model.compile(optimizer='adam', loss='categorical_crossentropy',
        metrics=['acc'])

    return joint_model


def train_discriminator(generator, discriminator, x_train, x_test, config):

    train, train_labels = mix_data(x_train, generator, length=config.discriminator_examples, seed_dim=config.generator_seed_dim)
    test, test_labels = mix_data(x_test, generator, length=config.discriminator_examples, seed_dim=config.generator_seed_dim)

    discriminator.trainable = True
    discriminator.summary()

    wandb_logging_callback = LambdaCallback(on_epoch_end=log_functions.log_discriminator)

    history = discriminator.fit(train, train_labels,
        epochs=config.discriminator_epochs,
        batch_size=config.batch_size, validation_data=(test, test_labels),
        callbacks = [wandb_logging_callback])

    discriminator.save(os.path.join(wandb.run.dir, "discriminator.h5"))


def train_generator(generator, discriminator, joint_model, config):
    num_examples = config.generator_examples

    train = generator_inputs(num_examples, config)
    labels = to_categorical(np.ones(num_examples))

    add_noise(labels)

    wandb_logging_callback = LambdaCallback(on_epoch_end=log_functions.log_generator)

    discriminator.trainable = False

    joint_model.summary()

    joint_model.fit(train, labels, epochs=config.generator_epochs,
            batch_size=config.batch_size,
            callbacks=[wandb_logging_callback])

    generator.save(os.path.join(wandb.run.dir, "generator.h5"))


def sample_images(generator, config):
    noise = generator_inputs(10, config)
    gen_imgs = generator.predict(noise)
    wandb.log({'examples': [wandb.Image(np.squeeze(i)) for i in gen_imgs]})


def generator_inputs(num_examples, config):
    return np.random.normal(0, 1, (num_examples, config.generator_seed_dim))

def add_noise(labels):
    for label in labels:
        noise = np.random.uniform(0.0,0.3)
        if label[0] == 0.0:
            label[0]+= noise
            label[1]-=noise
        else:
            label[0]-=noise
            label[1]+=noise
        if np.random.uniform(0,1) > 0.05:
            tmp = label[0]
            label[0] = label[1]
            label[1] = tmp

def mix_data(data, generator, length=1000, seed_dim=10):
    num_examples=int(length/2)

    data= data[:num_examples, :, :]


    seeds = np.random.normal(0, 1, (num_examples, seed_dim))

    fake_train = generator.predict(seeds)[:,:,:,0]

    combined  = np.concatenate([ data, fake_train ])

    # combine them together
    labels = np.zeros(combined.shape[0])
    labels[:data.shape[0]] = 1

    indices = np.arange(combined.shape[0])
    np.random.shuffle(indices)
    combined = combined[indices]
    labels = labels[indices]
    combined.shape += (1,)

    labels = to_categorical(labels)

    add_noise(labels)

    return (combined, labels)