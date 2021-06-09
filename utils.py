import numpy as np
import os
from numpy import random
import wandb
from wandb.keras import WandbCallback

import tensorflow as tf
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.utils import to_categorical

import log_functions
from tensorflow.keras.models import Sequential, Model

def create_joint_model(generator, discriminator):
    # joint_model = Sequential()
    # joint_model.add(generator)
    # joint_model.add(discriminator)

    discriminator.trainable = False

    noise_input, embedding_input = generator.input
    image_output = generator.output
    joint_model_output = discriminator([image_output, embedding_input])

    joint_model = Model([noise_input, embedding_input], joint_model_output, name="joint_model")

    joint_model.compile(optimizer='adam', loss='categorical_crossentropy',
        metrics=['acc'])

    return joint_model


def train_discriminator(generator, discriminator, x_train, y_train, x_test, y_test, config, logger):

    train, train_labels = mix_data(x_train, y_train, generator, length=config.discriminator_examples, seed_dim=config.generator_seed_dim)
    test, test_labels = mix_data(x_test, y_test, generator, length=x_test.shape[0], seed_dim=config.generator_seed_dim)

    discriminator.trainable = True
    # discriminator.summary()

    wandb_logging_callback = LambdaCallback(on_epoch_end=logger.log_discriminator)

    history = discriminator.fit(train, train_labels,
        epochs=config.discriminator_epochs,
        batch_size=config.batch_size, validation_data=(test, test_labels),
        callbacks = [wandb_logging_callback])

    discriminator.save(os.path.join(wandb.run.dir, "discriminator.h5"))


def train_generator(generator, discriminator, joint_model, config, logger):
    num_examples = config.generator_examples

    train = generator_inputs(num_examples, config)
    labels = to_categorical(np.ones(num_examples))

    add_noise(labels)

    wandb_logging_callback = LambdaCallback(on_epoch_end=logger.log_generator)

    discriminator.trainable = False

    # joint_model.summary()

    joint_model.fit(train, labels, epochs=config.generator_epochs,
            batch_size=config.batch_size,
            callbacks=[wandb_logging_callback])

    generator.save(os.path.join(wandb.run.dir, "generator.h5"))



def generator_inputs(num_examples, config, **kwargs):
    types = kwargs.get('types', np.random.randint(0, 10, size=(num_examples, 1)))
    assert(len(types) == num_examples)
    return [np.random.normal(0, 1, (num_examples, config.generator_seed_dim)), types]

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

def mix_data(x, y, generator, length=1000, seed_dim=10):
    num_examples=int(length/2)

    offset = np.random.randint(0,x.shape[0]-num_examples)
    x = x[offset:num_examples+offset, :, :]
    y = y[offset:num_examples+offset]

    seeds = np.random.normal(0, 1, (num_examples, seed_dim))
    fake_y = np.random.randint(0, 10, size=(num_examples,))

    fake_train = generator.predict([seeds, fake_y])[:,:,:,0]

    combined_images  = np.concatenate([x, fake_train])
    combined_y = np.concatenate([y, fake_y])

    # combine them together
    labels = np.zeros(combined_images.shape[0])
    labels[:x.shape[0]] = 1

    indices = np.arange(combined_images.shape[0])
    np.random.shuffle(indices)
    combined_images = combined_images[indices]
    combined_y = combined_y[indices]
    labels = labels[indices]

    combined_images.shape += (1,)

    labels = to_categorical(labels)

    add_noise(labels)

    return ((combined_images, combined_y), labels)

# @tf.function
# def train_step(images):
#     noise = tf.random.normal([BATCH_SIZE, noise_dim])

#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#       generated_images = generator(noise, training=True)

#       real_output = discriminator(images, training=True)
#       fake_output = discriminator(generated_images, training=True)

#       gen_loss = generator_loss(fake_output)
#       disc_loss = discriminator_loss(real_output, fake_output)

#     gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
#     gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

#     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
#     discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))