from numpy.lib.npyio import save
from tensorflow.keras import models
import wandb
import numpy as np
from tensorflow.keras.utils import plot_model

class WandbLogger():
    def __init__(self) -> None:
        self.logs = {}

    def push_logs(self):
        wandb.log(self.logs)
        self.logs = {}

    def log_generator(self, epoch, logs):
        for key, value in logs.items():
            self.logs['generator_{}'.format(key)] = value
        # self.logs['generator_loss'] = logs['loss']
        # self.logs['generator_acc'] = logs['acc']
        # self.logs['generator_TP'] = logs['TP']
        # self.logs['generator_TN'] = logs['TN']
        # self.logs['generator_FP'] = logs['FP']
        # self.logs['generator_FN'] = logs['FN']

    def log_discriminator(self, epoch, logs):
        for key, value in logs.items():
            self.logs['discriminator_{}'.format(key)] = value
        # self.logs['discriminator_loss'] = logs['loss']
        # self.logs['discriminator_acc'] = logs['acc']
        # self.logs['discrininator_TP'] = logs['TP']
        # self.logs['discrininator_TN'] = logs['TN']
        # self.logs['discrininator_FP'] = logs['FP']
        # self.logs['discrininator_FN'] = logs['FN']

    def sample_images(self, generator, noise, samples):
        gen_imgs = generator.predict(noise)
        for i, s in enumerate(samples):
            s.append(np.reshape(gen_imgs[i], [1, 28, 28]) * 255.0)
        self.logs['examples'] = [wandb.Image(np.squeeze(i)) for i in gen_imgs]
        self.logs['progession'] = [wandb.Video(np.array(s)) for s in samples]

    def log_model_images(self, model):
        save_file = "{}.png".format(model.name)
        plot_model(model, to_file=save_file)
        self.logs['{} architecture'.format(model.name)] = wandb.Image(save_file)