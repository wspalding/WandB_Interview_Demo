import wandb
import numpy as np

def log_generator(epoch, logs):
    wandb.log({'generator_loss': logs['loss'],
                     'generator_acc': logs['acc'],
                     'discriminator_loss': 0.0,
                     'discriminator_acc': (1-logs['acc'])/2.0+0.5})

def log_discriminator(epoch, logs):
    wandb.log({
            'generator_loss': 0.0,
            'generator_acc': (1.0-logs['acc'])*2.0,
            'discriminator_loss': logs['loss'],
            'discriminator_acc': logs['acc']})

def sample_images(generator, noise, samples):
    gen_imgs = generator.predict(noise)
    for i, s in enumerate(samples):
        s.append(np.reshape(gen_imgs[i], [1, 28, 28]) * 255.0)
    wandb.log({
        'examples': [wandb.Image(np.squeeze(i)) for i in gen_imgs],
        'progession': [wandb.Video(np.array(s)) for s in samples]
        })

