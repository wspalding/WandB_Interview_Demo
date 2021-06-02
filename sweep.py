import wandb
from train import train

sweep_config = {
    'method': 'random', #grid, random, bayes
    'metric': {
      'name': 'generator_acc',
      'goal': 'maximize'   
    },
    'parameters': {
            'image_shape': {
                'value': (28, 28, 1)
            },
            'generator_seed_dim': {
                'distribution': 'int_uniform',
                'min': 5,
                'max': 20
            },
            'adversarial_epochs': {
                'distribution': 'int_uniform',
                'min': 25,
                'max': 100
            },
            'discriminator_examples': {
                'value': 60000
            },
            'generator_examples': {
                'value': 60000
            },
            'generator_epochs': {
                'value': 1
            },
            'discriminator_epochs': {
                'value': 1
            },
            'batch_size': {
                'values': [64, 128, 256]
            },
            'generator_learning_rate': {
                'distribution': 'log_uniform',
                'min': -10,
                'max': -7
            },
            'discriminator_learning_rate': {
                'distribution': 'log_uniform',
                'min': -10,
                'max': -7
            }
    },
    'early_terminate': {
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 27
    }
}



if(__name__ == '__main__'):
    sweep_id = wandb.sweep(sweep_config, 
                            project="")

    wandb.agent(sweep_id, train)