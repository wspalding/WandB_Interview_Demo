import wandb
from train import train

# sweep_config = {
#     'method': 'random', #grid, random, bayes
#     'metric': {
#       'name': 'average total reward',
#       'goal': 'maximize'   
#     },
#     'parameters': {
#         'learning_rate': {
#             # 'values': [0.01]
#             'min': 0.001,
#             'max': 0.1
#         },
#         'epochs': {
#             'values': [5000]
#         },
#         'batch_size': {
#             'values': [64, 128]
#         },
#         'training_epochs': {
#             'values': [1]
#         },
#         'loss_function': {
#             'values': ['mse', 'huber']
#         },
#         'optimizer': {
#             'values': ['adam', 'sgd']
#         },
#         'frame_skipping': {
#             # 'values': [1, 4, 10]
#             'values': [1]
#         },
#     },
#     'early_terminate': {
#         'type': 'hyperband',
#         's': 2,
#         'eta': 3,
#         'max_iter': 27
#     }
  
# }



if(__name__ == '__main__'):
    sweep_id = wandb.sweep(sweep_config, 
                            project="")

    wandb.agent(sweep_id, train)