


# def train():
#     wandb.init()
#     config = wandb.config

#     num_samples = 20
#     sample_noise = utils.generator_inputs(num_samples, config, types=np.array([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]))
#     samples = [[] for _ in range(num_samples)]

#     discriminator = create_discriminator(config)
#     generator = create_generator(config)

#     joint_model = utils.create_joint_model(generator, discriminator)
    
#     # generator.summary()
#     # discriminator.summary()
#     # joint_model.summary()
#     log_functions.log_model_images(generator)
#     log_functions.log_model_images(discriminator)
#     log_functions.log_model_images(joint_model)

#     for i in range(config.adversarial_epochs):
#         print('=====================================================================')
#         print('Adversarian Epoch: {}/{}'.format(i+1, config.adversarial_epochs))
#         print('=====================================================================')
#         utils.train_discriminator(generator, discriminator, x_train, y_train, x_test, y_test, config)
#         utils.train_generator(generator, discriminator, joint_model, config)
#         log_functions.sample_images(generator, sample_noise, samples)