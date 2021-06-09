



# def train(config=None):
#     wandb.init(config=config)
#     config = wandb.config

#     generator = make_generator_model(config)
#     discriminator = make_discriminator_model(config)
#     generator_optimizer = tf.keras.optimizers.Adam(config.generator_learning_rate)
#     discriminator_optimizer = tf.keras.optimizers.Adam(config.discriminator_learning_rate)

#     logs = defaultdict(int)
#     acc = tf.keras.metrics.BinaryAccuracy()

#     @tf.function
#     def train_step(images):
#         types = np.random.randint(0,10, (config.batch_size,))
#         noise = tf.random.normal([config.batch_size, config.generator_seed_dim])
#         metrics = {}
        
#         with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#             generated_images = generator([noise, types], training=True)

#             real_output = discriminator(images, training=True)
#             fake_output = discriminator([generated_images, types], training=True)

#             gen_loss = generator_loss(fake_output)
#             disc_loss = discriminator_loss(real_output, fake_output)

#         gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
#         gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

#         generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
#         discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


#         metrics['gen_batch_loss'] = gen_loss
#         metrics['disc_batch_loss'] = disc_loss
#         acc.reset_states()
#         acc.update_state(tf.ones_like(fake_output), fake_output)
#         metrics['gen_batch_acc'] = acc.result()
#         acc.reset_states()
#         acc.update_state(tf.zeros_like(fake_output), fake_output)
#         acc.update_state(tf.ones_like(real_output), real_output)
#         metrics['disc_batch_acc'] = acc.result()

#         return metrics

#     # Batch and shuffle the data
#     dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(config.num_examples).batch(config.batch_size)
#     sample_noise = tf.random.normal([config.num_samples, config.generator_seed_dim])
#     sample_types = np.array([0,1,2,3,4,0,1,2,3,4,5,6,7,8,9,5,6,7,8,9])
#     samples = [[] for _ in range(config.num_samples)]

#     log_functions.log_model_images(generator)
#     log_functions.log_model_images(discriminator)

#     log_functions.sample_images(generator, [sample_noise, sample_types], samples)

#     for epoch in range(config.adversarial_epochs):
#         print('=====================================================================')
#         print('Adversarian Epoch: {}/{}'.format(epoch+1, config.adversarial_epochs))
#         print('=====================================================================')
        
#         start = time.time()
#         gen_loss = 0
#         disc_loss = 0
#         gen_acc = 0
#         disc_acc = 0
#         for i, image_batch in enumerate(dataset):
#             print('{}/{}'.format(i+1, len(dataset)), end='\r')
#             metrics = train_step(image_batch)
#             gen_loss += metrics['gen_batch_loss']
#             disc_loss += metrics['disc_batch_loss']
#             gen_acc += metrics['gen_batch_acc'].numpy()
#             disc_acc += metrics['disc_batch_acc'].numpy()

#         print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

#         logs['gen_loss'] = gen_loss / len(dataset)
#         logs['disc_loss'] = disc_loss / len(dataset)
#         logs['gen_acc'] = gen_acc / len(dataset)
#         logs['disc_acc'] = disc_acc / len(dataset)
#         log_functions.log_metrics(logs)

#         generator.save(os.path.join(wandb.run.dir, "generator.h5"))
#         discriminator.save(os.path.join(wandb.run.dir, "discriminator.h5"))

#         log_functions.sample_images(generator, [sample_noise, sample_types], samples)
        


