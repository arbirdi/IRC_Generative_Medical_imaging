import tensorflow as tf 
import matplotlib.pyplot as plt 
import wandb

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
# @tf.function
def train_step(images, BATCH_SIZE, noise_dim, generator, discriminator, generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    wandb.log(
            {'discriminator_loss': disc_loss,
            'generator_loss': gen_loss
            }
            )

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(20, 20))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')
    
    plt.title("16 images at epoch {epoch}")

    plt.savefig('outputs/image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()