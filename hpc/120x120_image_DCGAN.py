import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers
import time
from sklearn import preprocessing
from model import make_discriminator_model, make_generator_model
from train import generate_and_save_images, train_step
from IPython import display
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='vae', help='model for dimensionality reduction')
parser.add_argument('--prefix', default='/home/emily/phd/drives/IRC_Generative_Medical_imaging/', help='when using RDS')
parser.add_argument('--EPOCHS', default=50,type=int, help='epochs to train')
parser.add_argument('--IMG_DIM', default=120,type=int, help='input size of img')
parser.add_argument('--BATCH_SIZE', default=100,type=int, help='number ')
parser.add_argument('--BUFFER_SIZE', default=4455,type=int, help='number ')
parser.add_argument('--LATENT_DIM', default= 128, type=int, help='dataset')
parser.add_argument('--wandb_name', default='dummy',type=str, help='name of run')
opt = parser.parse_args()
print(opt)

# SETUP WANDB
wandb.login(key='929507aa6962c12d80d4911b183780dfd225bef6') # ENTER KEY TO WANDB LOGIN HERE
# Pass them to wandb.init
wandb.init(config=opt)
# Access all hyperparameter values through wandb.config
wandb.init(id = opt.wandb_name, project='IRC', entity='emilymuller1991')

# DEVICES
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# DATA 
train_images = np.load(opt.prefix + '2d_data.npy')
train_labels = np.load(opt.prefix + '2d_label.npy')

training_image = np.zeros(shape=(240,240))
resized_training_images = np.zeros(shape=(train_images.shape[0], 120,120, 1))#train_images.shape[1]/2, train_images.shape[2]/2, 1))
for i in range(train_images.shape[0]):
    train_images[i] = train_images[i]/max(train_images[i].flatten()) #normalise images to [0,1] - possibly in the future change this to [-1, 1]
    resized_training_images[i] = train_images[i,::2,::2,:]#training_image[::2,::2,:]

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(resized_training_images).shuffle(opt.BUFFER_SIZE).batch(opt.BATCH_SIZE)
del train_images

# MODEL
generator = make_generator_model(opt.LATENT_DIM)
discriminator = make_discriminator_model(opt.IMG_DIM)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

seed = tf.random.normal([16, opt.LATENT_DIM])

################## TRAIN
for epoch in range(opt.EPOCHS):
    start = time.time()
    print(time.strftime("%H:%M:%S", time.localtime()))

    for image_batch in train_dataset:
        train_step(image_batch, opt.BATCH_SIZE, opt.LATENT_DIM, generator, discriminator, generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss)

    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)


    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec at {}'.format(epoch + 1, time.time()-start, time.strftime("%H:%M:%S", time.localtime())))

    generate_and_save_images(generator,
                            opt.EPOCHS,
                            seed)


