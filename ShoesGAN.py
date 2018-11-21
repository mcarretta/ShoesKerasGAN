import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import imageio
import matplotlib.gridspec as gridspec
from keras.layers import Input, Activation, BatchNormalization, Dot, add
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing import image
from keras import initializers
from keras.utils.vis_utils import plot_model
from keras.backend import shape
from PIL import ImageFile
from PIL import Image
from tqdm import tqdm

import cv2

import glob
K.set_image_dim_ordering('tf') #height, width, channel

BATCH_SIZE = 64
EPOCHS = 200000
HEIGHT, WIDTH, CHANNEL = 128,256, 3 
input_shape = (HEIGHT, WIDTH, CHANNEL)

adam = Adam(lr=0.0002, beta_1=0.5) #Adam was chosen as suggested 
dLosses = []
gLosses = []

def norm_img(img): #normalizes an image
    img = (img / 127.5) - 1
    return img

def denorm_img(img): #denormalizes an image
    img = (img + 1) * 127.5
    return img.astype(np.uint8) 

def sampleFromDataset(batch_size, image_shape, data_dir=None, data = None):
    sample_dim = (batch_size,) + image_shape
    sample = np.empty(sample_dim, dtype=np.float32)
    all_data_dirlist = list(glob.glob(data_dir))
    sample_imgs_paths = np.random.choice(all_data_dirlist,batch_size)
    for index,img_filename in enumerate(sample_imgs_paths):
        image = Image.open(img_filename)        
        image = image.resize((image_shape[1],image_shape[0]))
        image = image.convert('RGB') #remove transparent ('A') layer
        image = np.asarray(image)
        image = norm_img(image)
        sample[index,...] = image 
    return sample

def generator(noise_shape): #takes random noise as input and expands it into an image
    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32  # channel num
    output_dim = CHANNEL
    gen_input = Input(shape = noise_shape)
    generator = Conv2DTranspose(c4, kernel_size = (4,4), strides = (1,1), padding = "valid", data_format = "channels_last", kernel_initializer = "glorot_uniform")(gen_input)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
        
    generator = Conv2DTranspose(c8, kernel_size = (4,4), strides = (2,4), padding = "same", data_format = "channels_last", kernel_initializer = "glorot_uniform")(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    
    generator = Conv2DTranspose(c16, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = "glorot_uniform")(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    
    generator = Conv2DTranspose(c32, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = "glorot_uniform")(generator)    
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    
    generator = Conv2D(c64, kernel_size = (3,3), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = "glorot_uniform")(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)

    generator = Conv2DTranspose(c64, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = "glorot_uniform")(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)

    generator = Conv2DTranspose(output_dim, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = "glorot_uniform")(generator)
    generator = Activation('tanh')(generator)

        
    gen_opt = Adam(lr=0.00015, beta_1=0.5)
    generator_model = Model(input = gen_input, output = generator)
    generator_model.compile(loss='binary_crossentropy', optimizer=gen_opt, metrics=['accuracy'])
    return generator_model

def discriminator(input_shape):
    c2, c4, c8, c16 = 64, 128, 256, 512  # channel num: 64, 128, 256, 51
    
    dis_input = Input(shape = input_shape)
    
    discriminator = Conv2D(c2, kernel_size = (4,4), strides = (2,4), padding = "same", data_format = "channels_last", kernel_initializer = "glorot_uniform")(dis_input)
    discriminator = LeakyReLU(0.2)(discriminator)
    
    discriminator = Conv2D(c4, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = "glorot_uniform")(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)
    discriminator = Dropout(0.4)(discriminator)

    discriminator = Conv2D(c8, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = "glorot_uniform")(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)
    discriminator = Dropout(0.4)(discriminator)

    discriminator = Conv2D(c16, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = "glorot_uniform")(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)
    discriminator = Dropout(0.4)(discriminator)

    discriminator = Flatten()(discriminator)
    
    discriminator = Dense(1)(discriminator)
    discriminator = Activation('sigmoid')(discriminator)
    
    discriminator_model = Model(input = dis_input, output = discriminator)
    discriminator_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return discriminator_model

def generate_noise(batch_size, noise_shape):
    #input noise to gen seems to be very important!
    return np.random.normal(0, 1, size=(batch_size,)+noise_shape)

# Plot the loss from each batch
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/dcgan_loss_epoch_%d.png' % epoch)
    plt.close()


# the version as provided here prints and saves 16 different images in 16 different files in order to be able to obtain an evaluation from Google Images and Image Net. 
# If you uncomment the commented lines and comment the uncommented lines extempt the for cycle the system will print the images in a 4x4 grid
def plotGeneratedImages(img_batch,img_save_dir): 
        # Allocate figure
    #plt.figure(figsize=(WIDTH/4, HEIGHT/4))
    #gs1 = gridspec.GridSpec(4, 4)
    #gs1.update(wspace=0, hspace=0)
    #rand_indices = np.random.choice(img_batch.shape[0],16,replace=False)
    for i in range(16): 
        #plt.subplot(4, 4, i+1)
        #ax1 = plt.subplot(gs1[i])
        #ax1.set_aspect('auto')
        #rand_index = rand_indices[i]
        #image = img_batch[rand_index, :,:,:]
        image = img_batch[i, :,:,:] 
        image = denorm_img(image)
        #fig = plt.imshow(denorm_img(image))
        cv2.imwrite(img_save_dir+"_"+str(i)+".png",image)
        #fig.axes.get_xaxis().set_visible(False)
        #fig.axes.get_yaxis().set_visible(False)
        #plt.tight_layout()
        #plt.savefig(img_save_dir+"_"+str(i)+".png",bbox_inches='tight',pad_inches=0)
        #plt.savefig(img_save_dir,bbox_inches='tight',pad_inches=0)
        #plt.close()


# Save the generator and discriminator networks (and weights) for later use
def saveModels(epoch, generator, discriminator):
    generator.save('models/dcgan_generator_epoch_%d.h5' % epoch)
    discriminator.save('models/dcgan_discriminator_epoch_%d.h5' % epoch)


# The purpose of this training mechanism is to load just a portion of the images of the training batch at each iteration, and not to load into memory all the images of the training set. This heps to run it on limited
# resources and to fasten up the training procedure.
def train(epochs, batchSize):
    current_dir = os.getcwd()
    input_path = os.path.join(current_dir, 'training_set/*.png') #for copyright reason I cant include here the training set provided by Adidas, but you can try to download another training set with shoes
                                                                 # and resize it to 256x128
    print( 'Epochs:', epochs)
    print ('Batch size:', batchSize)
    random_dim = 100
    noise_shape = (1,1,random_dim)
    disc = discriminator(input_shape)
    gen = generator(noise_shape)

    disc.trainable = False

    ganInput = Input(shape=noise_shape)
    GAN_inp = gen(ganInput)
    GAN_opt = disc(GAN_inp)
    gan = Model(input = ganInput, output = GAN_opt)
    gan.compile(loss = 'binary_crossentropy', optimizer = adam, metrics=['accuracy'])

    #Here the system prints a graph of the structure of generator, discriminator and gan
    disc.summary()
    gen.summary()
    gan.summary()    

    plot_model(disc, to_file='./discriminator.png', show_shapes=True)
    plot_model(gen, to_file='./generator.png', show_shapes=True)
    plot_model(gan, to_file='./gan.png', show_shapes=True)

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        # Get a random set of input noise and images
        noise = generate_noise(batchSize,noise_shape)
        imageBatch = sampleFromDataset(batchSize, input_shape, data_dir = input_path)
        # Generate fake images
        generatedImages = gen.predict(noise)            
        concatenated_real_images = np.concatenate([imageBatch, generatedImages])
        # Labels for generated and real data
        real_data_Y = np.ones(batchSize) - np.random.random_sample(batchSize)*0.2
        fake_data_Y = np.random.random_sample(batchSize)*0.2
        data_Y = np.concatenate((real_data_Y,fake_data_Y))
        # Train discriminator
        disc.trainable = True
        gen.trainable = False
        # Print losses
        dloss_real = disc.train_on_batch(imageBatch, real_data_Y)
        dloss_fake = disc.train_on_batch(generatedImages, fake_data_Y)

        # Train generator
        gen.trainable = True
        GAN_X = generate_noise(batchSize, noise_shape)
        GAN_Y = real_data_Y
        disc.trainable = False
        gan_loss = gan.train_on_batch(GAN_X, GAN_Y)

    # Store loss of most recent batch from this epoch
        dLosses.append(dloss_fake)
        gLosses.append(gan_loss)

        if e == 1 or e % 200 == 0:
            if not os.path.exists(os.path.join(current_dir,"newShoes")):
                os.makedirs(os.path.join(current_dir,"newShoes"))
            newShoesDir = os.path.join(current_dir,"newShoes/%d" %e)
            plotGeneratedImages(generatedImages, newShoesDir)
            plotLoss(e)
        if e == 1 or e % 10000 == 0:
            saveModels(e, gen, disc)
# Plot losses from every epoch
        

if __name__== "__main__":
    train(EPOCHS,BATCH_SIZE)


