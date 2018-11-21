import keras
import numpy as np
from keras.applications import vgg16
from keras.utils.vis_utils import plot_model

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import Input, Activation, BatchNormalization, Dot, add
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing import image
from keras import initializers
import matplotlib.pyplot as plt
from keras.optimizers import SGD
import glob
from tqdm import tqdm
import cv2
#Load the VGG model
count_running_shoe_cat = 0
vgg_model = vgg16.VGG16(weights='imagenet')

EPOCH_MAX = 200000
EPOCH_MIN = 0
EPOCH_STEP = 200


K.set_image_dim_ordering('tf')

BATCH_SIZE = 1000
HEIGHT, WIDTH, CHANNEL = 128,256, 3
input_shape = (HEIGHT, WIDTH, CHANNEL)

adam = Adam(lr=0.0002, beta_1=0.5)
dLosses = []
gLosses = []

def generate_noise(batch_size, noise_shape):
    #input noise to gen seems to be very important!
    return np.random.normal(0, 1, size=(batch_size,)+noise_shape)

def generator(noise_shape):
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


if __name__== "__main__":
    for epoch in range (29999, 30002):
        if  epoch == 30000:
            noise_shape = (1,1,100)
            count_running_shoe_cat = 0
            model = generator(noise_shape)
            model.summary()
            plot_model(model, to_file='./generator.png', show_shapes=True)
            #Model Loading
            model.load_weights("models_test2/dcgan_generator_epoch_30000.h5")

            #Image generation
            noise = generate_noise(BATCH_SIZE, noise_shape)
            gen_imgs = model.predict(noise)
            i = 0
            #Save images in a folder called Generated
            for im in gen_imgs:
                im = im * 127.5 + 127.5
                cv2.imwrite("generated/"+str(i)+".png",im)
                i += 1

            #Image classification
            image_count = 0
            for filename in glob.glob("generated/*.png"):
                image_count += 1

                original = load_img(filename, target_size=(224, 224))

                numpy_image = img_to_array(original)

                image_batch = np.expand_dims(numpy_image, axis=0)

                processed_image = vgg16.preprocess_input(image_batch.copy())
                
                # get the predicted probabilities for each class
                predictions = vgg_model.predict(processed_image)
                #print (predictions)
                cat = 'n04120489'
                cat2 = 'n03680355'
                cat3 = 'n03047690'
                cat4 = 'n04133789'

                label = decode_predictions(predictions)
                #print (label)

                #TOP 5 ACCURACY -> 1 se classe trovata nei primi 5 risultati
                for i in label:
                    for x in i:     
                        if x[0] == cat or x[0] == cat2 or x[0] == cat3 or x[0] == cat4:
                            count_running_shoe_cat += 1 #ranking
                            break
                        #ranking -= 1
            accuracy = count_running_shoe_cat / image_count #(((EPOCH_MAX-EPOCH_MIN) / (EPOCH_STEP) * 15) * 5)
            accuracy_percentage = accuracy * 100
            #print(count_running_shoe_cat)
            #print(accuracy)
            print(epoch, accuracy_percentage)
        #Test1 (1-200k) 6406 0.08541376040213534 8.541376040213535
        #Test2 (100k-200k) 3673 0.09794666666666667 9.794666666666666
        #Test3 (100k-130k) 1173 0.10426666666666666 10.426666666666666
        #Test4 (80k-150k) 1173 0.10426666666666666 10.426666666666666
        #Test5 (110k-160k) 1835 0.09786666666666667 9.786666666666667