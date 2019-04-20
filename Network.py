#!/usr/bin/env python
#title           :Network.py
#description     :Architecture file(Generator and Discriminator)
#author          :Deepak Birla
#date            :2018/10/30
#usage           :from Network import Generator, Discriminator
#python_version  :3.5.4 

# Modules
from keras.layers import Dense
from keras.layers import Activation, Add, Concatenate, Multiply
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add


# Residual block
def res_block_gen(model, kernal_size, filters, strides):
    
    gen = model
    
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
        
    model = add([gen, model])
    
    return model

def dense_block(input):
            x1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(input)
            x1 = LeakyReLU(0.2)(x1)
            x1 = Concatenate()([input, x1])

            x2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x1)
            x2 = LeakyReLU(0.2)(x2)
            x2 = Concatenate()([input, x1, x2])

            x3 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x2)
            x3 = LeakyReLU(0.2)(x3)
            x3 = Concatenate()([input, x1, x2, x3])

            x4 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x3)
            x4 = LeakyReLU(0.2)(x4)
            x4 = Concatenate()([input, x1, x2, x3, x4])  

            x5 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x4)
            x5 = Lambda(lambda x: x * 0.2)(x5)
            x = Add()([x5, input])
            return x
def RRDB(input):
            x = dense_block(input)
            x = dense_block(x)
            x = dense_block(x)
            x = Lambda(lambda x: x * 0.2)(x)
            out = Add()([x, input])
            return out
    
    
def up_sampling_block(model, kernal_size, filters, strides):
    
    # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
    # Even we can have our own function for deconvolution (i.e one made in Utils.py)
    #model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = UpSampling2D(size = 2)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model


def discriminator_block(model, filters, kernel_size, strides):
    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model

# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Generator(object):

    def __init__(self, noise_shape):
        
        self.noise_shape = noise_shape

    def generator(self):
        
	    gen_input = Input(shape = self.noise_shape)
	    
	    model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(gen_input)
	    model = LeakyReLU(0.2)(model)
	    
	    x = RRDB(model)
        
        # Post-residual block
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = Lambda(lambda x: x * 0.2)(x)
        x = Add()([x, model])

	    
	    
	    # Using 2 UpSampling Blocks
	    for index in range(2):
	        x = up_sampling_block(x, 3, 256, 1)
	    
	    x = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(x)
	    hr_output = Activation('tanh')(x)
	   
	    generator_model = Model(inputs = gen_input, outputs = hr_output)
        
	    return generator_model

# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Discriminator(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape
    
    def discriminator(self):
        
        dis_input = Input(shape = self.image_shape)
        
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(dis_input)
        model = LeakyReLU(alpha = 0.2)(model)
        
        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)
        
        model = Flatten()(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha = 0.2)(model)
       
        model = Dense(1)(model)
        model = Activation('sigmoid')(model) 
        
        discriminator_model = Model(inputs = dis_input, outputs = model)
        
        return discriminator_model
