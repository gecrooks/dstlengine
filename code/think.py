from generate import batches

import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, Convolution2D, Lambda, Dropout
from keras.utils.np_utils import to_categorical        
from keras.optimizers import RMSprop, SGD, Nadam

from keras.metrics import mean_absolute_error

width = 64

def think() :
    
    input_shape=(243, 243, 20)
    
    model= Sequential()
 
    model.add( Convolution2D(width, 1,1, border_mode='same', input_shape=input_shape) )
    model.add( Activation('relu') )     
    
    model.add( Convolution2D(width, 3,3, border_mode='same') )
    model.add( Activation('relu') )
    
    model.add( Convolution2D(10, 1,1, border_mode='same') )    
    model.add( Activation('sigmoid') )
    
    default_nadam_lr = 0.002
    opt = Nadam(default_nadam_lr)       
    model.compile(optimizer=opt,
              loss='kld',
              metrics=['accuracy',mean_absolute_error])
    model.summary()              


    
    train_gen, train_samples, val_gen, val_samples = batches()
    
    print(train_samples, val_samples)
    
    model.fit_generator( train_gen, train_samples, 10, validation_data=val_gen, nb_val_samples=val_samples)
    
think()