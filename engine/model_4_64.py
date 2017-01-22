#shallow model


import sys

import h5py

import keras
from keras.models import Sequential
from keras.layers import *
#from keras.optimizers import RMSprop, SGD, Nadam

#from keras.metrics import mean_absolute_error
import keras.backend as K

from core_dstl import datadir_default

def create(filename) :
    features = 64
    
    input_shape=(20,64*3, 64*3)
    
    model= Sequential()
    
    model.add( BatchNormalization(axis=1,input_shape=input_shape ) )
    
    model.add( Convolution2D(features, 1,1, activation='relu', border_mode='valid') )
    model.add( BatchNormalization(axis=1,input_shape=input_shape ) )
    
    model.add( Convolution2D(features, 1,1, activation='relu', border_mode='valid') )
    model.add( BatchNormalization(axis=1,input_shape=input_shape ) )


        
    model.add( Convolution2D(features, 3,3, activation='relu', border_mode='valid') )
    model.add( BatchNormalization(axis=1,input_shape=input_shape ) )

    model.add( AtrousConvolution2D(features, 3,3, activation='relu', border_mode='valid', atrous_rate=(2,2)) )    
    model.add( BatchNormalization(axis=1,input_shape=input_shape ) )

    model.add( AtrousConvolution2D(features, 3,3, activation='relu', border_mode='valid', atrous_rate=(4,4)) )
    model.add( BatchNormalization(axis=1,input_shape=input_shape ) )

    model.add( AtrousConvolution2D(features, 3,3, activation='relu', border_mode='valid', atrous_rate=(8,8)) )
    model.add( BatchNormalization(axis=1,input_shape=input_shape ) )

    model.add( AtrousConvolution2D(features, 3,3, activation='relu', border_mode='valid', atrous_rate=(16,16)) )    
    model.add( BatchNormalization(axis=1,input_shape=input_shape ) )

    model.add( Convolution2D(features, 3,3, activation='relu', border_mode='valid') )
    model.add( BatchNormalization(axis=1,input_shape=input_shape ) )



    model.add( Convolution2D(features, 3,3, activation='relu', border_mode='valid') )
    model.add( BatchNormalization(axis=1,input_shape=input_shape ) )

    model.add( AtrousConvolution2D(features, 3,3, activation='relu', border_mode='valid', atrous_rate=(2,2)) )    
    model.add( BatchNormalization(axis=1,input_shape=input_shape ) )

    model.add( AtrousConvolution2D(features, 3,3, activation='relu', border_mode='valid', atrous_rate=(4,4)) )
    model.add( BatchNormalization(axis=1,input_shape=input_shape ) )

    model.add( AtrousConvolution2D(features, 3,3, activation='relu', border_mode='valid', atrous_rate=(8,8)) )
    model.add( BatchNormalization(axis=1,input_shape=input_shape ) )

    model.add( AtrousConvolution2D(features, 3,3, activation='relu', border_mode='valid', atrous_rate=(16,16)) )    
    model.add( BatchNormalization(axis=1,input_shape=input_shape ) )

    model.add( Convolution2D(features, 3,3, activation='relu', border_mode='valid') )
    model.add( BatchNormalization(axis=1,input_shape=input_shape ) )


    
    model.add( Convolution2D(features, 1,1, activation='relu', border_mode='valid') )
    model.add( BatchNormalization(axis=1,input_shape=input_shape ) )

    model.add( Convolution2D(10, 1,1, activation='sigmoid', border_mode='valid') )    
      
      
#    model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy') 
    model.summary()              

    model.save(filename)
    
    

if __name__ == "__main__":
    args = sys.argv
    modelName = args[0].split('.')[0]
    datadir = datadir_default if len(args)==1 else args[1]
    fn =datadir+'/'+modelName+'.hdf5'
    print('Creating model ',fn)

    create(fn)
