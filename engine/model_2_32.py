from core_dstl import *

import keras

from keras.models import Sequential, Model
from keras.layers import *
from keras.utils.np_utils import to_categorical        
from keras.optimizers import RMSprop, SGD, Nadam

from keras.metrics import mean_absolute_error
import keras.backend as K


def resin_block(input_tensor, features, subfeatures, rate=1):
    x = Convolution2D(subfeatures, 1, 1)(input_tensor)
    x = AtrousConvolution2D(subfeatures, 3,3, border_mode='valid', atrous_rate=(rate,rate))(x)    
    x = Activation('relu')(x)
    x = Convolution2D(features, 1, 1)(x)

    y = Cropping2D(cropping=((rate, rate), (rate, rate)))(input_tensor)
    x = merge([x, y], mode='sum')

    return x


def create(filename='model2.hdf5') :
    features = 32
    subfeatures= 32 
    input_shape=(20,64*3, 64*3)
    
    inp = Input(shape=input_shape)
    
    x = BatchNormalization(axis=1)(inp)
    
    x = Convolution2D(features, 1,1, activation='relu', border_mode='valid')(inp)
    x = resin_block(x, features, subfeatures, 1) 
    x = resin_block(x, features, subfeatures, 2) 
    x = resin_block(x, features, subfeatures, 4) 
    x = resin_block(x, features, subfeatures, 8) 
    x = resin_block(x, features, subfeatures, 1) 
    
    x = resin_block(x, features, subfeatures, 1) 
    x = resin_block(x, features, subfeatures, 2) 
    x = resin_block(x, features, subfeatures, 4) 
    x = resin_block(x, features, subfeatures, 8) 
    x = resin_block(x, features, subfeatures, 1) 

    x = resin_block(x, features, subfeatures, 1) 
    x = resin_block(x, features, subfeatures, 2) 
    x = resin_block(x, features, subfeatures, 4) 
    x = resin_block(x, features, subfeatures, 8) 
    x = resin_block(x, features, subfeatures, 1) 

    x = resin_block(x, features, subfeatures, 1) 
    x = resin_block(x, features, subfeatures, 2) 
    x = resin_block(x, features, subfeatures, 4) 
    x = resin_block(x, features, subfeatures, 8) 
    x = resin_block(x, features, subfeatures, 1)            
       
       
    x = Convolution2D(10, 1,1, activation='sigmoid', border_mode='valid')(x) 
    
      
    model = Model(inp, x)    
    model.summary()                           
    model.save(filename)
     



if __name__ == "__main__":
    args = sys.argv
    modelName = args[0].split('.')[0]
    datadir = datadir_default if len(args)==1 else args[1]
    fn =datadir+'/'+modelName+'.hdf5'
    print('Creating model ',fn)

    create(fn)



