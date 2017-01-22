#!/usr/bin/env python

from core_engine import *
from core_dstl import _DATADIR

def create() :

    features=64
    
    input_shape=(20, 64*3, 64*3)
    
    model= Sequential()
     
    model.add( Cropping2D(cropping=((11, 11), (11, 11)),input_shape=input_shape  ) )
    
    model.add( BatchNormalization(axis=1 ) )
    
    model.add( Convolution2D(features, 1,1, activation='relu', border_mode='valid') )
    model.add( BatchNormalization(axis=1 ) )
    
    model.add( Convolution2D(features, 3,3, activation='relu', border_mode='valid') )
    model.add( BatchNormalization(axis=1 ) )

    model.add( Convolution2D(features, 3,3, activation='relu', border_mode='valid') )
    model.add( BatchNormalization(axis=1) )

    model.add( AtrousConvolution2D(features, 3,3, activation='relu', border_mode='valid', atrous_rate=(2,2)) )    
    model.add( BatchNormalization(axis=1 ) )

    model.add( AtrousConvolution2D(features, 3,3, activation='relu', border_mode='valid', atrous_rate=(3,3)) )
    model.add( BatchNormalization(axis=1 ) )

    model.add( AtrousConvolution2D(features, 3,3, activation='relu', border_mode='valid', atrous_rate=(5,5)) )
    model.add( BatchNormalization(axis=1 ) )

    model.add( AtrousConvolution2D(features, 3,3, activation='relu', border_mode='valid', atrous_rate=(8,8)) )    
    model.add( BatchNormalization(axis=1 ) )

    model.add( AtrousConvolution2D(features, 3,3, activation='relu', border_mode='valid', atrous_rate=(13,13)) )    
    model.add( BatchNormalization(axis=1 ) )
    
    model.add( AtrousConvolution2D(features, 3,3, activation='relu', border_mode='valid', atrous_rate=(8,8)) )    
    model.add( BatchNormalization(axis=1 ) )
    
    model.add( AtrousConvolution2D(features, 3,3, activation='relu', border_mode='valid', atrous_rate=(5,5)) )
    model.add( BatchNormalization(axis=1 ) )

    model.add( AtrousConvolution2D(features, 3,3, activation='relu', border_mode='valid', atrous_rate=(3,3)) )
    model.add( BatchNormalization(axis=1 ) )

    model.add( AtrousConvolution2D(features, 3,3, activation='relu', border_mode='valid', atrous_rate=(2,2)) )    
    model.add( BatchNormalization(axis=1) )

    model.add( Convolution2D(features, 3,3, activation='relu', border_mode='valid') )
    model.add( BatchNormalization(axis=1 ) )

    model.add( Convolution2D(features, 3,3, activation='relu', border_mode='valid') )
    model.add( BatchNormalization(axis=1 ) )

    model.add( Convolution2D(10, 1,1, activation='sigmoid', border_mode='valid') )    
      
    model.summary()              

    return model
    
    
if __name__ == "__main__":
    args = sys.argv
    model_name = os.path.basename(args[0]).split('.')[0]
    datadir = _DATADIR if len(args)==1 else args[1]
    filename =os.path.join(datadir, model_name+'_e0.hdf5')
    print('Creating model ',filename)
    model = create()
    model.save(filename)
# End