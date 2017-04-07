#!/usr/bin/env python

""" Fully convolutional Fibonacci neural network for image segmentation.
Gavin E. Crooks (2017)

A stack of convolutional layers. The dilation rates of the layers follow the
Fibonacci sequence.

F_n = 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233

The Fibonacci sequence grows exponentially (asymptotically), and creates dense 
connections between nodes.

Here's a illustration of connections for exponentially expanding convolutional
nets, which shows the advantage of the Fibonacci sequence. Suppose you 
increase by powers of 3. Then there are no cross connections, and you just get 
a flat distribution of one connection from each bottom node to the top node.

rates = 9,3,1

             1    
    1        1        1 
 1..1..1..1..1..1..1..1..1
111111111111111111111111111

(Counts here are connections from top node to any given lower node.)

If you increase by powers of 2 then things get weird! The distribution is 
uneven. Note in particular there is only ever one connection from a top node to
the node directly underneath (Suppose at the top I hop right 4, but then hope
left 2 and then 1. No matter the depth of the network I can't ever get back to 
the orignal position). 

rates = 4,2,1

       1 
   1   1   1
 1 1 2 1 2 1 1
112132313231211

But with a Fibonacci sequence of rates you get dense connections.  In 
particular since F_n + F_n+1 = F_n+2 (by definition), you can always get back 
to the same position after three layers.

rates = 3,2,1

        1
     1  1  1
   1 111111 1
  112233332211 

"""

import argparse
import os
import sys

import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *


__description__ = 'Fully convolutional Fibonacci neural network for image segmentation.'
__author__ = 'Gavin E. Crooks'


def fibnet(depth, features, tilesize, input_features=None, output_features=None, border_mode = 'valid', shortcut = True) :
    """Construct a fully convolutional Fibonacci neural network suitable for image segmentation.
    
    Keyword arguments:
     depth -- integer number of convolutional layers (between 6 and 28)
     features -- integer features per layer
     tilesize -- integer side length of output tiles
     input_features -- integer features of input layer (default: features)
     output_features -- integer features of output layer (default: features)
     border_mode -- 'valid' (default) or 'same'. If 'valid' the input layer is padded with border so that output layer stays same size.
     shortcut -- add shortcuts through network
    """
    assert depth >= 6
    assert depth <= 28
    assert border_mode == 'valid' or border_mode == 'same'

    input_features = features if input_features is None else input_features
    output_features = features if output_features is None else output_features

      
    fibonacci = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]   
    dilation_rates = []
    dilation_rates.extend(fibonacci[0:depth//2])
    dilation_rates.extend(fibonacci[0:(depth+1)//2][::-1])
    
    if border_mode == 'valid' :
        tileborder = sum(rates) 
    else :
        tileborder = 0

    input_shape=(input_features, tilesize + 2*tileborder, tilesize + 2*tileborder)
 
    conv = [None] * depth
 
    inputs = Input(input_shape)

    conv[0] = Convolution2D(features, 1, 1, activation='relu', border_mode='same')(inputs)
    
    conv[1] = Convolution2D(features, 3, 3, activation='relu', border_mode='same')(conv[0]) 
    conv[2] = Convolution2D(features, 3, 3, activation='relu', border_mode='same')(conv[1])
    
    for d in range(3, (depth+2)//2) :
        rate = dilation_rates[d]
        conv[d] = AtrousConvolution2D(features, 3, 3, activation='relu', border_mode='same', atrous_rate=(rate,rate))(conv[d-1]) 

    for d in range( (depth+2)//2, depth-3) :
        rate = dilation_rates[d]
        if shortcut :
            source = merge([conv[d-1], conv[depth-d-1]], mode='concat', concat_axis=1)
        else :
            source = conv[d-1]
        conv[d] = AtrousConvolution2D(features, 3, 3, activation='relu', border_mode='same', atrous_rate=(rate,rate))(source) 
    
    conv[depth-3] = Convolution2D(features, 3, 3, activation='relu', border_mode='same')(conv[depth-4])
    conv[depth-2] = Convolution2D(features, 3, 3, activation='relu', border_mode='same')(conv[depth-3])
 
    conv[depth-1] = Convolution2D(output_features, 1, 1, activation='sigmoid', border_mode='same')(conv[depth-2])

    outputs = conv[depth-1]
    if border_mode == 'valid' :
        outputs =  Cropping2D(cropping=((53, 53), (53, 53)))(outputs)

    model = Model(input=inputs, output=outputs)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')
    return model

          
    
def _cli() :
    """Commmand line interface for creating a fibnet"""
    
    parser = argparse.ArgumentParser(
        description = __description__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        'depth', 
        action='store', 
        type = int,
        help='Number of layers in fibnet. Between 6 and 26')
                          
    parser.add_argument(
        'features', 
        action='store', 
        type = int,
        help='Number of features in each internal layer')
 
    parser.add_argument(
        'tilesize', 
        action='store', 
        type=int,
        help='Tile size')
                          
    parser.add_argument(
        '--input_features', 
        action='store', 
        type=int,
        default=20,
        help='Features in input layer')
    
    parser.add_argument(
        '--output_features', 
        action='store', 
        type=int,
        default=10,   
        help='features in output layer')

    parser.add_argument(
        '--name', 
        action='store', 
        type=str,
        default='fibnet',
        help='Name of model')

    parser.add_argument(
        '--border_mode', 
        action='store', 
        type=str,
        default='valid',
        help="Border mode: 'same' or 'valid'")

    parser.add_argument(
        '--shortcut', 
        action='store_true', 
        default=False,
        help="Add shortcuts between layers")


    parser.add_argument(
        '--verbose', 
        action='store_true', 
        default=False,
        help='Print summary of model.')
    
    opts = vars(parser.parse_args()) 
    
    FILENAME = '{name}{depth}_f{features}_i{input_features}_o{output_features}_t{tilesize}.hdf5'
    filename = FILENAME.format(**opts)
    
    model_name = opts.pop('name') 
    verbose = opts.pop('verbose')
    
    sys.stderr.write('Creating model: {}\n'.format(filename))
    if os.path.exists(filename) : 
        sys.exit("Error: Model exists: "+ filename)
    model=fibnet(**opts)
     
    if verbose:
        model.summary()
        
        
    model.save(filename)
    
if __name__ == "__main__":
    _cli()

