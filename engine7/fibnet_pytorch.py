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

# import keras
# from keras.models import *
# from keras.layers import *
# from keras.optimizers import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



__description__ = 'Fully convolutional Fibonacci neural network for image segmentation.'
__author__ = 'Gavin E. Crooks'


# define the CNN architecture
class Fibnet(nn.Module):
    """Construct a fully convolutional Fibonacci neural network suitable for image segmentation.
                
    Keyword arguments:
        depth (int): number of convolutional layers (between 6 and 28)
        features (int): features per layer
        tilesize (int): side length of output tiles
        input_features (int): features of input layer (default: features)
        output_features (int): features of output layer (default: features)
        border_mode (str): 'valid' (default) or 'same'. If 'valid' the input layer is padded with border so that output layer stays same size.
        shortcut (bool): add shortcuts through network
    """

    def __init__(self, arg):
        super(Fibnet, self).__init__()
        
        self.fibonacci = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]   
        self.dilation_rates = []
        self.dilation_rates.extend(fibonacci[0:depth//2])
        self.dilation_rates.extend(fibonacci[0:(depth+1)//2][::-1])

        self.depth = 0
        self.features = None
        self.tilesize = None
        self.input_features = None
        self.output_features =None
        self.border_mode = 'valid'
        self.shortcut = True

        self.optimizer = None
        self.loss = nn.CrossEntropyLoss()
        self.scheduler = None
        self.input_shape= None
        self.outputs = None

        assert self.depth >= 6
        assert self.depth <= 28
        assert self.border_mode == 'valid' or border_mode == 'same'


        conv = [None] * depth
     
        # TODO find equivalent pytorch function
        inputs = Input(input_shape)

        
        if border_mode == 'valid' :
            tileborder = sum(rates) 
        else:
            tileborder = 0

        input_shape=(self.input_features, self.tilesize + 2*tileborder, self.tilesize + 2*tileborder)

        self.input_features = self.features if self.input_features is None else self.input_features
        self.output_features = self.features if self.output_features is None else self.output_features

       

        self.conv0 = nn.Conv2d(features, 1, 1, padding=1)
        self.conv1 = nn.Conv2d(features, 1, 1, padding='same')
        self.conv2 = nn.Conv2d(features, 3, 3, padding=1)
        self.conv3 = nn.Conv2d(features, 3, 3, padding=1)
        self.conv[depth-3] = nn.Conv2d(features, 3, 3, dilation=(rate,rate), padding=1)
        self.conv[depth-2] = nn.Conv2d(features, 3, 3, dilation=(rate,rate), padding=1)
        self.conv[depth-1] = nn.Conv2d(features, 3, 3, dilation=(rate,rate), padding=1)


        self.outputs = self.conv[depth-1]

        if self.border_mode == 'valid' :
            # original keras class
            self.outputs =  Cropping2D(cropping=((53, 53), (53, 53)))(outputs) image[0:128, 0:128]

            # possible translation to pytorch???  pytorch discussion says cropping can be accomplished with negative numbers
            self.outputs = nn.functional.pad(self.outputs, (-53, -53, -53, -53) ) 



        def forward(self, x):
            """ how the network feeds data forward through the layers
            
                Parameters
                    x (tensor): being passhed through the layer
            """

            x = self.nn.MaxPool2d((F.relu(self.conv0(x))))
            x = self.nn.MaxPool2d((F.relu(self.conv1(x))))
            x = self.nn.MaxPool2d((F.relu(self.conv2(x))))
            x = self.nn.MaxPool2d((F.relu(self.conv3(x))))

            for d in range(3, (depth+2)//2) :
                rate = dilation_rates[d]
             #how to pass dilation to atrous layers?
            x = self.nn.MaxPool2d((F.relu(self.conv[depth-3](x))
            x = self.nn.MaxPool2d((F.relu(self.conv[depth-2](x)))
            x = self.nn.Sigmoid((F.relu(self.conv[depth-1](x)))
            
            if shortcut :
                x = torch.cat(([conv[d-1], conv[depth-d-1]]), dim=1)


          
    
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
    model=Fibnet(**opts)
     
    if verbose:
        model.summary()
        
        
    model.save(filename)
    
if __name__ == "__main__":
    _cli()
