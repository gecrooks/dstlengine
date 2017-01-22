
from core_dstl import *
from core_dstl import _rounddown

import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import *

# th: theano or 'tf': tensorflow
_BACKEND = 'th'


# ================ Meta ====================

__description__ = 'DSTL Prognostication Engine'    
__version__ = '0.0.0'
__author__ = 'Gavin Crooks (@threeplusone) & Melissa Fabros (@mfab)'


# TODO: print pakage versions


def true_pos(y_true, y_pred):
    return K.mean(y_true * y_pred)

def false_pos(y_true, y_pred):
    return K.mean(y_pred) - true_pos(y_true, y_pred)

def false_neg(y_true, y_pred):
    return K.mean(y_true) - true_pos(y_true, y_pred)
    
def jaccard(y_true, y_pred):
    tpv = true_pos(y_true, y_pred)
    fpv = false_pos(y_true, y_pred)
    fnv = false_neg(y_true, y_pred)
    intersection = tpv+fpv+fnv+ K.epsilon()
    return tpv/intersection
    
def javg(y_true, y_pred):
    J= ( j1(y_true, y_pred) +
        j2(y_true, y_pred) +
        j3(y_true, y_pred) +
        j4(y_true, y_pred) +
        j5(y_true, y_pred) +
        j6(y_true, y_pred) +
        j7(y_true, y_pred) +
        j8(y_true, y_pred) +
        j9(y_true, y_pred) +
        j10(y_true, y_pred) )
    return J/10.


def jaccard_class(y_true, y_pred,c):
    tpv = true_pos(y_true[c], y_pred[c])
    fpv = false_pos(y_true[c], y_pred[c])
    fnv = false_neg(y_true[c], y_pred[c])
    intersection = tpv+fpv+fnv+ K.epsilon()
    #if intersection == 0 : return 0
    return tpv/intersection
    
def j1(y_true, y_pred): return  jaccard_class(y_true, y_pred,0)
def j2(y_true, y_pred): return  jaccard_class(y_true, y_pred,1)
def j3(y_true, y_pred): return  jaccard_class(y_true, y_pred,2)
def j4(y_true, y_pred): return  jaccard_class(y_true, y_pred,3)
def j5(y_true, y_pred): return  jaccard_class(y_true, y_pred,4)
def j6(y_true, y_pred): return  jaccard_class(y_true, y_pred,5)
def j7(y_true, y_pred): return  jaccard_class(y_true, y_pred,6)
def j8(y_true, y_pred): return  jaccard_class(y_true, y_pred,7)
def j9(y_true, y_pred): return  jaccard_class(y_true, y_pred,8)
def j10(y_true, y_pred): return  jaccard_class(y_true, y_pred,9)



def the_loss(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1.-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1.-K.epsilon())
    loss =  K.mean(y_true * K.log(y_true / y_pred), axis=-1)
    loss +=  K.mean((1.-y_true) * K.log((1.- y_true) / (1.-y_pred) ), axis=-1)
    return loss



def chunk_image(imageId, chunksize=64, chunkborder=64, train=True):
    # Chunk up an image into chunks chunksize*chunksize
    # return a list of (region, rrow, rcol), where (row col) are origin point of chunks
    
    region, reg_row, reg_col = parse_viewId(imageId)[0:3]
    rrow, rcol = image_to_region_coords( (0,0), (reg_row, reg_col) )
    
    if train :
        # round to chunksize.
        # when training we don't want edge chunks since they don't have complete target information
        rrow_stop = rrow + _rounddown(std_height, chunksize)
        rcol_stop = rcol + _rounddown(std_width, chunksize)    
    else:
        # round to chunksize, and make sure we cover entire image
        rrow_stop = rrow + _roundup(std_height, chunksize)
        rcol_stop = rcol + _roundup(std_width, chunksize)
            
    coords = []
    
    for row in range( rrow, rrow_stop, chunksize) :
        for col in range( rcol, rcol_stop, chunksize ) :
            coords.append( ( region, row, col ) )
    
    return coords
    
    
def make_batch(batch_coords, chunksize=64, chunkborder=64) :
    with Dstl() as dstl :
        data = np.stack([ 
                    dstl.data(region)[x-chunkborder:x+chunksize+chunkborder,
                                    y-chunkborder:y+chunksize+chunkborder, :] 
                        for region, x, y in batch_coords ]) 
        #print('data.shape', data.shape)
        data = np.transpose(data, (0, 3,1,2) )  # Theano transpose   
        #print('data.shape', data.shape)                       
        data = K.cast_to_floatx(data)

        labels = np.stack([ 
            dstl.targets(region)[x:x+chunksize, y:y+chunksize, :] 
                        for region, x, y in batch_coords ])
        labels = np.transpose(labels, (0, 3, 1, 2) )  # Theano transpose                          
        labels =  K.cast_to_floatx(labels)
        labels /=255.
        
    return data, labels


def gen_batch(coords, chunksize=64, chunkborder=64, batchsize=32) :
    while 1:
        #print('HERE', len(coords))
        for i in range(0, len(coords), batchsize):
            batch_coords = coords[i:i+batchsize] 
            #print(i, len(batch_coords))
            yield make_batch(batch_coords, chunksize, chunkborder)




