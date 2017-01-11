
from core_dstl import *

import keras
import keras.backend as K




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


def image_chunks(imageId, chunksize=64, train=True):
    # Chunk up an image into chunks chunksize*chunksize
    # return a list of (region, row, col), where (row col) are origin point of chunks
    
    region, reg_row, reg_col = parse_viewId(imageId)[0:3]
    rrow, rcol = image_to_regional_coords(reg_row, reg_col,0,0)
    
    to_rrow = rrow + std_height
    to_rcol = rcol + std_width
    
    # round to chunksize, and make sure we cover entire image
    rrow = roundup(rrow, chunksize)-chunksize
    rcol = roundup(rcol, chunksize) -chunksize
    to_rrow = roundup(to_rrow, chunksize)
    to_rrcol = roundup(to_rcol, chunksize)
    
    if train :
        # when training we don't want edge chunks which may not have complete target information
        rrow +=chunksize
        rcol +=chunksize
        to_rrow -= chunksize
        to_rcol -= chunksize
        
    coords = []
    
    for row in range( rrow, to_rrow, chunksize) :
        for col in range( rcol, to_rcol, chunksize ) :
            coords.append( ( region, row, col ) )
    
    return coords
    

def image_to_regional_coords(reg_row, reg_col, row=0, col=0) :
    rrow = border + reg_row* std_height + row
    rcol = border + reg_col * std_width + col
    return rrow, rcol 
    

def gen_chunks(datadir, coords, chunksize=64, batchsize=32) :
    fn = getpath('data', datadir=datadir)
    datafile = h5py.File(fn, mode='r')   
    
    while 1:
        for i in range(0, len(coords), batchsize):
            yield construct_batch(datafile, coords[i: i+batchsize], chunksize)
     
     
def construct_batch(datafile, batch_coords, chunksize=64) :
    data = np.stack([ datafile[iid][1:21, x-chunksize:x+chunksize+chunksize, y-chunksize:y+chunksize+chunksize] 
                        for iid, x, y in batch_coords ])
    data =  K.cast_to_floatx(data)
        
    labels = np.stack([ datafile[iid][22:32, x:x+chunksize, y:y+chunksize] 
                        for iid, x, y in batch_coords ])
    labels =  K.cast_to_floatx(labels)
    labels /=255.
               
    return data, labels 
    