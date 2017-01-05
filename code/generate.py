
# Bit of a hack, but seems to he creating batch generators. 

import random, os
from dstl_utils import load_train_iids, output_dir
import numpy as np
from keras import backend as K



def batch_generator(data_dict, labels_dict, coords, batchsize, chunksize=243):
    while 1:
        for i in range(0, len(coords), batchsize):
            
            data = np.stack([ data_dict[iid][:, x:x+chunksize, y:y+chunksize] 
                        for iid, x, y in coords[i: i+batchsize] ])
            data =  K.cast_to_floatx(data)
            
            labels = np.stack([ labels_dict[iid][:, x:x+chunksize, y:y+chunksize] 
                        for iid, x, y in coords[i: i+batchsize] ])
            labels =  K.cast_to_floatx(labels)
            
            
         #   data = np.rollaxis(data, 1, 4) 
         #   labels = np.rollaxis(labels, 1, 4) 
            
            
            yield data, labels
            


def batches(batchsize=32, chunksize = 243) :
    """ return train_gen, train_samples, val_gen, val_samples"""
    iids = load_train_iids()
    random.shuffle(iids)

    train_data = []
    val_data = []

    val_fraction = 0.2
    #chunksize = 243 # 3^5

    data_dict = {}
    labels_dict = {}

    train_coords = []
    val_coords = []


    for iid in iids :
        filepath = os.path.join(output_dir, "{}.npy".format(iid) )
        data = np.load(filepath, mmap_mode='r')
        data_dict[iid] = data
        print(data.shape)

        filepath = os.path.join(output_dir, "{}_labels.npy".format(iid) )
        labels = np.load(filepath, mmap_mode='r')
        labels_dict[iid] = labels #[4]    # Just trees
    
        C, X, Y = data.shape
        coords = []
        for x in range(random.randint(0, chunksize-1), X-chunksize,chunksize) :
            for y in range(random.randint(0, chunksize-1), Y-chunksize, chunksize) :
                coords.append( (iid, x,y) )              

        random.shuffle(coords)
        vn = int(len(coords) * val_fraction)
        val_coords.extend(coords[0:vn])
        train_coords.extend(coords[vn:])

    train_gen = batch_generator(data_dict, labels_dict, train_coords, batchsize, chunksize)
    train_samples = len(train_coords)
    train_samples -= train_samples % batchsize
    val_gen = batch_generator(data_dict, labels_dict, train_coords, batchsize, chunksize)
    val_samples = len(val_coords)
    val_samples -= val_samples % batchsize
    
    return train_gen, train_samples, val_gen, val_samples


#train_gen, train_samples, val_gen, val_samples = batches()
#
#for data, labels in train_gen:
#    print( data.shape, labels.shape)



