#!/usr/bin/env python

from core_engine import *
from core_dstl import _rounddown, _DATADIR
          
 
def train(datadir, model_name, nb_epoch=1, chunksize=64, chunkborder=64, batchsize=32) :  # magic constants
    valsize = 2048 # magic constant
    
    dstl =  Dstl(datadir = datadir)
    
    coords = [] 
    for iid in dstl.train_imageIds:
        chunks = chunk_image(iid, chunksize, train=True) 
        coords.extend(chunks)
        random.shuffle(coords)     
        L = _rounddown(len(coords), batchsize) 
        train_coords = coords[:L][:-valsize]
        val_coords = coords[:L][-valsize:]
     
    #print(len(train_coords), len(val_coords))
    filename = os.path.join(datadir, model_name+'.hdf5')
    print('    Loading '+ filename)
        
    model = keras.models.load_model(filename)
    #model.summary() 
     
    model_basename, model_epoch = model_name.split('_e')
    model_epoch = int(model_epoch)
        
    for epoch in range( model_epoch, model_epoch+nb_epoch) :
        model.compile(optimizer='rmsprop',
                  loss=the_loss,
                  metrics=[javg, true_pos, false_pos, false_neg])
            
        model.fit_generator(gen_batch(train_coords),
                                 len(train_coords),  
                                 nb_epoch=1, 
                                 validation_data=gen_batch(val_coords),
                                 nb_val_samples=len(val_coords) )   
            
        # can't save with custom loss function 
        model.compile(optimizer='rmsprop', loss='binary_crossentropy')
             
        new_modelname = '{}_e{}'.format(model_name, epoch)
        new_filename = os.path.join(dstl.datadir, new_modelname+'hdf5')
        model.save(new_filename)
            
            
if __name__ == "__main__":
    args = sys.argv
    model_name = args[1]
    datadir = _DATADIR if len(args)==2 else args[2]
    train(datadir,model_name)




            