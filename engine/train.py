
from core_dstl import *
from core_model import *

          
 
def train(datadir, modelname) :  
      
     batchsize = 32
     chunksize = 64
     valsize = 2048
     
     coords = [] 
     for iid in train_imageIds():
         chunks = image_chunks(iid, chunksize, train=True) 
         coords.extend(chunks)
     random.shuffle(coords)     # Check is this in place or not!
     L = roundup(len(coords), batchsize) -batchsize
     train_coords = coords[:L][:-valsize]
     val_coords = coords[:L][-valsize:]
     
     
     filename = datadir+'/'+modelname+'.hdf5'
     progress('    Loading '+ filename)
     model = keras.models.load_model(filename)
     model.summary() 
     model.compile(optimizer='rmsprop',
               loss=the_loss,
               metrics=[javg, j1,j2,j3,j4,j5,j6,j7,j8,j9,j10, true_pos, false_pos, false_neg])
     
     model.fit_generator(gen_chunks(datadir, train_coords), len(train_coords),  nb_epoch=1, 
        validation_data=gen_chunks(datadir, val_coords), nb_val_samples=len(val_coords) )   
            
     model.compile(optimizer='rmsprop', loss='binary_crossentropy')        
     model.save(filename)
            
            
if __name__ == "__main__":
    args = sys.argv
    modelname = args[1]
    datadir = datadir_default if len(args)==2 else args[2]
    #print(datadir, modelname)
    train(datadir,modelname)




            