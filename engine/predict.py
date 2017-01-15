from __future__ import print_function

from core_dstl import *
from core_model import *
from build_dstl import *
          
 
def predict(datadir, modelname) :  
    filename = datadir+'/'+modelname+'.hdf5'
    progress('    Loading '+ filename)
    model = keras.models.load_model(filename)
    #model.summary() 
   # model.compile(optimizer='rmsprop',
    #           loss=the_loss,
    #           metrics=[javg, j1,j2,j3,j4,j5,j6,j7,j8,j9,j10, true_pos, false_pos, false_neg])
     
    progress('done')
    
    batchsize = 32
    chunksize = 64
    
    fn = getpath('data', datadir=datadir)
    datafile = h5py.File(fn, mode='r') 
     
    for iid in train_imageIds():
    #for iid in ['6100_1_3',] :
        progress('    '+iid)
        # Add border of chunsize, since chunks don't line up with image coordinates
        targets = np.zeros( (10, std_height+chunksize*2, std_width+ chunksize*2) )
         
        region, reg_row, reg_col = parse_viewId(iid)[0:3]
        chunks = image_chunks(iid, chunksize, train=False) 
        L = len(chunks)
        D = roundup(L,batchsize) -L
        chunks.extend(chunks[0:D])

        
        
        #todo  round chunks up to whole number of batchsizes ?
         
        for n in range(0, len(chunks), batchsize) :
        #for n in range(0, 64, batchsize) :
            progress()
            batch_coords = chunks[n:n+batchsize]
            data, labels = construct_batch(datafile, batch_coords) 

            
            results = model.predict_on_batch(data)
            
            for c in range(0, batchsize ) :
                region, rrow, rcol = chunks[n+ c] #check order row col
                icol, irow = region_to_image_coords( (rcol, rrow), (reg_col, reg_row) )
                targets[:, irow+chunksize: irow+2*chunksize, icol+chunksize:icol+2*chunksize] = results[c]
        
        targets = targets[:, chunksize:-chunksize, chunksize:-chunksize]
        
        
        # Save 
        poly = {}
        fn =  datadir+'/out.csv'
        f = open(fn, 'w')
        print('ImageId,ClassType,MultipolygonWKT')
        for m in range(0,10) :
            polys = mask_to_polygons( targets[m] ,xmax=std_xmax, ymin=std_ymin)
            poly[m+1] = polys
            if len(polys) == 0 :
                print('{},{},{}'.format(iid, m+1,'MULTIPOLYGON EMPTY'), file=f )
            else: 
                print('{},{},"{}"'.format(iid, m+1,  polys), file=f )
        
        fn =  datadir+'/'+iid+'_'+modelname + '_prediction.png'
        polygons_to_composite(poly, std_xmax, std_ymin, std_width, std_height, fn, outline=False)  
        progress('done')

            
            
if __name__ == "__main__":
    args = sys.argv
    modelname = args[1]
    datadir = datadir_default if len(args)==2 else args[2]
    predict(datadir,modelname)




            