#!/usr/bin/env python

# Munge image data
#   For each channel of each image class of each image,
#       extract channel, rescale to consistant size
#   stack all 20 channels and save as numpy array.

# to do : image registration
# to do : do for all data, not just training data

from core_dstl import *

for iid in train_imageIds() :
    progress(iid)
    
    width, height = image_size(iid)
    data=[]
    
    for img_type in imageTypes :
        a = load_image(iid, img_type)
        a = a / image_depth[img_type] # scale to 8 bit dynamic range
        for channel in range(a.shape[0]) :
            img = Image.fromarray(a.astype('uint8')[channel])          
            img = img.resize( (width, height) )
            data.append( np.asarray(img) )
            progress()
    
    d = np.stack(data)  
    filename = "{}.npy".format(iid)
    filepath = os.path.join(output_dir, filename)     
    np.save(filepath, d)
    progress('Done')

    

 


 
        