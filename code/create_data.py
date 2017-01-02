
# Munge image data
#   a) Create RGB png image from 3 channel images
#   b) For each channel of each image class of each image,
#       extract channel, rescale to consistant size, and save as greyscale png
#   c) stack all 20 channels and save as numpy array.

# Supersedes create_images.py



from dstl_utils import image_classes, load_train_iids, load_image, progress, image_depth, output_dir
import numpy as np
from PIL import Image
import os


for iid in load_train_iids():
    progress(iid+' ')

    data = []
        
    # Rescale all images to same size as '3' channel images.
    a = load_image(iid, '3') 
    C,H,W = a.shape            # axes are backwards here (why?)
 
    # RGB version
    a = np.rollaxis(a, 0, 3) 
    a = a / image_depth['3']
    img = Image.fromarray(a.astype('uint8'))  
    filename = "{}_3.png".format(iid)
    filepath = os.path.join(output_dir, filename)
    img.save(filepath) 
    progress() 
 
    
    for imgcls in image_classes :
        a = load_image(iid, imgcls)
        a = a / image_depth[imgcls] # scale to 8 bit dynamic range
        for channel in range(a.shape[0]) :
            img = Image.fromarray(a.astype('uint8')[channel])          
            img = img.resize( (W,H) )
            data.append( np.asarray(img) )
      
            filename = "{}_{}{}.png".format(iid, imgcls, channel+1)
            filepath = os.path.join(output_dir, filename)
            img.save(filepath) 
            progress()
    
    d = np.stack(data)  
    filename = "{}.npy".format(iid)
    filepath = os.path.join(output_dir, filename)     
    np.save(filepath, d)
    progress()
    print()
    

 


 
        