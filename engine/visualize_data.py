#!/usr/bin/env python
    
# Create visualizations of the training data    
    
from core_dstl import *

# FIXME. I don't think these are correct.
image_depth = { '3' : 8,
                 'P': 8,  
                 'A': 64,
                 'M': 8 }


grid_sizes = load_grid_sizes()
wkt_polygons = load_wkt_polygons()
       
for iid in train_imageIds() :
    progress(iid)
    (xmax, ymin) = grid_sizes[iid]  
    width, height = image_size(iid)
    class_polygons = wkt_polygons[iid]
    
    # Targets
    fn = filename(iid, extra = 'targets')
    class_polygons_to_composite(class_polygons, xmax, ymin, width, height, fn)
    progress()
    
    # Outline of targets
    fn = filename(iid, extra = 'outline')    
    class_polygons_to_composite(class_polygons, xmax, ymin, width, height, fn, outline = True)
    outline = Image.open(fn)
    progress()


    # RGB version
    a = load_image(iid, '3') 
    a = np.rollaxis(a, 0, 3) 
    a = a/8 # 10 bit to 8 bit data
    img = Image.fromarray(a.astype('uint8'))  
    fn = filename(iid, '3')
    img.save(fn) 
    progress() 
    
    
    # Each channel, and each channel with targets overlay
    for itp in imageTypes:
        a =  load_image(iid, itp) 
        a = a/image_depth[itp]
        channels = a.shape[0]
        for c in range(0, channels) :
            fn = filename(iid, itp, str(c+1) )
            img = Image.fromarray(a.astype('uint8')[c] ) 
            img = img.resize( (width,height) , resample = Image.LANCZOS)  
            img.save(fn)  
            progress()  
            
            img = img.convert(mode = 'RGBA')
            fn = filename(iid, itp, str(c+1), 'outline')
            imp_comp = Image.alpha_composite(img, outline)
            imp_comp.save(fn) 
            progress()
 
    progress('Done')   
        