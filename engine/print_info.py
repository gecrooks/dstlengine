#!/usr/bin/env python

# Script to print some pertinent statistics about the training images


from core_dstl import *


grid_sizes = load_grid_sizes()
wkt_polygons = load_wkt_polygons()

for iid in train_imageIds() :
    print(iid)
    
    # x is width, y is height
    # Aspect ratio is width/height 
    (xmax, ymin) = grid_sizes[iid]  
    print("xmax, ymin: ", xmax, ymin, "aspect:", - xmax/ymin)  
    
    print("# type: (channels, height, width) ")
    for image_type in imageTypes:
        img = load_image(iid, image_type)
        channels, height, width = img.shape
        print(" ", image_type, ' : ', channels, height, width , "aspect:", width/height, "range:", img.min(), img.max())
        
    
    for ct in classTypes :
        polygons = wkt_polygons[iid][ct]
        print('{} : {} \tcount = {}'.format(ct, class_shortname[ct], len(polygons)))
    
