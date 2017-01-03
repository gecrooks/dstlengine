#!/usr/bin/env python

# Create the target lables from the raw training data
# Save data in output folder.

# TODO: change output orders to tf default of width, height, channels?


from core_dstl import *


grid_sizes = load_grid_sizes()
wkt_polygons = load_wkt_polygons()

for iid in train_imageIds() :
    progress(iid)
    (xmax, ymin) = grid_sizes[iid]  
    width, height = image_size(iid)
    class_polygons = wkt_polygons[iid]
    
    masks = []
    for ct in classTypes:
        polygons = class_polygons[ct]
        
        fn = os.path.join(output_dir,'{}_C{}.png'.format(iid,ct))        
        
        mask = polygons_to_mask(polygons, xmax, ymin, width, height)
        masks.append(mask)
        progress()
        
    targets = np.stack(masks)  
    #print("labels shape", targets.shape)
    fn = os.path.join(output_dir, "{}_targets.npy".format(iid))     
    np.save(fn, targets)
    progress('Done')
    




