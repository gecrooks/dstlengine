
# Quick script to extract sizes of training images.

from dstl_utils import load_grid_sizes, load_train_iids, load_image, image_classes



train_iids = load_train_iids()
gs = load_grid_sizes()

for iid in train_iids : 
    print(iid)
    
    # x is width, y is height
    # Aspect ratio is width/height 
    (xmax, ymin) = gs[iid]  
    print("xmax, ymin: ", xmax, ymin, "aspect:", - xmax/ymin)  
    
    print("# type: (channels, height, width) ")
    for image_type in image_classes:
        img = load_image(iid, image_type)
        s = img.shape
        
        H = s[1]
        W = s[2]
        
        print(" ", image_type, ' : ', s, "aspect:", W/H, "range:", img.min(), img.max())
        
        
        
