
# Quick script to extract sizes of training images.

from dstl_utils import load_grid_sizes, load_train_iids, load_image, image_classes
from dstl_utils import get_image_ids


train_iids = load_train_iids()
gs = load_grid_sizes()

iids = sorted(gs.keys())
print(len(iids))

for iid in iids : 
    img = load_image(iid, '3')
    C, H, W = img.shape
    print(iid, C, W, H) # Width x Height
  
        
