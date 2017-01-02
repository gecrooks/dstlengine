from rasterio import features
import rasterio
import shapely
import shapely.geometry

import cv2

from dstl_utils import *

# From 'shawn' on forums
def mask_to_polygons(mask):
    all_polygons=[]
    for shape, value in features.shapes(mask.astype(np.int16),
                                mask = (mask==1),
                                transform = rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):

        all_polygons.append(shapely.geometry.shape(shape))

    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        print("NOT VALID")
 #       all_polygons = all_polygons.buffer(0)
#        #Sometimes buffer() converts a simple Multipolygon to just a Polygon,
#        #need to keep it a Multi throughout
#        if all_polygons.type == 'Polygon':
#            all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    return all_polygons
  

def test() :
    iids = load_train_iids()[0:2]
    for iid in iids :
        filepath = os.path.join(output_dir, "{}_labels.npy".format(iid) )
        labels = np.load(filepath)

    
        C, X, Y = labels.shape
        
        for c in range(0,C):
        #c = 1
            mask = labels[c,:,:]
            polygons = mask_to_polygons(mask )
            print(c, len(polygons))
        #print(polygons)
        
            image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
        
        
test()