from core_dstl import *
from build_dstl import *

wkt_polygons = load_wkt_polygons()

for iid in train_imageIds() :
    progress(iid)
    for c in range(1,11) :
        # Test polygons_to_mask
        actual_polygons = wkt_polygons[iid][c]
        progress()
        
        mask = polygons_to_mask(actual_polygons, std_xmax, std_ymin, std_width, std_height, filename=None) 
        progress()
        
        predicted_polygons = mask_to_polygons(mask, std_xmax, std_ymin)
        progress()
        
        total_area = - std_ymin * std_xmax
        
        jaccard, tp, fp, fn = polygon_jaccard(actual_polygons, predicted_polygons)
        
        print(jaccard, tp/total_area, fp/total_area, fn/total_area)
