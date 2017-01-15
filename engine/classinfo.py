

from core_dstl import *

import geojson

from shapely import geometry

from shapely.geometry import shape
from shapely.ops    import cascaded_union

geodir =  '../input/'+'train_geojson_v3'
from shapely.geometry import mapping, shape, MultiPolygon

import json

gs = load_grid_sizes()

image_areas = {} 

for iid in train_imageIds():
 #   xmax, ymin = gs[iid]
    
 #   total_area = -xmax*ymin
    
    #
    
    
    areas = []
    count = []
    image_areas[iid] = areas, count
    
    
    targets = filename_to_classType.keys()
    
    print(iid)
    for t, target in enumerate(extended_classes) :
        fn = geodir+'/'+ iid+'/'+ target+'.geojson' 

        std_area = -std_xmax*std_ymin

  
        if os.path.exists(fn) :
            with open(fn, 'r') as f:
                data =geojson.load(f)
                
                poly = [geometry.shape(f['geometry']) for f in data['features'] ]
                count.append( len(poly))
                poly = cascaded_union(poly)
                
                areas.append(poly.area / std_area) 
                
        else:
             areas.append(0.0)
             count.append(0)

    #print(areas)
    
total = np.zeros( (30,) )    
N = np.zeros( (30,))

for iid in image_areas.keys():
    a, c = image_areas[iid]
    total += a
    N += c
    
total /= 25

for t in range(0,30) :
    cls = extended_classes[t]
    ctype = filename_to_classType[cls] if cls in filename_to_classType else 0
        
        
    print(ctype, extended_classes[t].ljust(40), "\t{:.0f}\t{:0.5f}".format(N[t], total[t]) )
    
    
    
    
    
    
    
    
    
    
    

