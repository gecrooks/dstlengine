#!/usr/bin/env python

from core_dstl import *
from build_dstl import *

# Test progress
progress("Testing build")
progress()

wkt_polygons = load_wkt_polygons()
grid_sizes = load_grid_sizes()
xmax, ymin = grid_sizes['6010_1_2']
width, height= image_size('6010_1_2')

# Test class_polygons_to_composite
iid= '6010_1_2'
class_polygons = wkt_polygons[iid]

fn = iid +'_composite_test.png'
polygons_to_composite(class_polygons, xmax, ymin, width, height, fn, outline=False)
fn = iid +'_outline_test.png'
polygons_to_composite(class_polygons, xmax, ymin, width, height, fn, outline=True)
progress()


# Test polygons_to_mask
polygons = wkt_polygons['6010_1_2'][4]
assert(len(polygons) == 12)
xmax, ymin = grid_sizes['6010_1_2']
width, height = image_size('6010_1_2')
polygons_to_mask(polygons, xmax, ymin, width, height, filename=None) # FIXME
progress()


# Test mask_to_polygons
actual_polygons = wkt_polygons['6010_1_2'][4]
assert(len(actual_polygons) == 12)
xmax, ymin = grid_sizes['6010_1_2']
width, height = image_size('6010_1_2')
mask = polygons_to_mask(actual_polygons, xmax, ymin, width, height)
predicted_polygons = mask_to_polygons(mask, xmax, ymin)
new_mask = polygons_to_mask(actual_polygons, xmax, ymin, width, height)
progress()


# Test polygon_jaccard
jaccard, tp, fp, fn = polygon_jaccard(actual_polygons, predicted_polygons)
assert(jaccard>0.9)
#print(jaccard, tp, fp, fn)
progress()



