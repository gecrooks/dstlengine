#!/usr/bin/env python

from core_dstl import *


def assertAlmostEqual(x,y,acc=7) :
    assert(round(x-y, acc) == 0 )


# Test progress
progress("Testing")
progress()

# Test data
assert(len(classTypes) == len(class_shortname))
assert(len(classTypes) == len(class_color))
assert(len(classTypes) == len(class_zorder))

assert(len(classTypes) == len(set(filename_to_classType.values() )) )

ext = set(extended_classes)
for cls in filename_to_classType.keys():
    assert( cls in ext)

assert( os.path.isdir(input_dir) )
assert( os.path.isdir(output_dir) )
progress()


# Test load_wkt_polygons
wkt_polygons = load_wkt_polygons()
assert( len(wkt_polygons['6060_2_3'])==10 )
assert( len(wkt_polygons['6010_1_2'][5])==1733) # trees
progress()

#for iid, class_polygons in polygons.items() :
#    for ct in classTypes :
#        multipolygon = class_polygons[ct]
#        print('{} : {} {} \tcount = {}'.format(iid, ct, class_shortnames[ct], len(multipolygon)))
 

# Test load_grid_sizes
grid_sizes = load_grid_sizes()
xmax, ymin = grid_sizes['6010_1_2']
assertAlmostEqual(xmax,  0.009169) 
assertAlmostEqual(ymin, -0.009042)
progress()


# Test load_image_sizes
width, height = image_size('6010_1_2')
assertAlmostEqual( - xmax/ymin, 1.* width/height, acc=4)
progress()


# Test load_image
img = load_image('6010_1_2', '3')
assert(img.shape == (3, 3349, 3396))
img = load_image('6010_1_2', 'A')
assert(img.shape == (8, 134, 136) )
img = load_image('6010_1_2', 'M')
assert(img.shape == (8, 837, 849))
img = load_image('6010_1_2', 'P')
assert(img.shape == (1, 3348, 3396))
progress()


# Test imageIds
assert(len(imageIds())==450)
progress()


# Test train_imageIds
assert(len(train_imageIds())==25)
progress()


# Test test_imageIds
assert(len(test_imageIds())==429)
progress()


# Test regionIds
assert(len(regionIds())==18)
progress()


# Test imageId_to_region
assert( imageId_to_region('6020_1_2') == ('6020', 1, 2) )
progress()


# Test classId_to_imageId
assert( classId_to_imageId('6020_1_2_P_10') == ('6020_1_2','P','10') )
assert( classId_to_imageId('6020_1_2_P') == ('6020_1_2','P', None) )
assert( classId_to_imageId('6020_1_2') == ('6020_1_2',None, None) )
progress()


# Test imageId_to_classId
assert( imageId_to_classId('6020_1_2','P','10') =='6020_1_2_P_10')
assert( imageId_to_classId('6020_1_2','P', None) == '6020_1_2_P')
assert( imageId_to_classId('6020_1_2',None, None) == '6020_1_2' )
progress()


# Test filename
fn =filename('6020_1_2','P','10', 'extra', '.quack')
assert(fn== '../output/6020_1_2_P_10_extra.quack')
progress()




# Test class_polygons_to_composite
iid= '6010_1_2'
class_polygons = wkt_polygons[iid]
fn = filename(iid, extra = 'test_composite')
class_polygons_to_composite(class_polygons, xmax, ymin, width, height, fn, outline=False)
fn = filename(iid, extra = 'test_outline')
class_polygons_to_composite(class_polygons, xmax, ymin, width, height, fn, outline=True)
progress()


# Test polygons_to_mask
polygons = wkt_polygons['6010_1_2'][4]
assert(len(polygons) == 12)
xmax, ymin = grid_sizes['6010_1_2']
width, height = image_size('6010_1_2')
polygons_to_mask(polygons, xmax, ymin, width, height, filename=None)
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



progress('Done')

        

