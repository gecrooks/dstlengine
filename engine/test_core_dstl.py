#!/usr/bin/env python

import core_dstl
from core_dstl import *


def assertAlmostEqual(x,y,acc=7) :
    assert(round(x-y, acc) == 0 )


# Test progress
progress("Testing core")
progress()

# Test getpath
path = getpath('data', datadir=datadir_default)
assert(path == 'dstl.data/dstl.hdf5')




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

# Test round_up
assert( 100 == core_dstl.roundup(94,10) )
progress()

# Test feature_loc
for itype in imageTypes : assert( len( feature_loc[itype] ) == imageChannels[itype] )
assert(len(range(0,32)[feature_loc['data']]) == 20)
assert(len(range(0,32)[feature_loc['targets']]) == 10) 
progress()


# Test load_wkt_polygons
wkt_polygons = load_wkt_polygons(['6010_1_2','6060_2_3'] )
assert( len(wkt_polygons['6060_2_3'])==10 )
assert( len(wkt_polygons['6010_1_2'][5])==1733) # trees
assert( wkt_polygons['6060_2_3'][2].is_valid)
progress()
#for iid, class_polygons in wkt_polygons.items() :
#    for ct in classTypes :
#        multipolygon = class_polygons[ct]
#        print('{} : {} {} \tcount = {}'.format(iid, ct, class_shortname[ct], len(multipolygon)))
 
 
wkt_polygons = WktPolygons() 
assert( len(wkt_polygons['6060_2_3'])==10 )
assert( len(wkt_polygons['6010_1_2'][5])==1733) # trees
assert( wkt_polygons['6060_2_3'][2].is_valid)
progress()
 
 
 

# test load_grid_size
gs = load_grid_sizes()
xmax, ymin = gs['6010_1_2']
assertAlmostEqual(xmax,  0.009169) 
assertAlmostEqual(ymin, -0.009042)
progress()


gs = GridSizes()
xmax, ymin = gs['6010_1_2']
assertAlmostEqual(xmax,  0.009169) 
assertAlmostEqual(ymin, -0.009042)
progress()




# Test load_image_sizes
width, height= image_size('6010_1_2')
assertAlmostEqual( - xmax/ymin, 1.* width/height, acc=4)
progress()



# Test grid_to_image_coords 
(x,y) = image_to_grid_coords( grid_to_image_coords( (0.6, 0.4) )  )
assertAlmostEqual( x, 0.6)
assertAlmostEqual( y, 0.4)

col,row = grid_to_image_coords( (0.0, -1.0), image_size=(100,100), grid_size=(1.,1.) )
#print(col,row)
assertAlmostEqual(col, 0.0)
assertAlmostEqual(row, 99.)


# Test image_to_grid_coords
(col, row) = grid_to_image_coords( image_to_grid_coords( (1000, 2000) ) )
assertAlmostEqual( col, 1000)
assertAlmostEqual( row, 2000)


# Test image_to_region_coords
rcol, rrow = image_to_region_coords( (100,200), (2,3), image_size=(1000,1000), border=50) 
assert(rcol == 2150)
assert(rrow == 3250)

# Test region_to_image_coords
rcol, rrow = image_to_region_coords( (100,200), (2,3) ) 
icol, irow = region_to_image_coords( (rcol, rrow) , (2,3) )
assert(icol==100)
assert(irow ==200)




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


# Test stretch_to_uint8
data = np.asarray( range(-1,11), dtype='float' )
data = stretch_to_uint8(data, 0., 8.)
assert(127==stretch_to_uint8( np.asarray( range(4,5), dtype='float' ) ,  0., 8.) ) 


# Test dynamic_range
low, high = core_dstl._dynamic_range('3', 0)
assert(low==124)
assert(high==763)
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

# Test parse_viewId
assert( parse_viewId('6020') == ('6020', None, None, None, None) )
assert( parse_viewId('6020_1_2') == ('6020', 1, 2, None, None) )
assert( parse_viewId('6020_5x5') == ('6020', 5, 5, None, None) )
assert( parse_viewId('6020_X_X') == ('6020', -1, -1, None, None) )
assert( parse_viewId('6020_1_2_P') == ('6020', 1, 2, 'P', None) )
assert( parse_viewId('6020_1_2_P_00') == ('6020', 1, 2 ,'P', 0) )
assert( parse_viewId('XXXX_X_X_X_XX') == ('XXXX', -1, -1, 'X', -1) )
progress()

# Test compose_viewId
assert( ('6020') == compose_viewId('6020', None, None, None, None) )
assert( ('6020_1_2') == compose_viewId('6020', 1, 2, None, None) )
assert( ('6020_5x5') == compose_viewId('6020', 5, 5, None, None) )
assert( ('6020_1_2_P') == compose_viewId('6020', 1, 2, 'P', None) )
assert( ('6020_1_2_P_00') == compose_viewId('6020', 1, 2 ,'P', 0) )
progress()





progress('done')



        

