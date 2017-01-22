#!/usr/bin/env python

import unittest

import coredstl
from coredstl import *



# --------------- unit tests ------------------  
  
class TestCoreDstl(unittest.TestCase):
    """ Unit tests """
    def test_data(self) :
        self.assertEqual(len(class_types), len(class_shortname))
        self.assertEqual(len(class_types), len(class_color))
        self.assertEqual(len(class_types), len(class_zorder))

        self.assertEqual(len(class_types), len(set(filename_to_classType.values() )) )

        ext = set(extended_classes)
        for cls in filename_to_classType.keys():
            self.assertIn(cls,ext)

    def test_feature_loc(self):
        for itype in image_types : 
            self.assertEqual( len( feature_loc[itype] ), nb_channels[itype] )



    def test_grid_to_image_coords(self) :
        coords = grid_to_image_coords( (0.001, 0.001) )
        grid = image_to_grid_coords( coords)
        self.assertAlmostEqual( grid[0], 0.001 )
        self.assertAlmostEqual( grid[1], 0.001 )

        (col, row) = grid_to_image_coords( image_to_grid_coords( (1000, 2000) ) )
        self.assertAlmostEqual( col, 1000)
        self.assertAlmostEqual( row, 2000)

        col,row = grid_to_image_coords( (0.0, -1.0), image_size=(100,100), grid_size=(1.,1.) )
        self.assertAlmostEqual(col, 0.0)
        self.assertAlmostEqual(row, 99.)

    def test_image_to_grid_coords(self):
        (x,y) = image_to_grid_coords( grid_to_image_coords( (0.6, 0.4) )  )
        self.assertAlmostEqual( x, 0.6)
        self.assertAlmostEqual( y, 0.4)




    def test_image_to_region_coords(self) :
        rcoords = image_to_region_coords( (100,100) , (1,4) )
        icoords = region_to_image_coords( rcoords, (1,4) )
        self.assertEqual( icoords[0], 100 )
        self.assertEqual( icoords[1], 100 )
        
        rcol, rrow = image_to_region_coords( (100,200), (2,3), image_size=(1000,1000), border=50) 
        self.assertEqual(rcol, 2150)
        self.assertEqual(rrow, 3250)

    def test_region_to_image_coords(self):
        rcol, rrow = image_to_region_coords( (100,200), (2,3) ) 
        icol, irow = region_to_image_coords( (rcol, rrow) , (2,3) )
        self.assertEqual(icol, 100)
        self.assertEqual(irow, 200)

        
        
  
    def test_parse_viewId(self) :
        self.assertEqual( parse_viewId('6020'), ('6020', None, None, None, None) )
        self.assertEqual( parse_viewId('6020_1_2'), ('6020', 1, 2, None, None) )
        self.assertEqual( parse_viewId('6020_5x5'), ('6020', 5, 5, None, None) )
        self.assertEqual( parse_viewId('6020_X_X'), ('6020', -1, -1, None, None) )
        self.assertEqual( parse_viewId('6020_1_2_P'), ('6020', 1, 2, 'P', None) )
        self.assertEqual( parse_viewId('6020_1_2_P_00'), ('6020', 1, 2 ,'P', 0) )
        self.assertEqual( parse_viewId('XXXX_X_X_X_XX'), ('XXXX', -1, -1, 'X', -1) )

    def test_compose_viewId(self) :
        self.assertEqual( ('6020'), compose_viewId('6020', None, None, None, None) )
        self.assertEqual( ('6020_1_2'), compose_viewId('6020', 1, 2, None, None) )
        self.assertEqual( ('6020_5x5'), compose_viewId('6020', 5, 5, None, None) )
        self.assertEqual( ('6020_1_2_P'), compose_viewId('6020', 1, 2, 'P', None) )
        self.assertEqual( ('6020_1_2_P_0'), compose_viewId('6020', 1, 2 ,'P', 0) )




    # ---- Utility routines ---

    def test_round(self):
        # Test round_up
        self.assertEqual(100, _roundup(94,10) )
        self.assertEqual(90, _rounddown(94,10) )

        
        
    def test_stretch_to_uint8(self):
        data = np.asarray( range(0,11), dtype='uint8' )
        data = _stretch_to_uint8(data, 2., 8.)
        self.assertEqual( (0, 0, 0, 42, 85, 127, 170, 212, 255, 255, 255), tuple(data) )
        
        data = _stretch_to_uint8( np.asarray( range(4,5), dtype='float' ) ,  0., 8.) 
        self.assertEqual([127], data)

class TestDstl(unittest.TestCase):
    # dstl
    
    def test_with(self) :
        with Dstl() as dstl: 
            pass

    def test_path(self) :
        dstl = Dstl()
        fn = dstl.path('train_wkt')
        self.assertEqual( fn, '../input/train_wkt_v4.csv')


    def test_grid_sizes(self):
        dstl = Dstl()
        xmax, ymin = dstl.grid_sizes['6010_1_2']
        self.assertAlmostEqual(xmax,  0.009169) 
        self.assertAlmostEqual(ymin, -0.009042)


    def test_wkt_polygons(self):
        dstl = Dstl()
        fn = dstl.path('train_wkt')
        wkt_polygons = load_wkt_polygons(fn, ['6010_1_2','6060_2_3'] )
        assert( len(wkt_polygons['6060_2_3'])==10 )
        assert( len(wkt_polygons['6010_1_2'][5])==1733) # trees
        assert( wkt_polygons['6060_2_3'][2].is_valid)
 
 
    def test_load_image(self):
        dstl = Dstl()
        img = dstl.load_image('6010_1_2', '3')
        self.assertEqual(img.shape, (3, 3349, 3396))
        img = dstl.load_image('6010_1_2', 'A')
        self.assertEqual(img.shape, (8, 134, 136) )
        img = dstl.load_image('6010_1_2', 'M')
        self.assertEqual(img.shape, (8, 837, 849))
        img = dstl.load_image('6010_1_2', 'P')
        self.assertEqual(img.shape, (1, 3348, 3396))


    def test_image_sizes(self):
        dstl = Dstl()
        width, height= dstl.image_size('6010_1_2')
        self.assertAlmostEqual( - std_xmax/std_ymin, 1.* width/height, places=3)
        

    def test_imageIds(self):
        dstl = Dstl()
        self.assertEqual(len(dstl.imageIds), 450)


    def test_train_imageIds(self):
        dstl = Dstl()
        self.assertEqual(len(dstl.train_imageIds), 25)


    def test_test_imageIds(self):
        dstl = Dstl()
        self.assertEqual(len(dstl.test_imageIds), 429)


    def test_regionIds(self):
        dstl = Dstl()
        self.assertEqual(len(dstl.regionIds), 18)


    def test_channel_range(self):
        dstl = Dstl()
        low, high = dstl.channel_range('3', 0)
        self.assertEqual(low,124)
        self.assertEqual(high,763)


class TestPolygons(unittest.TestCase):
    def setUp(self):
        self.dstl = Dstl()
        
    def tearDown(self):
        self.dstl.close()

    def test_class_polygons_to_composite(self):
        xmax, ymin = self.dstl.grid_sizes['6010_1_2']
        width, height= self.dstl.image_size('6010_1_2')

        # Test class_polygons_to_composite
        iid= '6010_1_2'
        class_polygons = self.dstl.wkt_polygons[iid]

        fn = iid +'_composite_test.png'
        polygons_to_composite(class_polygons, xmax, ymin, width, height, fn, outline=False)
        fn = iid +'_outline_test.png'
        polygons_to_composite(class_polygons, xmax, ymin, width, height, fn, outline=True)


    def test_polygons_to_mask(self):
        polygons =  self.dstl.wkt_polygons['6010_1_2'][4]
        assert(len(polygons) == 12)
        xmax, ymin = self.dstl.grid_sizes['6010_1_2']
        width, height = self.dstl.image_size('6010_1_2')
        polygons_to_mask(polygons, xmax, ymin, width, height, filename=None) # FIXME



    def test_mask_to_polygons(self):
        actual_polygons = self.dstl.wkt_polygons['6010_1_2'][4]
        assert(len(actual_polygons) == 12)
        xmax, ymin = self.dstl.grid_sizes['6010_1_2']
        width, height = self.dstl.image_size('6010_1_2')
        mask = polygons_to_mask(actual_polygons, xmax, ymin, width, height)
        predicted_polygons = mask_to_polygons(mask, xmax, ymin)
        new_mask = polygons_to_mask(actual_polygons, xmax, ymin, width, height)


    #def test_polygon_jaccard(self):
        jaccard, tp, fp, fn = polygon_jaccard(actual_polygons, predicted_polygons)
        assert(jaccard>0.9)
        #print(jaccard, tp, fp, fn)



        
if __name__ == '__main__':
    unittest.main()
