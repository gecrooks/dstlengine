#!/usr/bin/env python

import unittest

from dstl import *

class TestDstlUtils(unittest.TestCase):
    def test_round(self):
        self.assertEqual(100, roundup(94,10) )
        self.assertEqual(90, rounddown(94,10) )
        self.assertEqual(100, roundup(100,10) )
        self.assertEqual(100, rounddown(100,10) )
        

class TestViewId(unittest.TestCase):
    def test_parse_viewId(self) :
        self.assertEqual( parse_viewId('6020'), ('6020', None, None, None, None) )
        self.assertEqual( parse_viewId('6020_1_2'), ('6020', 1, 2, None, None) )
        self.assertEqual( parse_viewId('6020_5x5'), ('6020', 5, 5, None, None) )
        self.assertEqual( parse_viewId('6020_X_X'), ('6020', None, None, None, None) )
        self.assertEqual( parse_viewId('6020_1_2_P'), ('6020', 1, 2, 'P', None) )
        self.assertEqual( parse_viewId('6020_1_2_P_00'), ('6020', 1, 2 ,'P', 0) )
        self.assertEqual( parse_viewId('6020_X_2_P_00'), ('6020', None, 2 ,'P', 0) )
        
        self.assertEqual( parse_viewId('XXXX_X_X_X_X'), (None, None, None, None, None) )

    def test_compose_viewId(self) :
        self.assertEqual( ('6020'), compose_viewId('6020', None, None, None, None) )
        self.assertEqual( ('6020_1_2'), compose_viewId('6020', 1, 2, None, None) )
        self.assertEqual( ('6020_5x5'), compose_viewId('6020', 5, 5, None, None) )
        self.assertEqual( ('6020_1_2_P'), compose_viewId('6020', 1, 2, 'P', None) )
        self.assertEqual( ('6020_1_2_P_0'), compose_viewId('6020', 1, 2 ,'P', 0) )


    def test_iterate_viewIds(self) :
        #for iid in iterate_viewIds('6020_X_X_P_0'): print(iid)
        
        self.assertEqual(25, len(list(iterate_viewIds('6020_X_X_P_0') ) ) )
        self.assertEqual(5, len(list(iterate_viewIds('6020_2_X_P_0') ) ) )
        self.assertEqual(18, len(list(iterate_viewIds('XXXX_2_3_P_0') ) ) )
        self.assertEqual(4, len(list(iterate_viewIds('6100_2_3_X_0') ) ) )
        self.assertEqual(8, len(list(iterate_viewIds('6100_2_3_A') ) ) )
        self.assertEqual(1, len(list(iterate_viewIds('6100_2_3_A_1') ) ) )     

class TestDstlInfo(unittest.TestCase):
    
    def setUp(self):
        self.info = DstlInfo()
        
    def tearDown(self):
        pass
   
    def test_path(self) :
        fn = self.info.path('train_wkt')
        self.assertEqual( fn, '../input/train_wkt_v4.csv')

    def test_grid_sizes(self):
        xmax, ymin = self.info.grid_sizes['6010_1_2']
        self.assertAlmostEqual(xmax,  0.009169) 
        self.assertAlmostEqual(ymin, -0.009042)
 
 
    def test_load_image(self):
        img = self.info.load_image('6010_1_2', '3')
        self.assertEqual(img.shape, (3, 3349, 3396))
        img = self.info.load_image('6010_1_2', 'A')
        self.assertEqual(img.shape, (8, 134, 136) )
        img = self.info.load_image('6010_1_2', 'M')
        self.assertEqual(img.shape, (8, 837, 849))
        img = self.info.load_image('6010_1_2', 'P')
        self.assertEqual(img.shape, (1, 3348, 3396))

    def test_imageIds(self):
        self.assertEqual(len(self.info.imageIds), 450)


    def test_targetIds(self):
        self.assertEqual(len(self.info.targetIds), 25)


    def test_test_imageIds(self):
        self.assertEqual(len(self.info.testIds), 429)


    def test_regionIds(self):
        self.assertEqual(len(self.info.regionIds), 18)


    #FIXME
    def test_image_sizes(self):
        width, height= self.info.image_size('6010_1_2', '3')
        xmax, ymin = self.info.grid_sizes['6010_1_2']
        self.assertAlmostEqual( - xmax/ymin, 1.* width/height, places=3)    
        

    def test_wkt_polygons(self):
        assert( len(self.info.wkt.polygons('6060_2_3') )==10 )
        assert( len(self.info.wkt.polygons('6010_1_2') ['5'])==1733) # trees
        assert( self.info.wkt.polygons('6060_2_3')['2'].is_valid)
    
    def test_band_range(self):
        for itype in self.info.band_types : 
            start, stop = self.info.band_range[itype]
            self.assertEqual( stop-start, self.info.channel_nb[itype] )
     
class TestPolygons(unittest.TestCase):
    def setUp(self):
        self.info = DstlInfo()
        
    def tearDown(self):
        pass    

    def test_polygons_to_mask(self):
        polygons =  self.info.wkt.polygons('6010_1_2')['4']
        assert(len(polygons) == 12)
        xmax, ymin = self.info.grid_sizes['6010_1_2']
        width, height = self.info.image_size('6010_1_2','3')
        polygons_to_mask(polygons, xmax, ymin, width, height, filename=None) # FIXME



    def test_mask_to_polygons(self):
        actual_polygons = self.info.wkt.polygons('6010_1_2')['4']
        assert(len(actual_polygons) == 12)
        xmax, ymin = self.info.grid_sizes['6010_1_2']
        width, height = self.info.image_size('6010_1_2','3')
        mask = polygons_to_mask(actual_polygons, xmax, ymin, width, height)
        predicted_polygons = mask_to_polygons(mask, xmax, ymin, tolerance=1)
        new_mask = polygons_to_mask(actual_polygons, xmax, ymin, width, height)

        jaccard, tp, fp, fn = polygon_jaccard(actual_polygons, predicted_polygons)
        print(jaccard, tp, fp, fn)
        self.assertGreater(jaccard,0.9)

        iid = '6010_1_2'
        ctype = '4'
        fnout = 'output/test_{}_P_{}.csv'.format(iid,ctype)
        with open(fnout, 'w') as csvfile:
            column_headings = ["ImageId","ClassType","MultipolygonWKT"]
            writer = csv.writer(csvfile)
            writer.writerow(column_headings)
            
            polygons_text = shapely.wkt.dumps(predicted_polygons, rounding_precision=7)
            if  polygons_text == "GEOMETRYCOLLECTION EMPTY":
                polygons_text = "MULTIPOLYGON EMPTY"
            writer.writerow((iid, ctype, polygons_text))
 
      
if __name__ == '__main__':
    unittest.main()
