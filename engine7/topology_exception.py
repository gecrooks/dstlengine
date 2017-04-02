# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 12:22:50 2016

@author: ironbar

Function for dealing with topology exception errors
"""
import pandas as pd
import numpy as np
import time
import shapely
import shapely.geometry
import shapely.wkt


def __get_valid_wkt_str(input, precision, debug_print):
    """
    Function that checks that a wkt str is valid
    """
    if type(input) is str:
        wkt_str = input
    else:
        #Get the initial str
        wkt_str = shapely.wkt.dumps(input, rounding_precision=precision)
    #Loop until we find a valid polygon
    for i in range(100):
        polygon = shapely.wkt.loads(wkt_str)
        if not polygon.is_valid:
            #print debugging info
            print('Invalid polygon in %s. Iteration: %i' % (debug_print, i))
            #Use buffer to try to fix the polygon
            polygon = polygon.buffer(0)
            #Get back to multipolygon if necesary
            if polygon.type == 'Polygon':
                polygon = shapely.geometry.MultiPolygon([polygon])
            #Dump to str again
            wkt_str = shapely.wkt.dumps(polygon, rounding_precision=precision)
        else:
            break
    return wkt_str

    
def __create_square_around_point(point, side):
    """
    Creates a square polygon with shapely given 
    the center point
    """
    #Create canonical square points
    square_points = np.zeros((4,2))
    square_points[1, 0] = 1
    square_points[2] = 1
    square_points[3, 1] = 1
    #Scale the square
    square_points *= side
    #Position to have the point in the center
    for i in range(2):
        square_points[:, i] += point[i] - side/2.
    pol = shapely.geometry.Polygon(square_points)
    return pol    
    
def repair_topology_exception(submission_path, precision, image_id, n_class, point, 
                              side=1e-4):
    """
    Tries to repair the topology exception error by creating a squared 
    hole in the given point with the given side
    """
    start_time = time.time()
    #Load the submission
    print('Loading the submission...')
    df = pd.read_csv(submission_path)
    #Loop over the values changing them
    #I'm going to use the values because I think iterating over the dataframe is slow
    #But I'm not sure
    print('Looping over the polygons')
    polygons = df.MultipolygonWKT.values
    img_ids = df.ImageId.values
    class_types = df.ClassType.values
    for i, polygon, img_id, class_type in zip(range(len(polygons)), polygons,
                                           img_ids, class_types):
        if img_id == image_id and n_class == class_type:
            polygon = shapely.wkt.loads(polygon)
            square = __create_square_around_point(point, side)
            #polygon = polygon.difference(square)
            polygons[i] = 'MULTIPOLYGON EMPTY'
#__get_valid_wkt_str(polygon, precision,
 #               '%s class%i' % (img_id, class_type))
    #Update the dataframe
    df.MultipolygonWKT = polygons
    #Save to a new file
    print('Saving the submission...')
    df.to_csv(submission_path,
              index=False)
    print('It took %i seconds to repair the submission' % (
        time.time() - start_time))

if __name__ == '__main__':
    #repair_topology_exception('simple.csv',
    #                              precision=6,
    #                              image_id='6150_3_4',
    #                              n_class=2,
    #                              point= (0.000154, -0.003785),
    #                              side=1e-3)



    #repair_topology_exception('simple.csv',
    #                              precision=6,
    #                              image_id='6020_4_3',
    #                              n_class=2,
    #                              point= (0.0040803529411764707, -0.00031935294117647103),
    #                         side=1e-3)

    #repair_topology_exception('simple.csv',
    #                              precision=6,
    #                              image_id='6020_4_3',
    #                              n_class=6,
    #                              point= (0.00759, -0.000113),
    #                              side=1e-3)
    
    #repair_topology_exception('simple.csv',
    #                              precision=6,
    #                              image_id='6110_2_3',
    #                              n_class=6,
    #                              point= ( 0.00395, -0.002246),
    #                              side=1e-3)
    #repair_topology_exception('simple.csv',
    #                              precision=6,
    #                              image_id='6030_1_2',
    #                              n_class=6,
    #                              point= ( 0.00213, -0.001663),
    #                              side=1e-3)

    #repair_topology_exception('simple.csv',
    #                              precision=6,
    #                              image_id='6060_1_4',
    #                              n_class=2,
    #                              point= ( 0.004672, -0.0023179),
    #                              side=1e-3)

    repair_topology_exception('e106_s14.csv',
                                  precision=6,
                                  image_id='6150_3_4',
                                  n_class=6,
                                  point= (0.0012444444444444434, -0.007412),
                                  side=1e-3)

#, class 6, (0.0012444444444444434, -0.007412, 
