
import os
import pandas
from shapely import wkt
import matplotlib.pyplot as plt
from descartes.patch import PolygonPatch
import tifffile as tiff
from PIL import Image
import numpy as np


import csv

import random


random.seed(42)


#import rasterio
#from rasterio import features, Affine



# Give short names, sensible colors and zorders to object types
# Adapted from amanbh

classes = {
        1 : 'Building',
        2 : 'Structure',
        3 : 'Road',
        4 : 'Track',
        5 : 'Trees',
        6 : 'Crops',
        7 : 'Waterway',
        8 : 'Lake',
        9 : 'Truck',
        10 : 'Car',
        }
        
class_color = {
        1 : '0.5',
        2 : '0.2',
        3 : '#b35806',
        4 : '#dfc27d',
        5 : '#1b7837',
        6 : '#a6dba0',
        7 : '#74add1',
        8 : '#4575b4',
        9 : '#f46d43',
        10: '#d73027',
        }
        
class_zorder = {
        1 : 5,
        2 : 6,
        3 : 4,
        4 : 1,
        5 : 3,
        6 : 2,
        7 : 7,
        8 : 8,
        9 : 9,
        10: 10,
        }

image_classes = ('3', 'A', 'M', 'P')


image_depth = { '3' : 8,
                 'P': 8,  
                 'A': 64,
                 'M': 8 }


input_dir = "../input"
output_dir = "../data"
training_data_filename =  "train_wkt_v4.csv"
grid_sizes_filename = "grid_sizes.csv"

grid_resolution = 1/1000000 
dpi = 512   # default dots per inch when creating images.

# Redundant?
def get_training_data() :  
    filename = os.path.join(input_dir,training_data_filename)
    names = ("image_id", "class_type", "wkt")
    td = pandas.read_csv(filename, names=names, skiprows=1)
    return td

def load_training_data() :  
    filename = os.path.join(input_dir,training_data_filename)
    names = ("image_id", "class_type", "wkt")
    td = pandas.read_csv(filename, names=names, skiprows=1)
    return td

def progress(string = '.') : print(string, end="", flush=True)





# deprecated 
def  get_grid_sizes():
     filename = os.path.join(input_dir,grid_sizes_filename)
     names = ['image_id', 'xmax', 'ymin']
     gs = pandas.read_csv(filename, names=names, skiprows=1)
     return gs
     
     
# deprecated     
def grid_size(grid_sizes, image_id) :
    # FIXME. Just return a dict from get_grid_sizes?
    xmax, ymin = grid_sizes[grid_sizes.image_id == image_id].iloc[0,1:].astype(float)
    return (xmax,ymin)
     

def load_grid_sizes():
    """Return a dictionary from image ids (iid) to (xmax, ymin) tuples """
    filepath = os.path.join(input_dir,grid_sizes_filename)
     
    gs = {}
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile) 
        next(reader)    # skip header row
        for row in reader :
            iid = row[0]
            xmax = float(row[1])
            ymin = float(row[2])
            gs[iid] = (xmax, ymin)
    
    return gs

def load_train_iids():
    return get_train_ids()







def get_image_ids(data):
    return sorted(data.image_id.unique())
    
def get_train_ids():
    return get_image_ids(get_training_data())

    
def load_image(image_id, image_type ):
    # image_type is one of 3, A, M or P

    # TODO: Pad P type dimensions?
    
    if image_type == '3' :
        filename = '{}.tif'.format(image_id, image_type)
        filepath = os.path.join(input_dir, 'three_band', filename)
    else : 
        filename = '{}_{}.tif'.format(image_id, image_type) 
        filepath = os.path.join(input_dir, 'sixteen_band', filename)

    image = tiff.imread(filepath)
    
    # P images are greyscale. But add extra dimension (length 1) 
    if image_type == 'P' : 
        image = np.expand_dims(image, axis=0)
    
    return image


    
    
    
        