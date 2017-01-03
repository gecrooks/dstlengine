from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
from shapely import wkt
from descartes.patch import PolygonPatch
import tifffile as tiff
from PIL import Image
import numpy as np

import os
import csv
import random ; random.seed(42)
import tempfile
import sys



# Give short names, sensible colors and zorders to object types
# Adapted from amanbh

classTypes = [1,2,3,4,5,6,7,8,9,10]

class_shortname = {
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


filename_to_classType = {
'001_MM_L2_LARGE_BUILDING':1,
'001_MM_L3_RESIDENTIAL_BUILDING':1,
'001_MM_L3_NON_RESIDENTIAL_BUILDING':1,
'001_MM_L5_MISC_SMALL_STRUCTURE':2,
'002_TR_L3_GOOD_ROADS':3,
'002_TR_L4_POOR_DIRT_CART_TRACK':4,
'002_TR_L6_FOOTPATH_TRAIL':4,
'006_VEG_L2_WOODLAND':5,
'006_VEG_L3_HEDGEROWS':5,
'006_VEG_L5_GROUP_TREES':5,
'006_VEG_L5_STANDALONE_TREES':5,
'007_AGR_L2_CONTOUR_PLOUGHING_CROPLAND':6,
'007_AGR_L6_ROW_CROP':6, 
'008_WTR_L3_WATERWAY':7,
'008_WTR_L2_STANDING_WATER':8,
'003_VH_L4_LARGE_VEHICLE':9,
'003_VH_L5_SMALL_VEHICLE':10,
'003_VH_L6_MOTORBIKE':10}



extended_classes = [
    '001_MM_L2_LARGE_BUILDING',
    '001_MM_L3_EXTRACTION_MINE',
    '001_MM_L3_NON_RESIDENTIAL_BUILDING',
    '001_MM_L3_RESIDENTIAL_BUILDING',
    '001_MM_L4_BRIDGE',
    '001_MM_L5_MISC_SMALL_STRUCTURE',
    '002_TR_L3_GOOD_ROADS',
    '002_TR_L4_POOR_DIRT_CART_TRACK',
    '002_TR_L6_FOOTPATH_TRAIL',
    '003_VH_L4_AQUATIC_SMALL',
    '003_VH_L4_LARGE_VEHICLE',
    '003_VH_L5_SMALL_VEHICLE',
    '003_VH_L6_MOTORBIKE',
    '004_UPI_L5_PYLONS',
    '004_UPI_L6_SATELLITE_DISHES_DISH_AERIAL',
    '005_VO_L6_MASTS_RADIO_TOWER',
    '005_VO_L7_FLAGPOLE',
    '006_VEG_L2_SCRUBLAND',
    '006_VEG_L2_WOODLAND',
    '006_VEG_L3_HEDGEROWS',
    '006_VEG_L5_GROUP_TREES',
    '006_VEG_L5_STANDALONE_TREES',
    '007_AGR_L2_CONTOUR_PLOUGHING_CROPLAND',
    '007_AGR_L2_DEMARCATED_NON_CROP_FIELD',
    '007_AGR_L2_ORCHARD',
    '007_AGR_L6_ROW_CROP',
    '007_AGR_L7_FARM_ANIMALS_IN_FIELD',
    '008_WTR_L2_STANDING_WATER',
    '008_WTR_L3_DRY_RIVERBED',
    '008_WTR_L3_WATERWAY']


imageTypes = ('3', 'A', 'M', 'P')


input_dir = os.path.join('..','input')
output_dir = os.path.join('..','output')

training_data_filename =  "train_wkt_v4.csv"
grid_sizes_filename = "grid_sizes.csv"


grid_resolution = 1/1000000 

# default dots per inch when creating images.
dpi = 512   


verbose = True
def progress(string = '.') :
    if verbose :
        if string == 'Done': 
            print('Done')
        else :
            print(string, end="")
        sys.stdout.flush()



def load_train_wkt() :  
    filename = os.path.join(input_dir,"train_wkt_v4.csv")
    names = ("imageId", "classType", "wkt")
    data = pandas.read_csv(filename, names=names, skiprows=1)
    return data

def load_wkt_polygons():
    polygons = {}
    td = load_train_wkt() 
    
    for iid in train_imageIds():
        df_image = td[td.imageId == iid]
        classPolygons = {}
        for classType in classTypes:
            multipolygon = wkt.loads(df_image[df_image.classType == classType].wkt.values[0])
            classPolygons[classType] = multipolygon
        polygons[iid] = classPolygons
    
    return polygons
        

def load_sample_submission() :  
    filename = os.path.join(input_dir,"sample_submission.csv")
    names = ("imageId", "classType", "wkt")
    data = pandas.read_csv(filename, names=names, skiprows=1)
    return data


def load_grid_sizes():
    """Return a dictionary from imageIds to (xmax, ymin) tuples """
    filepath = os.path.join(input_dir,"grid_sizes.csv")
     
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
    
    
def image_size(imageId):
    """ Return the canonical (width, height) in pixels of a image region"""
    # Currently taking width and height from type '3' images
    img = load_image(imageId, '3')
    channels,  height, width = img.shape
    return (width, height)
        
       
    

# Load tiff images 
# imageType is one of 3, A, M or P
def load_image(imageId, imageType):
    if imageType == '3' :
        filename = '{}.tif'.format(imageId, imageType)
        filepath = os.path.join(input_dir, 'three_band', filename)
    else : 
        filename = '{}_{}.tif'.format(imageId, imageType) 
        filepath = os.path.join(input_dir, 'sixteen_band', filename)

    image = tiff.imread(filepath)
    
    # P images are greyscale. But add extra dimension (length 1) 
    if imageType == 'P' : 
        image = np.expand_dims(image, axis=0)
    
    return image



# Return a list of all image ids
def imageIds() :
    return sorted( load_grid_sizes().keys() )


# Return a list of all training data image ids
def train_imageIds() :
    return sorted(load_train_wkt().imageId.unique())

    
# Return a list of all test image ids.
# in same order as sample_submission.csv
def test_imageIds() :
    return load_sample_submission().imageId.unique()
  

# Return a list of all unique region ids
def regionIds():
    return sorted(set( [iid[0:4] for iid in imageIds()]) )


def imageId_to_region(imageid):
    """ Reads imageid such as "6020_1_2" and returns tuple regionid, region_x, region_y ("6020", 1, 2)"""
    regionId, region_x, region_y = imageid.split('_')
    return ( regionId, int(region_x), int(region_y) )
    
    
def classId_to_imageId(classId):
    """
    classId -> (imageId, imageType, channelId)
    imageType and channel are optional, can return None
     e.g. 6020_1_2_P_10 -> ("6020_1_2", "P", "10")
    """
    imageId = classId[0:8]
    tail = classId[8:].split('_')
    if len(tail)==1 : return (imageId, None, None) 
    if len(tail)==2 : return (imageId, tail[1], None) 
    return (imageId, tail[1], tail[2])


def imageId_to_classId(imageId, imageType=None, channelId=None, extra=None):
    classId = imageId
    if imageType : classId = classId + '_' + imageType
    if channelId : classId = classId + '_' + channelId
    if extra : classId = classId + '_' + extra
    return classId
        
        
def filename(imageId, imageType=None, channelId=None, extra=None, ext='.png'):
    fn = imageId_to_classId(imageId, imageType, channelId, extra)
    fn += ext
    fn = os.path.join(output_dir, fn)
    return fn
    
    
    

def polygons_to_mask(multipolygon, xmax, ymin, width, height, filename=None) :
    width /= dpi
    height /= dpi    
    fig = plt.figure(figsize=(width,height), frameon=False)
    axes = plt.Axes(fig, [0., 0, 1, 1]) # One axis, many axes
    axes.set_axis_off()         
    fig.add_axes(axes)  
    for polygon in multipolygon:
        patch = PolygonPatch(polygon,
                            color='#000000',
                            lw=0,               # linewidth
                            antialiased = True)
        axes.add_patch(patch)
    axes.set_xlim(0, xmax)
    axes.set_ylim(ymin, 0)
    axes.set_aspect(1)
    plt.axis('off')
    
    if filename is None :
        filename = tempfile.NamedTemporaryFile(suffix='.png')
    plt.savefig(filename, pad_inches=0, dpi=dpi, transparent=False)
    plt.clf()
    plt.close()
    a = np.asarray(Image.open(filename))
    a = (1.- a[:,:,0]/255.)  # convert from B&W to zeros and ones.
    return a    

    

def class_polygons_to_composite(class_polygons, xmax, ymin, width, height, filename, outline=False) :
    """ If outline is true, create transparent outline of classes suitable for layering over other images."""
    width /= dpi
    height /= dpi
     
    fig = plt.figure(figsize=(width,height), frameon=False)
    axes = plt.Axes(fig, [0., 0, 1, 1]) # One axis, many axes
    axes.set_axis_off()         
    fig.add_axes(axes)
    
    if outline :
        linewidth = 0.2
        transparent = True
        fill = False
    else:
        linewidth = 0.0
        transparent = False
        fill = True
        
    for classType, multipolygon in class_polygons.items():
        for polygon in multipolygon:
            patch = PolygonPatch(polygon,
                                color=class_color[classType],
                                lw=linewidth,   
                                alpha=1.0,
                                zorder=class_zorder[classType],
                                antialiased =True,
                                fill = fill)
            axes.add_patch(patch)
    axes.set_xlim(0, xmax)
    axes.set_ylim(ymin, 0)
    axes.set_aspect(1)
    plt.axis('off')
    
    plt.savefig(filename, pad_inches=0, dpi=dpi, transparent=transparent)
    plt.clf()
    plt.close()    


        
        