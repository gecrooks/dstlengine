
"""
DSTL Prognostication Engine: Core code
"""


from __future__ import print_function
from __future__ import division
from __future__ import absolute_import



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas

import shapely
from shapely import wkt
import shapely.geometry
import shapely.affinity

import rasterio
import rasterio.features

import h5py

import descartes
from descartes.patch import PolygonPatch

import tifffile as tiff

import PIL
from PIL import Image

import numpy as np

import os
import csv
import random ; random.seed(42)
import tempfile
import sys

description = 'DSTL Prognostication Engine'
version = '0.0.0'

# python -c 'import core_dstl; core_dstl.package_versons()'
def package_versions() :
    print('python   \t', sys.version[0:5] )
    print('numpy    \t', np.__version__ )
    print('matplotlib\t', matplotlib.__version__, '\t(',matplotlib.get_backend(),'backend )')
    print('tifffile  \t', tiff.__version__)
    print('shapely   \t', shapely.__version__)
    print('pandas    \t', pandas.__version__)
    print('rasterio  \t', rasterio.__version__)
    print('h5py      \t', h5py.__version__)
    print('descartes\t',  '???')
    print('pillow (PIL)\t', PIL.__version__)
    
    import theano
    print('theano     \t', theano.__version__)   
     
    import keras
    print('keras     \t', keras.__version__)




# Give short names, sensible colors and zorders to object types
# Adapted from @amanbh

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
      #  7 : '#74add1',
        7:  '#40a4df',            # Waterway 'Clear Water Blue'
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


# 30 extended classes
extended_classes = (
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
    '008_WTR_L3_WATERWAY')


imageTypes = ('3', 'A', 'M', 'P')

imageChannels = {'3':3, 'A':8, 'M':8, 'P':1}

imageType_names = {
    '3': '3-band',
    'P' : 'Panchromatic',
    'M' : 'Multispectral',
    'A' : 'SWIR'
    }





sourcedir_default = os.path.join('..','input')
sourcedir = sourcedir_default   # deprecate

datadir_default = 'dstl.data'



verbose = True

train_wkt_fname = 'train_wkt_v4.csv'
grid_sizes_fname = 'grid_sizes.csv'



_paths = {
    'train_wkt' : '{sourcedir}{sep}' + 'train_wkt_v4.csv',
    'grid_sizes': '{sourcedir}{sep}' + 'grid_sizes.csv',
    'sample'    : '{sourcedir}{sep}' + 'sample_submission.csv',
    'data'      : '{datadir}{sep}'+'dstl.hdf5',    
    'composite' : '{datadir}{sep}{imageId}_composite'+'.png',
}

def getpath(name, **kwargs) :
    kwargs['sep'] = os.sep
    kwargs['sourcedir'] = sourcedir
    path = _paths[name]
    path = path.format(**kwargs)
    return path
    


 
grid_resolution = 1./1000000 

# default dots per inch when creating images.
dpi = 512   


# Canonical width and height of each image. 
# The '3' band images are always almost this size, within a few pixels,
# except the right and bottom boundary images (of a 5x5 region), 
# which can be cropped smaller.
std_height = 3348
std_width = 3396


#  to do: Check these are correct
std_xmax =   2.7 * grid_resolution * std_width
std_ymin = - 2.7 * grid_resolution * std_height

 


# Maximum size of tiles (width and height) we test and train on.
tilesize = 64
tileborder = 64

# deprecate: Replace with tilesize and tileborder
chunksize = tilesize



# Minimum border of zero data around valid data in a region
# Should be several times tileborder. 
regionborder = 256
border = regionborder    # deprecate (ambiguous)



# Either "gzip" (moderate speed, good compression) or 'lzf' (very fast, moderate compression)
hdf5_compression = 'gzip'

# Chunksize optimized for training rather than data creation.
# autochunking dramatically improves data creation time,
# but slows batch read time
# hdf5_chunks = (64,128,128) 
hdf5_chunks = True  # autochunking



# Round up to next size
def roundup(x, size) : return ( (x+size-1) // size ) * size


# Height and width of a 5x5 region in pixels
# Assumes canonical image sizes, a border on each side,
region_width = 5 * std_width + 2 * border
region_height = 5 * std_height + 2 * border



# Number of features in feature maps. 
# Currently 20 input channels and 10 target masks
nb_features = 64


# Tells us where features are stored.
# 'data' and 'targets' return a slice object
# img_type returns list of where channels are stored
# loc = feture_loc[img_type][channel]
feature_loc = {
            'data': slice(1,21),
            'targets': slice(22,32),
            'valid': 0,                 # Not used yet
            '3': (1,2,3), 
            'M': (4,5,6,7,8,9,10,11), 
            'A': (12,13,14,15,16,17,18,19), 
            'P': (20,),
            # The 10 class catogories are indexed from 1
            # Class zero not used (yet)
            'C': (-1,22,23,24,25,26,27,28,29,30,31) }

# Note: bands not channels


# Kudos: @FPP_UK
#
#   If you look at the TIFF metadata you get the following info:
#   [3-bands] 6nnn_n_n: TIFFTAG_IMAGEDESCRIPTION={ bandlist = [ 4; 3; 2;] }
#   [16-bands] 6nnn_n_n_M: TIFFTAG_IMAGEDESCRIPTION={ bandlist = [ 6; 2; 3; 7; 4; 8; 5; 9;] }
#   [SWIR] 6nnn_n_n_A: TIFFTAG_IMAGEDESCRIPTION={ bandlist = [ 10; 11; 12; 13; 14; 15; 16; 17;] }

# This means that the 3-band images are actually redundant with 3 of the Multispectral bands.
# However the corresponding images appear to be translated by a pixel or two.
# Apparently the polygons are registered to the 3-band images (?), so if we ignore any bands it should
# be the 2nd, 3rd and 5th Multispectral bands.


banklists = { 
    'P' : (1),
    '3' : (4, 3, 2) ,
    'M' : ( 6, 2, 3, 7, 4, 8, 5, 9) , 
    'A' : ( 10, 11, 12, 13, 14, 15, 16, 17)
    }

#
# Based on the WorldView online bandwidth info and QGis extracted bandlist metadata and data description
#
# Sensor-Bands :  WV3-Type, WV3-label, WV3range, Image-ID Notes
# Kudos: @FPP_UK
bandinfo = {
    1 : ('Panchromatic', 'Panchromatic', '450-800nm', '6xxx_n_n_P 0.31m resolution'),
    2 : ('Multi-spectral', 'Coastal', '400-450nm', '6xxx_n_n_M, RGB 1.24m resolution'),
    3 : ('Multi-spectral', 'Blue', '450-510nm', '6xxx_n_n_M, RGB 1.24m resolution'),
    4 : ('Multi-spectral', 'Green', '510-580nm', '6xxx_n_n_M, RGB 1.24m resolution'),
    5 : ('Multi-spectral', 'Yellow', '585-625nm', '6xxx_n_n_M 1.24m resolution'),
    6 : ('Multi-spectral', 'Red', '630-690nm', '6xxx_n_n_M 1.24m resolution'),
    7 : ('Multi-spectral', 'Red Edge', '705-745nm', '6xxx_n_n_M 1.24m resolution'),
    8 : ('Multi-spectral', 'Near-IR1', '770-895nm', '6xxx_n_n_M 1.24m resolution'),
    9 : ('Multi-spectral', 'Near-IR2', '860-1040nm', '6xxx_n_n_M 1.24m resolution'),
    10 : ('SWIR', 'SWIR-1', '1195-1225nm', '6xxx_n_n_A resolution reduced from 3.7m to 7.5m'),
    11 : ('SWIR', 'SWIR-2', '1550-1590nm', '6xxx_n_n_A resolution reduced from 3.7m to 7.5m'),
    12 : ('SWIR', 'SWIR-3', '1640-1680nm', '6xxx_n_n_A resolution reduced from 3.7m to 7.5m'),
    13 : ('SWIR', 'SWIR-4', '1710-1750nm', '6xxx_n_n_A resolution reduced from 3.7m to 7.5m'),
    14 : ('SWIR', 'SWIR-5', '2145-2185nm', '6xxx_n_n_A resolution reduced from 3.7m to 7.5m'),
    15 : ('SWIR', 'SWIR-6', '2185-2225nm', '6xxx_n_n_A resolution reduced from 3.7m to 7.5m'),
    16 : ('SWIR', 'SWIR-7', '2235-2285nm', '6xxx_n_n_A resolution reduced from 3.7m to 7.5m'),
    17 : ('SWIR', 'SWIR-8', '2295-2365nm', '6xxx_n_n_A resolution reduced from 3.7m to 7.5m') 
}



verbose = True
def progress(string = None) :
    if verbose :
        if string == 'done': 
            print(' done')
        elif string:
            print(string, end=' ')
        else :
            print('.', end='')
        sys.stdout.flush()


class WktPolygons:
    def __init__(self, dir=sourcedir, fname = train_wkt_fname) :
        fn = os.path.join(dir, fname)
        names = ("imageId", "classType", "wkt")
        data = pandas.read_csv(fn, names=names, skiprows=1)
        self._data = data
    
    def __getitem__(self, imageId) :
        td = self._data
        df_image = td[td.imageId == imageId]
        classPolygons = {}
        for ctype in classTypes:
            multipolygon = wkt.loads(df_image[df_image.classType == ctype].wkt.values[0])
            
            # At least one polygon in the training data is invalid. Fix (Kudos: amanbh)
            # https://www.kaggle.com/amanbh/dstl-satellite-imagery-feature-detection/dealing-with-invalid-polygons    
            if not multipolygon.is_valid : 
                multipolygon = multipolygon.buffer(0)
                progress('[Fixed invalid multipolygon {} {}]'.format(imageId, ctype) )
                 
            classPolygons[ctype] = multipolygon
            
        return classPolygons
        
    
    

def load_train_wkt(sdir = sourcedir, fname = train_wkt_fname) :  
    fn = os.path.join(sdir, fname)
    
    names = ("imageId", "classType", "wkt")
    data = pandas.read_csv(fn, names=names, skiprows=1)
    return data


def load_wkt_polygons(imageIds = None, sdir = sourcedir, fname = train_wkt_fname):
    if imageIds is None : imageIds = train_imageIds()

    polygons = {}
    td = load_train_wkt(sdir, fname) 
    
    for iid in imageIds:
        df_image = td[td.imageId == iid]
        classPolygons = {}
        for ctype in classTypes:
            multipolygon = wkt.loads(df_image[df_image.classType == ctype].wkt.values[0])
            
            # At least one polygon in the training data is invalid. Fix (Kudos: amanbh)
            # https://www.kaggle.com/amanbh/dstl-satellite-imagery-feature-detection/dealing-with-invalid-polygons    
            if not multipolygon.is_valid : 
                multipolygon = multipolygon.buffer(0)
                progress('Fixed invalid multipolygon {} {}'.format(iid, ctype) )
                 
            classPolygons[ctype] = multipolygon
            
        polygons[iid] = classPolygons
    return polygons




def load_sample_submission() :  
    names = ("imageId", "classType", "wkt")
    data = pandas.read_csv(sample_filename, names=names, skiprows=1)
    return data


class GridSizes(dict):
    def __init__(self, dir=sourcedir, fname='grid_sizes.csv'):
        fn = os.path.join(dir, fname)
    
        with open(fn) as csvfile:
            reader = csv.reader(csvfile) 
            next(reader)    # skip header row
            for row in reader :
                iid = row[0]
                xmax = float(row[1])
                ymin = float(row[2])
                self[iid] = (xmax, ymin)



def load_grid_sizes(dir=sourcedir, fname='grid_sizes.csv'):
    """Return a dictionary from imageIds to (xmax, ymin) tuples """  
    progress('[load_grid_sizes deprecated]')
    return GridSizes()
 
 #   gs = {}
 #   fn = os.path.join(dir, fname)
    
 #   with open(fn) as csvfile:
 #       reader = csv.reader(csvfile) 
 #       next(reader)    # skip header row
 #       for row in reader :
 #           iid = row[0]
 #           xmax = float(row[1])
 #           ymin = float(row[2])
 #           gs[iid] = (xmax, ymin)
    
 #   return gs
    
    
    
### Coordinate transforms ###
   
# Confusion reigns. We typicaly report image size as (width x height)
# Convention is that origin is top left.
# x-axis is width, negative y-axis is height
# But Image arrays are organized as height (rows) by width (columns)
# aspect ratio is width/height

# We'll use (x,y) for grid, (col, row) for raster
# (icol, irow) location within an image
# (icol, irow) location within a region

# Kaggle specifies an off-by-one correction for the gird to image coordiantes that makes no sense
#    width = 1.0*W*W/(W+1.) 
#    height = 1.0*H*H/(H+1.)
# But this is very nearly width = W-1, height = H-1, which does make sense. So we'll use that.


def grid_to_image_coords(coords, image_size = (std_width, std_height), grid_size = (std_xmax, std_ymin) ) :
    x, y = coords
    W, H = image_size
    xmax, ymin = grid_size  
    
    col = x * (W-1.) / xmax
    row = - y * (H-1.) / ymin
    
    return col, row
    
    
def image_to_grid_coords(coords, image_size = (std_width, std_height), grid_size = (std_xmax, std_ymin) ) :
    col, row = coords
    W, H = image_size
    xmax, ymin = grid_size
    
    x = col * xmax / (W-1.)
    y = - row * ymin / (H-1.)
 
    return x, y
 
 
     
def image_to_region_coords(coords, subregion, image_size=(std_width, std_height), border=regionborder) :
    icol, irow = coords
    reg_col, reg_row = subregion
    W, H = image_size
    
    rcol = icol + border + reg_col *W
    rrow = irow + border + reg_row *H
    
    return rcol, rrow
     
     
def region_to_image_coords(coords, subregion, image_size = (std_width, std_height), border=regionborder) :
    rcol, rrow = coords
    reg_col, reg_row = subregion
    W, H = image_size
    
    icol = rcol - border - reg_col * W
    irow  = rrow - border - reg_row * H
    
    return icol, irow
    




def image_size(imageId):
    """ Return the canonical (width, height) in pixels of a image region"""
    # Currently taking width and height from type '3' images
    img = load_image(imageId, '3')
    channels,  height, width = img.shape
    return (width, height)
   
       
def load_image(imageId, imageType):
    """Load a tiff image from input data
    imageType is one of 3, A, M or P
    """
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


def stretch_to_uint8(data, low, high):
    """ Stretch the dynamic range of an array to 8 bit data [0,255]
    Numbers outside range (low, high) are clipped to [0,255]
    """
    stretched = 255.* (data - low) / (high - low)    
    stretched[stretched<0] = 0
    stretched[stretched>255] = 255
    return stretched.astype(np.uint8)



low_percent_default=2
high_percent_default=98

def _dynamic_range(imageType, imageChannel, low_percent=low_percent_default, high_percent=high_percent_default) :
    """ Calculate the effective dynamic range of an image channel
    returns (low, high)
    """
    data = []
    for iid in train_imageIds() :
        d = load_image(iid, imageType)[imageChannel]
        data.append(d.flatten())
        
    data = np.concatenate(data)
    low, high = np.percentile(data, (low_percent, high_percent) ) 
    return (int(low), int(high) )


_imageRange = {}

def load_dynamic_range() :
    """ returns dictionary imageRange[imageType][imageChannel] -> (low, high)
    """    
    global _imageRange 
    if not _imageRange :    
        imageRange = {}
        for imageType in imageTypes : 
            imageRange[imageType] = [ 
                _dynamic_range(imageType, channel)
                for channel in range(0,imageChannels[imageType]) ]
        _imageRange = imageRange
    return _imageRange




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



  

def parse_viewId(viewId) :
    """ 
    Different views of the DSTL data are encoded as a string
        RRRR_(r_c|5x5)_T_CC   (e.g. '6010_2_1_P_00')
    
    RRRR    4 digit region as string (e.g. '6010')
    r       region row, as int between 0 and 4 (e.g. '6010_2_1_P_00' -> 2)
    c       region column, as int between 0 and 4 (e.g. '6010_2_1_P_00' -> 1)
    5x5     Entire 5x5 region
    T       Type character. (e.g. '6010_2_1_P_00' -> 'P')
                '3', 'P', 'M', 'A'  image types
                'C'                 categories
                'E'                 extended categories (not used yet)
                'N'                 Direct reference to dstl data (not implemented yet)
    CC      Two digit channel number, return as int (e.g. '6010_2_1_P_00' -> 0)
    
    """
    L = len(viewId)
    assert(L>=4)
    
    region    = viewId[0:4]
    
    if L>5 :
        row = viewId[5:6]    
        row = int(row) if row != 'X' else -1
        col = viewId[7:8]  
        col = int(col) if col != 'X' else -1
    else :
        row = None
        col = None
    
    if L>9 :
        imageType = viewId[9:10]   
    else :
        imageType = None
    
    if L>11 :
        channel = viewId[11:13]
        channel = int(channel) if channel != 'XX' else -1
    else : 
        channel = None       
    
    return region, row, col, imageType, channel
    

def compose_viewId(region, row=None, col=None, imageType=None, channel= None):
    assert(len(region) ==4)
    viewId =region
    if row is None : return viewId
    if row ==5 and col ==5: 
        viewId += '_5x5' 
    else :
        viewId += '_'+ str(row) + '_' + str(col)
    if imageType is None : return viewId
    viewId += '_' + imageType
    if channel is None : return viewId
    viewId += '_' + str(channel).zfill(2)
    return viewId
 
  
        
# ===== DEPRECATED ===== 


# Deprecated 
def imageId_to_region(imageid):
    """ Reads imageid such as "6020_1_2" and returns tuple regionid, region_x, region_y ("6020", 1, 2)"""
    sys.exit('Deprecated 2r')
    regionId, region_x, region_y = imageid.split('_')
    return ( regionId, int(region_x), int(region_y) )

# Deprecated 
def region_to_imageId(region, x,y) :
    sys.exit('Deprecated re')
    return region+'_'+str(x)+'_'+str(y)

# Deprecated     
def classId_to_imageId(classId):
    """
    classId -> (imageId, imageType, channelId)
    imageType and channel are optional, can return None
     e.g. 6020_1_2_P_10 -> ("6020_1_2", "P", "10")
    """
    sys.exit('Deprecated 3')
    imageId = classId[0:8]
    tail = classId[8:].split('_')
    if len(tail)==1 : return (imageId, None, None) 
    if len(tail)==2 : return (imageId, tail[1], None) 
    return (imageId, tail[1], tail[2])


# Deprecated 
def imageId_to_classId(imageId, imageType=None, channelId=None, extra=None):
    sys.exit('Deprecated to')
    classId = imageId
    if imageType : classId = classId + '_' + imageType
    if channelId : classId = classId + '_' + channelId
    if extra : classId = classId + '_' + extra
    return classId
 
# Deprecated 
def decompose_imageId(imageId) :
    sys.exit('Deprecated dec')
    region = imageId[0:4]     # 4 digits as string
    row = imageId[5:6]          # e.g., '1_2', '0_1', '5x5', ''
    col = imageId[7:8]          
    imageType = imageId[9:10]   # '3', 'P', 'M', 'C' '*'
    channel = imageId[11:]      # '' or integer or '*'
    
    return region, row, col, imageType, channel
 

        
 # DEPRECATED        
def filename(imageId, imageType=None, channelId=None, extra=None, ext='.png'):
    sys.exit('Deprecated fn')
    fn = imageId_to_classId(imageId, imageType, channelId, extra)
    fn += ext
    fn = os.path.join(output_dir, fn)
    return fn
  
  
    
    

# DEPRECATED
# To do : Add default_ to these filenames
input_dir = os.path.join('..','input')
output_dir = os.path.join('..','output')


data_filename = "dstl.hdf5"







#train_wkt_filename =  os.path.join(input_dir,"train_wkt_v4.csv")
grid_sizes_filename = os.path.join(input_dir,"grid_sizes.csv")
sample_filename = os.path.join(input_dir,"sample_submission.csv")
   

# ===== END DEPRECATED  =====

if __name__ == "__main__":
    import core_dstl
    help(core_dstl)

     