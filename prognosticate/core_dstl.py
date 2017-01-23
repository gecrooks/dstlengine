from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import csv
import random ; random.seed(42)
import tempfile
import sys

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

from descartes.patch import PolygonPatch

import tifffile as tiff

import PIL
from PIL import Image

import cv2

import numpy as np


"""
DSTL Prognostication Engine: Core

Core code for the Kaggle Dstl Satellite Imagery Feature Detection competition.
https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection


Authors: Gavin Crooks (@threeplusone) & Melissa Fabros (@mfab)

With contributions from Kagglers @amanbh, @FPP_UK, @shawn, @visoft, and others.

2017-01-18


== Installation ==
core_dstl was developed with these packages.
    python   	     3.5.2
    numpy    	     1.11.1
    matplotlib	     1.5.3 	( agg backend )
    tifffile  	     0.10.0
    shapely   	     1.6b2
    pandas    	     0.19.1
    rasterio  	     0.36.0
    h5py      	     2.6.0
    descartes	     ?.?.?
    pillow (PIL)	 3.4.2
    openCV (cv2)	 3.1.0
If you use different package versions, your mileage may vary.

To view your installed package versions:
    python -c 'import core_dstl; coredstl.package_versions()'

By default, cored_stl.py expects input data in '../input', and writes data to 'dstl.dat'

Run the unit tests:
    ./test_core_dstl.py



== Example command line usage ==

Build main data structures. (This will take a few hours, and use 90+ gigabytes of disk space)
    ./build.py all

Alternatively, build one region at a time
    ./build.py all 6100

Create PNG image of a 5km x 5km region, panchromatic band 
    ./image.py view 6100_5x5_P_0

View training subregion with overlaid target polygons
    ./image.py 6100_1_3_P_0 --composite

API help
    python -c 'import core_dstl; help(core_dstl)'

"""



# ================ Meta ====================

__description__ = 'DSTL Prognostication Engine'    
__version__ = '0.0.0'
__author__ = 'Gavin Crooks (@threeplusone) & Melissa Fabros (@mfab)'
__license__ = 'MIT'

# python -c 'import coredstl; coredstl.package_versions()'
def package_versions():
    print('coredstl    \t', __version__)
    print('python      \t', sys.version[0:5])
    print('numpy       \t', np.__version__)
    print('matplotlib  \t', matplotlib.__version__, 
        '\t(', matplotlib.get_backend(), 'backend )')
    print('tifffile    \t', tiff.__version__)
    print('shapely     \t', shapely.__version__)
    print('pandas      \t', pandas.__version__)
    print('rasterio    \t', rasterio.__version__)
    print('h5py        \t', h5py.__version__)
    print('descartes   \t', '?.?.?') # todo: is it possibly to get descartes version?
    print('pillow (PIL)\t', PIL.__version__)
    print('openCV (cv2)\t', cv2.__version__)



# ================ DSTL basic information ====================

class_nb = 10 # 1-10 

class_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

class_loc = {'C': (-1 , 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)}

# Todo class_name

# Adapted from @amanbh
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
 
# Adapted from @amanbh
class_color = {
    1 : '0.5',
    2 : '0.2',
    3 : '#b35806',
    4 : '#dfc27d',
    5 : '#1b7837',
    6 : '#a6dba0',
    #  7 : '#74add1',
    7 :  '#40a4df',            # Waterway 'Clear Water Blue'
    8 : '#4575b4',       
    9 : '#f46d43',
    10 : '#d73027',
    }

# Adapted from @amanbh        
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




# DSTL image bands to WorldView bands
#       If you look at the TIFF metadata you get the following info:
#       [3-bands] 6nnn_n_n: TIFFTAG_IMAGEDESCRIPTION={ bandlist = [ 4; 3; 2;] }
#       [16-bands] 6nnn_n_n_M: TIFFTAG_IMAGEDESCRIPTION={ bandlist = [ 6; 2; 3; 7; 4; 8; 5; 9;] }
#       [SWIR] 6nnn_n_n_A: TIFFTAG_IMAGEDESCRIPTION={ bandlist = [ 10; 11; 12; 13; 14; 15; 16; 17;]}
#   Kudos: @FPP_UK
#
# This means that the 3-band images are redundant with three of the Multispectral bands.
# However the corresponding images appear to be translated by a pixel or two.
# Apparently the polygons are registered to the 3-band images, so if we ignore any bands it should
# be the 2nd, 3rd and 5th Multispectral bands.
#
# For no apparently good reason the dstl Multispectral bands have been reordered with respect to 
# the WorldView bands.
#
# Based on the WorldView online bandwidth info and QGis extracted bandlist metadata and data 
# description
#
# Sensor-Bands :  WV3-Type, WV3-label, WV3range, Image-ID Notes
# Kudos: @FPP_UK
band_info = {
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


# Total number of bands ( 16-band + 3-band + Pancromatic)
band_nb = 20

# number of bands in each image type.
nb_channels = {'3':3, 'A':8, 'M':8, 'P':1}

# Images cover 1km x 1km subregions, and come in 4 types.
band_types = ('3', 'A', 'M', 'P')

band_slice = { 
    '3' : slice(0,3),
    'P' : slice(3,4),
    'M' : slice(4,12), 
    'A' : slice(12,20), 
    'W' : slice(3,20),
    }


# Codes for feature types. Includes 'C' for target categories.
feature_types = ['3', 'A', 'M', 'P', 'W', 'C']


# Descriptive names of the different band and feature types
feature_names = {
    '3' : '3-band',
    'P' : 'Panchromatic',
    'M' : 'Multispectral',
    'A' : 'SWIR',
    'W' : 'WorldView',
    'C' : 'Classes',
    }



# ================ DSTL Prognostication engine basic information ====================


# Canonical width and height of each image. 
# The '3' band images are always almost this size, within a few pixels,
# except the right and bottom boundary images (of a 5x5 region), 
# which can be cropped smaller.
# So we pad to 0_0 size of same region, then scale to standard size
std_height = 3348
std_width = 3396

max_hegiht = 3350
max_width = 3403

# Minimum border of zero data around valid data in a region
region_border = 256


# Height and width of a 5x5 region in pixels
# Assumes canonical image sizes, and a border on each side.
region_width = 5 * std_width + 2 * region_border
region_height = 5 * std_height + 2 * region_border

# fixme
# The resolution used for the input wkt polygons 
grid_resolution = 1./1000000 



# ================ DSTL routines ====================

# Coordinate transforms 
#
# We have 4 principle coordinates. 
#  The grid location (x,y) in the scale used by the DSTL training polygons
#  The pixel location (icol, irow) within a 1km x 1km subregion
#  The pixel location (rcol, rrow) within a 5km x 5km region
#  The 1km x 1km subregion location (reg_col, reg_row) within a 5km x 5km region
#   
# Note that we report image size as (width x height)
# Convention is that origin is top left.
# x-axis is width, negative y-axis is height
# But Image arrays are organized as height (rows) by width (columns)
# aspect ratio is width/height
#
# We'll use (x,y) for grid, (col, row) for raster
# (icol, irow) location within an image
# (rcol, rrow) location within a region
# (subcol, subrow) for row and column of a 1km x 1km block within a 5km x 5km region 
# 
# Kaggle specifies an off-by-one correction for the grid to image coordinates that makes no sense
#    width = 1.0*W*W/(W+1.) 
#    height = 1.0*H*H/(H+1.)
# But this is very nearly width = W-1, height = H-1, [Taylor series expand about 0 wrt 1/W, width = W - 1 + O(1/W) ]
# and this alternative correction does make sense. Suppose the image is 100 pixels wide, and xmax= 1.0
# Then the last pixel is icol=99 which should correspond to x=1.0

def grid_to_image_coords(coords, grid_size, image_size = (std_width, std_height)):
    x, y = coords
    W, H = image_size
    xmax, ymin = grid_size  
    
    col = x * (W-1.) / xmax
    row = - y * (H-1.) / ymin
    
    return col, row
    
    
def image_to_grid_coords(coords, grid_size, image_size = (std_width, std_height) ):
    icol, irow = coords
    W, H = image_size
    xmax, ymin = grid_size
    
    x = icol * xmax / (W-1.)
    y = - irow * ymin / (H-1.)
 
    return x, y
 
 
     
def image_to_region_coords(coords, subregion, image_size=(std_width, std_height), border=region_border):
    icol, irow = coords
    subcol, subrow = subregion
    W, H = image_size
    
    rcol = icol + border + subcol * W
    rrow = irow + border + subrow * H
    
    return rcol, rrow
     
     
     
def region_to_image_coords(coords, subregion, image_size = (std_width, std_height), border=region_border):
    rcol, rrow = coords
    subcol, subrow = subregion
    W, H = image_size
    
    icol = rcol - border - subcol * W
    irow  = rrow - border - subrow * H
    
    return icol, irow
    



def parse_viewId(viewId):
    """ 
    Different views of the DSTL data are encoded as a string
        RRRR_(r_c|5x5)_T_CC   (e.g. '6010_2_1_P_0)
    
    RRRR    4 digit region as string (e.g. '6010')
    r       region row, as int between 0 and 4 (e.g. '6010_2_1_P_0' -> 2)
    c       region column, as int between 0 and 4 (e.g. '6010_2_1_P_0' -> 1)
    5x5     Entire 5x5 region
    T       Type character. (e.g. '6010_2_1_P_00' -> 'P')
                '3', 'P', 'M', 'A'  image types
                'W'                 WorldView bands 
    CC      One or two digit band number, return as int (e.g. '6010_2_1_P_0' -> 0)
    
    Everything after the region ID is optional. To parse an ImageId
    region, subrow, subcol  = parse_viewId('6010_1_2')[0:3]       
    
    """
    L = len(viewId)
    assert(L>=4)
    
    region    = viewId[0:4]
    
    if L>5:
        row = viewId[5:6]    
        row = int(row) if row != 'X' else -1
        col = viewId[7:8]  
        col = int(col) if col != 'X' else -1
    else:
        row = None
        col = None
    
    if L>9 :
        imageType = viewId[9:10]   
    else:
        imageType = None
    
    if L>11 :
        channel = viewId[11:13]
        channel = int(channel) if channel != 'XX' else -1
    else : 
        channel = None       
    
    return region, row, col, imageType, channel
    

def compose_viewId(region, row=None, col=None, imageType=None, channel= None):
    assert(len(region) == 4)
    viewId =region
    if row is None : 
        return viewId
    if row == 5 and col == 5: 
        viewId += '_5x5' 
    else :
        viewId += '_'+ str(row) + '_' + str(col)
    if imageType is None : return viewId
    viewId += '_' + imageType
    if channel is None : return viewId
    viewId += '_' + str(channel)
    return viewId
 
 
 
# ================ File loading and parsing ====================
  
def load_wkt(filename) :  
    """ Load a WKT polygon CSV file""" 
    names = ("imageId", "classType", "wkt")
    data = pandas.read_csv(filename, names=names, skiprows=1)
    return data

def wkt_polygons(data, imageId) :
    df_image = data[data.imageId == imageId]
    classPolygons = {}
    for ctype in class_types:
        multipolygon = wkt.loads(df_image[df_image.classType == ctype].wkt.values[0])
            
        # At least one polygon in the training data is invalid. Fix (Kudos: @amanbh)
        if not multipolygon.is_valid : 
            multipolygon = multipolygon.buffer(0)
            #_progress('Fixed invalid multipolygon {} {}'.format(iid, ctype))
                 
        classPolygons[ctype] = multipolygon
    return classPolygons
    


# ================ DSTL data ====================
#FIXME: don't need to be private

# Default sourcedir
_SOURCEDIR = os.path.join('..','input')

#FIXME: don't need to be private

# Default output directory
_DATADIR = os.path.join('..','dstl.dat')



# Either "gzip" (moderate speed, good compression), 'lzf' (very fast, moderate compression), or None
_HDF5_COMPRESSION = 'gzip'

# Chunksize for hdf data. True for autochunking
_HDF5_CHUNKS = True 

# Default verbosity
_VERBOSE = True

# used when estimating image effective dynamic range
_LOW_PERCENT=2
_HIGH_PERCENT=98

# default dots per inch when creating images. 
_DPI = 256   

# Default scaling for generating images of entire regions
_REGION_SCALE = 0.2

# Default scaling used when generating images of subregions
_SUBREGION_SCALE = 1.0

# Image resampling method when resizing: Image.BICUBIC or Image.LANCZOS
_RESAMPLE = Image.BICUBIC

# Interpolation method when resizing with cv2: cv2.INTER_CUBIC or cv2.INTER_LANCZOS4
_INTERPOLATION = cv2.INTER_CUBIC          

# Mode used to find alignments between images
# cv2.MOTION_TRANSLATION    -- faster and seems to be sufficent for the job
# cv2.MOTION_AFFINE
_WARP_MODE = cv2.MOTION_TRANSLATION


# Datatype used to store band data
# np.uint8 for compact storage, or np.uint16 to match source data
band_dtype = np.uint8


# Datatype used to store class masks
class_dtype = np.uint8


# Paths to various input and output files
_PATHS = {
    # source files
    'train_wkt'     : '{sourcedir}{sep}train_wkt_v4.csv',
    'grid_sizes'    : '{sourcedir}{sep}grid_sizes.csv',
    'sample'        : '{sourcedir}{sep}sample_submission.csv',     
    'three_band'    : '{sourcedir}{sep}three_band{sep}{imageId}.tif',
    'sixteen_band'  : '{sourcedir}{sep}sixteen_band{sep}{imageId}_{imageType}.tif',

    # output files
    'data'          : '{datadir}{sep}dstl.hdf5',                 # Main datafile
    'composite'     : '{datadir}{sep}{imageId}_composite.png',
    'image'         : '{datadir}{sep}{imageId}.png',
    
}







class Dstl(object):
    """Access and analysis of the dstl dataset.""" 
    def __init__(self, sourcedir=_SOURCEDIR, datadir=_DATADIR, verbose=_VERBOSE):
        self.sourcedir = sourcedir
        self.datadir = datadir
        self.verbose = verbose
        
        self._datafile = None
        self._wkt_data = None

        # Dictionary mapping image type to dynamic range (low, high)
        # These are defaults. Reload with dstl.load_dynamic_range
        self.dynamic_range = {'3': (157, 765), 'A': (539, 7381), 'M': (203, 914), 'P': (265, 797)}
        

    # Define __enter__ and __exit__ so can use 
    # with Dstl(...) as dstl:
    #   dstl.do_something()
    
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        
      
    def open(self):  
        dfpath = self.path('data')
        if not os.path.exists(dfpath): 
            self._initialize()
        self._datafile = h5py.File(dfpath, 'r+') # open for read/write 
        return self._datafile
             
    def close(self) :
        if self._datafile is not None :
            self._datafile.close()
            self._datafile = None
        
    
    def data(self, region) :
        datafile  = self.open()
        datagroup = datafile.require_group('data')  # FIXME
        return datagroup[region]
    
    def targets(self, region) :
        datafile  = self.open()
        datagroup = datafile.require_group('targets')   # FIXME
        dataset    = datagroup[region]
        return dataset
        
    
    def _progress(self, msg=None, end=' ') :
        _progress(msg, end, self.verbose)


    def path(self, name, **kwargs) :
        """Return path to various named input and output files"""
        kwargs['sep'] = os.sep
        kwargs['sourcedir'] = self.sourcedir
        kwargs['datadir'] = self.datadir
        path = _PATHS[name]
        path = path.format(**kwargs)
        return path


    @property
    def grid_sizes(self):
        """Return a dictionary from imageIds to (xmax, ymin) tuples """  
        fn = self.path('grid_sizes')
        gs = {}
        with open(fn) as csvfile:
            reader = csv.reader(csvfile) 
            next(reader)    # skip header row
            for row in reader :
                iid = row[0]
                xmax = float(row[1])
                ymin = float(row[2])
                gs[iid] = (xmax, ymin)
        return gs
    
    
    def wkt_polygons(self, imageId):
        if self._wkt_data is None :
            self._wkt_data= load_wkt( self.path('train_wkt'))            
        return wkt_polygons( self._wkt_data, imageId)
 
 
 
    def load_image(self, imageId, imageType):
        """Load a tiff image from input data
        imageType is one of 3, A, M or P
        """
        if imageType == '3' :
            fn = self.path('three_band', imageId=imageId, imageType=imageType)
        else :
            fn = self.path('sixteen_band', imageId=imageId, imageType=imageType)
            
        image = tiff.imread(fn)
         
        if imageType == 'P' : 
            # P images single channel greyscale. But add extra dimension (length 1) for consistency
            image = np.expand_dims(image, axis=-1)
        else : 
            # tiff files are channels x height x width. Change to height x width x channels
            image = np.transpose(image, (1, 2, 0))
        
        return image



    def image_size(self,imageId):
        """ Return the canonical (width, height) in pixels of a subregion"""
        # Currently taking width and height from type '3' images
        img = self.load_image(imageId, '3')
        height, width, bands = img.shape
        return (width, height)

      
    
    @property
    def imageIds(self) :
        """ return A list of all imageIds """
        return sorted( self.grid_sizes.keys())



    # Return a list of all training data image ids
    @property
    def train_imageIds(self) :
        """A sorted list of all training data image ids"""
        fn = self.path('train_wkt')
        return sorted(load_wkt(fn).imageId.unique())


    
    @property
    def test_imageIds(self):
        """ A list of all test image ids, in same order as 'sample_submission.csv'.
        Note that a few test ids were latter added to the train ids. We still need to return
        dummpy predictons for those ids, but they wont be scored.
        """        
        
        fn = self.path('sample')
        names = ("imageId", "classType", "wkt")
        data = pandas.read_csv(fn, names=names, skiprows=1)
        
        return data.imageId.unique()
  


    @property
    def regionIds(self):
        """A sorted list of all 4 digit region ids."""
        return sorted(set( [iid[0:4] for iid in self.imageIds]))
    

    # FIXME: docs. Test
    def load_dynamic_range(self):
        dynamic_range = {}
        for itype in band_types : 
                data = []
                for iid in self.train_imageIds :
                    img = self.load_image(iid, itype)
                    data.append(img.flatten())
                data = np.concatenate(data)
                low, high = np.percentile(data, (_LOW_PERCENT, _HIGH_PERCENT) ) 
                dynamic_range[itype] = (int(low), int(high) )
        
        self.dynamic_range = dynamic_range

    

    
    # ---------- Dstl munge data ----------
    
 
    def _initialize(self):
        self._progress('Initializing data structures...', end='\n')   
        
        if not os.path.exists(self.datadir): os.makedirs(self.datadir)
            
        # Create hdf5 file, fail if it exists.
        datafile = h5py.File( self.path('data'), 'x', libver='latest')
    
        datagroup = datafile.require_group('data') 
        for region in self.regionIds:
            dataset   = datagroup.require_dataset(region, 
                           (region_height, region_width, band_nb), 
                           dtype = band_dtype, 
                           chunks = _HDF5_CHUNKS,
                           compression = _HDF5_COMPRESSION,)
   
        datagroup = datafile.require_group('targets') 
        for region in self.regionIds:
            dataset = datagroup.create_dataset(region, 
                        (region_height, region_width, class_nb), 
                        maxshape=(region_height, region_width, None),
                        dtype = class_dtype, 
                        chunks = _HDF5_CHUNKS,
                        compression = _HDF5_COMPRESSION,)
        
        self._progress('done')   
        
    def build(self, regions = None):
        """Build everything. (build_data, build_alignment, build_targets, build_composites)"""
        self.build_data(regions)
        self.build_alignment(regions)
        self.build_targets(regions)
        self.build_composites(regions)
     
        
    
    def build_data(self, regions = None, imageType = None) :   
        self._progress('Building data structures...', end='\n')   
        
        if not regions : regions = self.regionIds  
        
        if imageType is None or imageType == 'all' :
            itypes = band_types  
        else:
            itypes = [imageType,]
                
   
        for region in regions:
            dataset = self.data(region)
            
            # Fill background with (near) average values
            self._progress('    {} init'.format(region) )
            for itype in itypes :
                low, high =  self.dynamic_range[itype]
                avg = (low + high)//2
                bands  = band_slice[itype]
                dataset[:,:,bands].fill(avg) 
                self._progress()
            self._progress('done')
            
            
            # Loop over all subregions in 5x5 region    
            for reg_row in range(0,5): 
                for reg_col in range (0,5): # FIXME 5 magic constant
                    iid = compose_viewId(region, reg_row, reg_col)              

                    self._progress('    '+iid)
                    for itype in itypes : 
                        self._progress()
                        
                        data = self.load_image(iid, itype)

                        # If a right or bottom edge image, pad to size of _0_0 image
                        iid00 = compose_viewId(region, 0, 0)  
                        h00, w00, f00  = self.load_image(iid00, itype).shape
                        h, w, f= data.shape
                        if h != h00 or w != w00 :
                            padded_data = np.zeros( (h00, w00, f00), dtype =band_dtype)
                            padded_data[:h, :w, :] = data
                            data = padded_data
                                 
                        # Rescale all images to standard image size
                        img = data.astype(np.float32)  
                        img = cv2.resize(img, (std_width, std_height),  interpolation= _INTERPOLATION)   
                        if itype == 'P' : img = np.expand_dims(img, axis=-1) # Put back last axis
   
                        # Stretch dynamic range
                        low, high =  self.dynamic_range[itype]
                        img = _stretch(img, low, high, dtype = band_dtype)
  
                        # Save image data to correct location in region data
                        rcol, rrow = image_to_region_coords( (0,0), (reg_col, reg_row))
                        rcol_stop = rcol + std_width
                        rrow_stop = rrow + std_height
                        bands = band_slice[itype]
                        dataset[rrow : rrow_stop, rcol : rcol_stop, bands] = img  
                                              
                    self._progress('done') # finished subregion
               
        self._progress('done')  # finished all regions
        
    # ---------- End Dstl build_data ----------
 
 
    def build_alignment( self,
               regions = None,
               imageType = None,
               dry_run = False,
               ) :    
        """Register A, M and P images to 3-band images"""
        self._progress('Re-registering images...', end='\n') 
        
        if not regions :
            regions = self.regionIds

        if not imageType or imageType == 'all': 
            imageTypes = ('A', 'M', 'P')
        else:
            imageTypes = [imageType,]

        window_size = 4096   # FIXME magic constant
        start = region_height //2 - window_size //2
        stop = start + window_size


        for region in regions:     
            dataset = self.data(region)
            img_3 = dataset[start:stop, start:stop, band_slice['3'] ].astype(np.float32).mean(axis=-1)         
               
            for itype in imageTypes:  
                self._progress('    registering {} {}'.format(region, itype))
               
                bands = band_slice[itype]
                img_avg = dataset[start:stop, start:stop, bands].astype(np.float32).mean(axis=-1)
                
                warp_matrix = np.eye(2,3, dtype=np.float32) # 'eye' is I, identity matrix
   
                try:
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,  1e-5)
                    (cc, warp_matrix) = cv2.findTransformECC (img_3, img_avg, warp_matrix, _WARP_MODE, criteria)
                except cv2.error:
                    print('findTransformEEC Failed to converge: {}_5x5_{}'.format(region, itype))    
                    # if it fails, let it go.
                    return
   
                self._progress("cc: {:.2f} warp_matrix:".format(cc))
                self._progress("".join(str(warp_matrix).split("\n")))

                if not dry_run:
                    img2 = dataset[:, :, bands ]
                    if img2.shape[2]==1 : # fix for warpAffine dropping last axis when length 1
                        bands = bands.start    
                    dataset[:, :, bands]  = cv2.warpAffine(
                                                img2,
                                                warp_matrix,
                                                (img2.shape[1], img2.shape[0]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
 
                self._progress('done')
        self._progress('done') 
        
    # ---------- End Dstl build_alignment ----------
    
    
    
    
    
    def build_targets(self, regions = None):
        """Create target masks from the raw training data"""
        self._progress('Building category targets...', end='\n')
        self._progress('   Loading polygons...')
        wkt= load_wkt( self.path('train_wkt'))
        self._progress('done')
    
        if not regions: regions = self.regionIds
        gs = self.grid_sizes
        
        for region in regions:
            for iid in self.train_imageIds :
                reg, reg_row, reg_col = parse_viewId(iid)[0:3]
                if reg != region: continue 
                self._progress('    '+iid)
                
                class_polygons = self.wkt_polygons(iid)
                self._progress()
                
                dataset = self.targets(region)
                xmax, ymin = gs[iid]
                
                for ct in class_types:
                    polygons = class_polygons[ct]
                    
                    mask = polygons_to_mask(polygons, xmax, ymin, std_width, std_height)       
                    loc = class_loc['C'][ct] 
                    rcol, rrow = image_to_region_coords( (0,0), (reg_col, reg_row))
                    dataset[rrow:rrow+std_height, rcol:rcol+std_width, loc] = mask*255.
                
                    self._progress()
                self._progress('done')
                            
        self._progress('done')

 

    def build_composites(self, regions=None, outline = True):
        self._progress('Building target composites...', end ='\n    ')
    
        regions = regions if regions else self.regionIds
        #if not regions: regions = self.regionIds
        gs = self.grid_sizes
        
        for iid in self.train_imageIds :
            reg = parse_viewId(iid)[0]
            if reg not in regions: continue
            self._progress('    '+iid)
            
            xmax, ymin = gs[iid]
            class_polygons = self.wkt_polygons(iid)
            fn = self.path('composite', imageId = iid)
            polygons_to_composite(class_polygons, xmax, ymin, std_width, std_height, fn, outline)          
            
            self._progress('done')
        
        self._progress('done')


  

    def view(self, imageIds, scale=None, composite=False) : 
        """ Build png images of regions or subregions from data """      
        for iid in imageIds:
            region, reg_row, reg_col, image_type, channel = parse_viewId(iid)
        
            if reg_row is None: reg_row =5
            if reg_col is None: reg_col =5
        
            if image_type is None or image_type == '*': 
                itypes = band_types
            else:
                itypes = [image_type,]
        
            for itype in itypes:
                if channel is None or channel == '*' : 
                    channels = range(0, nb_channels[itype])
                else:
                    channels = [channel,]
                
                for chan in channels:
                    viewId = compose_viewId(region, reg_row, reg_col, itype, chan)
                    fn = self.path('image', imageId = viewId)
                    self._progress('creating '+ fn)
                
                    img = self.image(viewId, scale, composite) 
                    img.save(fn)  
                
                    self._progress('done')
 

    def image(self, viewId, scale=None, composite=False) :
        """Return a single png image. Expects fully specified imageId  """
        
        region, reg_row, reg_col, imageType, channel = parse_viewId(viewId)
  
        if channel is None: channel = 0
               
        if imageType == 'C' :
            dataset = self.targets(region)
            if channel<1 or channel>10: raise ValueError('Class channel out of range')
            loc = channel -1
        else :
            dataset = self.data(region)
            loc = band_slice[imageType].start + channel

        if reg_row == 5 and reg_col==5 :
            # whole 5km x 5km region
            rcol, rrow = image_to_region_coords( (0, 0), (0, 0))
            rcol_stop = rcol + 5*std_width
            rrow_stop = rrow + 5*std_height
            if not scale : scale = _REGION_SCALE
        else :
            # Single 1km by 1km view
            rcol, rrow = image_to_region_coords( (0, 0), (reg_col, reg_row))
            rcol_stop = rcol + std_width
            rrow_stop = rrow + std_height
            
            if not scale : scale = _SUBREGION_SCALE
  
  
        data = dataset[rrow : rrow_stop, rcol : rcol_stop, loc]
        dmin = np.iinfo(data.dtype).min
        dmax = np.iinfo(data.dtype).max
        
        data =  _stretch(data, dmin, dmax, dtype=np.uint8)
        
        #low, high = np.percentile(data, (_LOW_PERCENT, _HIGH_PERCENT)) #FIXME 

        #data = _stretch_to_uint8(data, low, high)
        img = Image.fromarray(data) 

        if composite :
            self._progress('(compositing)')
            fn = self.path('composite', imageId = viewId[0:8])
            img1 = img.convert(mode = 'RGBA')
            img2 = Image.open(fn)
            img = Image.alpha_composite(img1, img2)

        if scale != 1.0:
            width, height = img.size
            width = int(round( width*scale))
            height = int(round( height* scale))       
            img = img.resize( ( width, height) , resample = _RESAMPLE)
        return img
    


# ================ Polygon routines ====================

def polygons_to_composite(class_polygons, xmax, ymin, width, height, filename, outline=True) :
     """ If outline is true, create transparent outline of classes suitable for layering over other images."""
     width /= 1.*_DPI
     height /= 1.*_DPI
     
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
     
     plt.savefig(filename, pad_inches=0, dpi=_DPI, transparent=transparent)
     plt.clf()
     plt.close()    



def polygons_to_mask(multipolygon, xmax, ymin, width, height, filename=None) :
     img_width = 1.* width / _DPI       # In inches (!)
     img_height = 1.* height / _DPI    
     fig = plt.figure(figsize=(img_width,img_height), frameon=False)
     axes = plt.Axes(fig, [0., 0, 1, 1]) # One axis, many axes
     axes.set_axis_off()         
     fig.add_axes(axes)  
              
     for polygon in multipolygon:
         patch = PolygonPatch(polygon,
                             color='#000000',
                             lw=0,               # linewidth
                             antialiased = True)
         axes.add_patch(patch)
     axes.set_xlim(0, xmax * width/(width-1))
     axes.set_ylim(ymin * height/(height-1) , 0)
     
     #axes.set_aspect(1)
     plt.axis('off')
    
     if filename is None :
         filename = tempfile.NamedTemporaryFile(suffix='.png')
     plt.savefig(filename, pad_inches=0, dpi=_DPI, transparent=False)
     plt.clf()
     plt.close()
     a = np.asarray(Image.open(filename))
     a = (1.- a[:,:,0]/255.)  # convert from B&W to zeros and ones.
     return a    



# Adapted from code by '@shawn'
def mask_to_polygons(mask, xmax, ymin, threshold=0.4):
     all_polygons=[]
    
     mask[mask >= threshold] = 1
     mask[mask < threshold] = 0
    
     for shape, value in rasterio.features.shapes(mask.astype(np.int16),
                                 mask = (mask==1),
                                 transform = rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):

         all_polygons.append(shapely.geometry.shape(shape))

     all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    
     # simplify the geometry of the masks
     # FIXME: this seems to be in wrong place. working in pixel coordinates here, but rounds anyways!?
     all_polygons = all_polygons.simplify(grid_resolution, preserve_topology=False)
       
     # Transform from pixel coordinates to grid coordinates
     height, width = mask.shape 
     all_polygons = shapely.affinity.scale(all_polygons, xfact = xmax/(width-1), yfact = ymin/(height-1), origin=(0,0,0))
      
     if not all_polygons.is_valid:
         all_polygons = all_polygons.buffer(0)
     
     # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
     # need to keep it a Multi throughout
     if all_polygons.type == 'Polygon':
        all_polygons = shapely.geometry.MultiPolygon([all_polygons])    
    
     return all_polygons
     
        
        
def polygon_jaccard(actual_polygons, predicted_polygons) :
    """
    Calculate the Jaccard similiarity index between two sets of polygons
    Returns jaccard, true_positive, false_positive, false_negative
    """
    true_positive  = predicted_polygons.intersection(actual_polygons).area
    false_positive = predicted_polygons.area - true_positive
    false_negative = actual_polygons.area - true_positive    
    union = (true_positive+false_positive+false_negative)
    
    epsilon = 10e-16
    jaccard = (true_positive + epsilon) / (union + epsilon)
    
    return jaccard, true_positive, false_positive, false_negative
    
     

# ================ Utility routines ====================

# FIXME: no reason to be private

# Round up to next size
def _roundup(x, size) : return ( (x+size-1) // size) * size

# Round down to next size
def _rounddown(x, size) : return _roundup(x, size) - size


# Print progress reports to stdout if verbosity is True.
# FIXME: move to dstl

def _progress(string = None, end = ' ', verbose=_VERBOSE):
    if verbose :
        if string == 'done': 
            print(' done')
        elif string:
            print(string, end=end)
        else :
            print('.', end='')
        sys.stdout.flush()


def _stretch(data, low, high, dtype = np.uint8):
    """ Stretch the dynamic range of an array to range of dtype.
    Values outside of [low, high] are clipped.
    """
    dmin = np.iinfo(dtype).min
    dmax = np.iinfo(dtype).max

    data = data.astype(float)
    stretched = dmax* (data - low) / (high - low)    
    stretched[stretched<dmin] = dmin
    stretched[stretched>dmax] = dmax
    return stretched.astype(dtype)   

# FIXME Deprecated. Use _stretch
def _stretch_to_uint8(data, low, high):
    """ Stretch the dynamic range of an array to 8 bit data [0,255]
    Numbers outside range (low, high) are clipped to [0,255]
    """
    data = data.astype(float)
    stretched = 255.* (data - low) / (high - low)    
    stretched[stretched<0] = 0
    stretched[stretched>255] = 255
    return stretched.astype(np.uint8)

# End