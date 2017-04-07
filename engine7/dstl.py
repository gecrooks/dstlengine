#!/usr/bin/env python

"""DSTL Prognostication Engine: DSTL Core"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import json
import sys
import argparse
import random ; random.seed(42)
import csv ; csv.field_size_limit(sys.maxsize)
import tempfile

import numpy as np


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tifffile as tiff

import shapely
import shapely.geometry
import shapely.affinity
import shapely.wkt

import pandas

import rasterio
import rasterio.features

from descartes.patch import PolygonPatch

import PIL
from PIL import Image

import cv2


# ================ Meta ====================
__description__ = 'DSTL Prognostication Engine: DSTL Core'
__version__ = '0.7.0'
__license__ = 'MIT'
__author__ = 'Gavin Crooks (@threeplusone) & Melissa Fabros (@mfab)'
__status__ = "Prototype"
__copyright__ = "Copyright 2017"

# python -c 'import dstl.py; dstl.package_versions()'
def package_versions():
    print('dstl        \t', __version__)
    print('python      \t', sys.version[0:5])
    print('numpy       \t', np.__version__)
    print('matplotlib  \t', matplotlib.__version__,
          '\t(', matplotlib.get_backend(), 'backend )')
    print('tifffile    \t', tiff.__version__)
    print('shapely     \t', shapely.__version__)
    print('pandas      \t', pandas.__version__)
    print('rasterio    \t', rasterio.__version__)
    print('descartes   \t', '?.?.?') # todo: is it possibly to get descartes version?
    print('pillow (PIL)\t', PIL.__version__)
    print('openCV (cv2)\t', cv2.__version__)


SOURCEDIR = os.path.join('..', 'input')

# Default output directory. Location for generated files and models.
OUTPUTDIR = 'output'

SCALE = 4

LOW_PERCENT = 0.5
HIGH_PERCENT = 99.5

INTERPOLATION = cv2.INTER_CUBIC

# default dots per inch when creating images. Nice round number.
DPI = 256

ALPHA = 0.8

# Min size (3335, 3345)
# Max size (3403, 3350)

SUBREGION_MAX_WIDTH = 3600 * 5   # FIXME: Change to 3408
SUBREGION_MAX_HEIGHT = 3600 * 5  # FIXME: Change to 3408
REGION_BORDER = 512


# ==================== Utility routines ====================

# Round up to next sizes
def roundup(x, size):
    return ((x+size-1) // size) * size

# Round down to next size
def rounddown(x, size):
    return roundup(x-size+1, size)

def stretch(data, low, high, dtype=np.uint8):
    """ Stretch the dynamic range of an array to range of dtype.
    Values outside of [low, high] are clipped.
    """
    dmin = np.iinfo(dtype).min
    dmax = np.iinfo(dtype).max

    data = data.astype(float)
    stretched = dmax* (data - low) / (high - low)
    stretched[stretched < dmin] = dmin
    stretched[stretched > dmax] = dmax
    return stretched.astype(dtype)


def progress(string=None, end=' ', verbose=True):
    if verbose:
        if string == 'done':
            print(' done')
        elif string is None:
            print('.', end='')
        else:
            print(string, end=end)
        sys.stdout.flush()


# ==================== viewId routines ====================


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
    region, subrow, subcol = parse_viewId('6010_1_2')[0:3]
    """
    L = len(viewId)
    assert L >= 4

    region = viewId[0:4]
    if region == 'XXXX': region = None

    if L > 5:
        row = viewId[5:6]
        row = int(row) if row != 'X' else None
        col = viewId[7:8]
        col = int(col) if col != 'X' else None
    else:
        row = None
        col = None

    if L > 9:
        imageType = viewId[9:10]
        if imageType == 'X': imageType = None
    else:
        imageType = None

    if L > 11:
        channel = viewId[11:13]
        channel = int(channel) if channel != 'X' else None
    else:
        channel = None

    return region, row, col, imageType, channel


def compose_viewId(region, row=None, col=None, imageType=None, channel=None):
    assert len(region) == 4
    viewId = region
    if row is None:
        return viewId
    if row == 5 and col == 5:
        viewId += '_5x5'
    else:
        viewId += '_'+ str(row) + '_' + str(col)
    if imageType is None: return viewId
    viewId += '_' + imageType
    if channel is None: return viewId
    viewId += '_' + str(channel)
    return viewId

def iterate_viewIds(viewId):
    region, row, col, itype, channel = parse_viewId(viewId)
    info = DstlInfo()
    regions = info.regionIds if region is None else [region,]
    rows = [0, 1, 2, 3, 4] if row is None else [row,]
    cols = [0, 1, 2, 3, 4] if col is None else [col,]
    itypes = info.band_types if itype is None else [itype,]

    for reg in regions:
        for r in rows:
            for c in cols:
                for it in itypes:
                    channels = range(0, info.channel_nb[it]) if channel is None else [channel,]
                    for chan in channels:
                        yield compose_viewId(reg, r, c, it, chan)



# ==================== DstlInfo ====================


class DstlInfo(object):
    """ Access to raw DSTL data  """

    def __init__(self, sourcedir=SOURCEDIR):
        self._wkt = None
        self._grid_sizes = None

        self.sourcedir = sourcedir

        self.source_paths = {
            'train_wkt'     : os.path.join('{sourcedir}', 'train_wkt_v4.csv'),
            'grid_sizes'    : os.path.join('{sourcedir}', 'grid_sizes.csv'),
            'sample'        : os.path.join('{sourcedir}', 'sample_submission.csv'),
            'three_band'    : os.path.join('{sourcedir}', 'three_band'),
            '3'             : os.path.join('{sourcedir}', 'three_band', '{imageId}.tif'),
            'sixteen_band'  : os.path.join('{sourcedir}', 'sixteen_band'),
            'A'             : os.path.join('{sourcedir}', 'sixteen_band', '{imageId}_A.tif'),
            'M'             : os.path.join('{sourcedir}', 'sixteen_band', '{imageId}_M.tif'),
            'P'             : os.path.join('{sourcedir}', 'sixteen_band', '{imageId}_P.tif'),
            }

        self.class_nb = 10 # 1-10

        self.class_types = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

        self.class_index = {'1':0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8, '10':9}


        # Classes are numbered 1-10
        # Adapted from @amanbh
        self.class_shortname = {
            '1' : 'Building',
            '2' : 'Structure',
            '3' : 'Road',
            '4' : 'Track',
            '5' : 'Trees',
            '6' : 'Crops',
            '7' : 'Waterway',
            '8' : 'Lake',
            '9' : 'Truck',
            '10' : 'Car',
            }



        # Adapted from @amanbh
        self.class_color = {
            '1' : '0.7',
            '2' : '0.2',
            '3' : '#b35806',
            '4' : '#dfc27d',
            '5' : '#1b7837',
            '6' : '#a6dba0',
            '7' : '#40a4df',        # Waterway  Clear Water Blue
            '8' : '#191970',        # Lake      Navy
            '9' : '#f46d43',        # Trucks
            '10': '#FF0000',        # Cars      Red
            }

        # Adapted from @amanbh
        self.class_zorder = {
            '1' : 5,
            '2' : 6,
            '3' : 4,
            '4' : 1,
            '5' : 3,
            '6' : 2,
            '7' : 7,
            '8' : 8,
            '9' : 9,
            '10': 10,
            }


        # Filename of geojson files
        self.filename_to_class = {
            '001_MM_L2_LARGE_BUILDING':'1',
            '001_MM_L3_RESIDENTIAL_BUILDING':'1',
            '001_MM_L3_NON_RESIDENTIAL_BUILDING':'1',
            '001_MM_L5_MISC_SMALL_STRUCTURE':'2',
            '002_TR_L3_GOOD_ROADS':'3',
            '002_TR_L4_POOR_DIRT_CART_TRACK':'4',
            '002_TR_L6_FOOTPATH_TRAIL':'4',
            '006_VEG_L2_WOODLAND':'5',
            '006_VEG_L3_HEDGEROWS':'5',
            '006_VEG_L5_GROUP_TREES':'5',
            '006_VEG_L5_STANDALONE_TREES':'5',
            '007_AGR_L2_CONTOUR_PLOUGHING_CROPLAND':'6',
            '007_AGR_L6_ROW_CROP':'6',
            '008_WTR_L3_WATERWAY':'7',
            '008_WTR_L2_STANDING_WATER':'8',
            '003_VH_L4_LARGE_VEHICLE':'9',
            '003_VH_L5_SMALL_VEHICLE':'10',
            '003_VH_L6_MOTORBIKE':'10'}


        # 30 extended classes
        self.extended_classes = (
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


        # Total number of bands ( 16-band + 3-band + Pancromatic)
        self.band_nb = 20

        # number of bands in each image type.
        self.channel_nb = {'3':3, 'A':8, 'M':8, 'P':1, 'C':10}

        # Images cover 1km x 1km subregions, and come in 4 types.
        self.band_types = ('3', 'A', 'M', 'P')

        # Descriptive names of the different bands
        self.band_names = {
            '3' : '3-band',
            'P' : 'Panchromatic',
            'M' : 'Multispectral',
            'A' : 'SWIR',
            'W' : 'WorldView',
            }

        self.band_range = {
            '3' : (0, 3),
            'P' : (3, 4),
            'M' : (4, 12),
            'A' : (12, 20),
            'W' : (3, 20),
            }


        # DSTL image bands to WorldView bands
        # If you look at the TIFF metadata you get the following info:
        # [3-bands] 6nnn_n_n:
        #   TIFFTAG_IMAGEDESCRIPTION={ bandlist = [ 4; 3; 2;] }
        # [16-bands] 6nnn_n_n_M:
        #   TIFFTAG_IMAGEDESCRIPTION={ bandlist = [ 6; 2; 3; 7; 4; 8; 5; 9;] }
        # [SWIR] 6nnn_n_n_A:
        #   TIFFTAG_IMAGEDESCRIPTION={ bandlist = [ 10; 11; 12; 13; 14; 15; 16; 17;]}
        # Kudos: @FPP_UK
        #
        # For no apparently good reason the dstl Multispectral bands have been reordered
        # with respect to the WorldView bands.

        self.band_index = {
            '3' : (0, 1, 2),
            'P' : (3, ),
            'M' : (4, 5, 6, 7, 8, 9, 10, 11),
            'A' : (12, 13, 14, 15, 16, 17, 18, 19),
            'W' : (-1, 3, 8, 3, 5, 9, 6, 10, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19)
            }


        # Based on the WorldView online bandwidth info and QGis extracted bandlist metadata and data
        # description
        #
        # Sensor-Bands :  WV3-Type, WV3-label, WV3range, Image-ID Notes
        # Kudos: @FPP_UK
        self.band_info = (
            ('Panchromatic', 'Panchromatic', '450-800nm', '6xxx_n_n_P 0.31m resolution'),
            ('Multi-spectral', 'Coastal', '400-450nm', '6xxx_n_n_M, RGB 1.24m resolution'),
            ('Multi-spectral', 'Blue', '450-510nm', '6xxx_n_n_M, RGB 1.24m resolution'),
            ('Multi-spectral', 'Green', '510-580nm', '6xxx_n_n_M, RGB 1.24m resolution'),
            ('Multi-spectral', 'Yellow', '585-625nm', '6xxx_n_n_M 1.24m resolution'),
            ('Multi-spectral', 'Red', '630-690nm', '6xxx_n_n_M 1.24m resolution'),
            ('Multi-spectral', 'Red Edge', '705-745nm', '6xxx_n_n_M 1.24m resolution'),
            ('Multi-spectral', 'Near-IR1', '770-895nm', '6xxx_n_n_M 1.24m resolution'),
            ('Multi-spectral', 'Near-IR2', '860-1040nm', '6xxx_n_n_M 1.24m resolution'),
            ('SWIR', 'SWIR-1', '1195-1225nm', '6xxx_n_n_A resolution reduced from 3.7m to 7.5m'),
            ('SWIR', 'SWIR-2', '1550-1590nm', '6xxx_n_n_A resolution reduced from 3.7m to 7.5m'),
            ('SWIR', 'SWIR-3', '1640-1680nm', '6xxx_n_n_A resolution reduced from 3.7m to 7.5m'),
            ('SWIR', 'SWIR-4', '1710-1750nm', '6xxx_n_n_A resolution reduced from 3.7m to 7.5m'),
            ('SWIR', 'SWIR-5', '2145-2185nm', '6xxx_n_n_A resolution reduced from 3.7m to 7.5m'),
            ('SWIR', 'SWIR-6', '2185-2225nm', '6xxx_n_n_A resolution reduced from 3.7m to 7.5m'),
            ('SWIR', 'SWIR-7', '2235-2285nm', '6xxx_n_n_A resolution reduced from 3.7m to 7.5m'),
            ('SWIR', 'SWIR-8', '2295-2365nm', '6xxx_n_n_A resolution reduced from 3.7m to 7.5m')
        )


    # End __init__

    def path(self, name, **kwargs):
        """Return path to various source files"""
        kwargs['sourcedir'] = self.sourcedir
        path = self.source_paths[name].format(**kwargs)
        return path

    def load_image(self, imageId, image_type):
        """Load a tiff image from input data.
        image_type is one of 3, A, M or P

        tiff files are organized as channels x height x width.
        """
        fn = self.path(image_type, imageId=imageId)

        img = tiff.imread(fn)

        # P images single channel greyscale. But add extra dimension (length 1) for consistency
        if image_type == 'P':
            img = np.expand_dims(img, axis=0)

        return img


    def image_size(self, imageId, image_type='3'):
        img = self.load_image(imageId, image_type)
        _, height, width = img.shape
        return (width, height)


    @property
    def regionIds(self):
        return sorted(set([iid[0:4] for iid in self.imageIds]))

    @property
    def imageIds(self):
        return sorted(self.grid_sizes.keys())

    @property
    def targetIds(self):
        return sorted(self.wkt.data.imageId.unique())

    @property
    def testIds(self):
        fn = self.path('sample')
        return list(pandas.read_csv(fn).ImageId.unique())

    @property
    def repeatedIds(self):
        """ "4 images (the ones that we moved from test to train) have been ignored.
            For those 4 images, whatever you submit won't matter, so it's easiest to
            keep them empty 'MULTIPOLYGON EMPTY'. "
        """
        return list(set(self.targetIds) & set(self.testIds))



    @property
    def wkt(self):
        if self._wkt is None:
            fn = self.path('train_wkt')
            self._wkt = WktPolygons(fn)
        return self._wkt

    @property
    def grid_sizes(self):
        if self._grid_sizes is None:
            fn = self.path('grid_sizes')
            grid_sizes = {}
            with open(fn) as csvfile:
                reader = csv.reader(csvfile)
                next(reader)    # skip header row
                for row in reader:
                    iid = row[0]
                    xmax = float(row[1])
                    ymin = float(row[2])
                    grid_sizes[iid] = (xmax, ymin)
            self._grid_sizes = grid_sizes
        return self._grid_sizes



class DstlDataset(object):
    """Construct munged dstl dataset."""
    def __init__(self, sourcedir=SOURCEDIR, outputdir=OUTPUTDIR, scale=SCALE):
        self.outputdir = outputdir
        self.scale = scale

        self.info = DstlInfo(sourcedir)

        self._subregion_size = None
        self._cache = {}

        self.data_paths = {
            'subregion_size': os.path.join('{outputdir}', 'data', 'subregion_size.json'),
            'bands'       : os.path.join('{outputdir}', 'data', 'bands_{scale}_{region}.npy'),
            'targets'     : os.path.join('{outputdir}', 'data', 'targets_{scale}_{region}.npy'),
            'composite'   : os.path.join('{outputdir}', 'data', 'composite_{scale}_{imageId}.png'),

            'image'         : os.path.join('{outputdir}', '{imageId}.png'),
        }


    def path(self, name, **kwargs):
        """Return path to various source files"""
        kwargs['outputdir'] = self.outputdir
        kwargs['scale'] = self.scale
        path = self.data_paths[name].format(**kwargs)
        return path

    def dataset_shape(self, name):
        channels = {'bands': self.info.band_nb, 'targets': self.info.class_nb}
        dataset_width = SUBREGION_MAX_WIDTH//self.scale  + 2 * REGION_BORDER
        dataset_height = SUBREGION_MAX_HEIGHT//self.scale + 2 * REGION_BORDER

        return (channels[name], dataset_height, dataset_width)

    def load(self, name, region, mode='r'):
        fn = self.path(name=name, region=region)

        # Mode 'x': create if dosn't exist
        if mode == 'x': mode = 'r+' if os.path.exists(fn) else 'w+'

        if mode == 'r' and fn in self._cache:
            return self._cache[fn]

        shape = self.dataset_shape(name)
        data = np.memmap(fn, dtype=np.uint8, mode=mode, shape=shape)
        if mode == 'r': self._cache[fn] = data
        return data

    def bands(self, region, mode='r'):
        return self.load('bands', region, mode)

    def targets(self, region, mode='r'):
        return self.load('targets', region, mode)

    def path_exists(self, name):
        fn = self.path(name)
        return os.path.exists(fn)

    def _json_load(self, name):
        fn = self.path(name)
        with open(fn) as fp:
            return json.load(fp)


    def _json_dump(self, data, name):
        fn = self.path(name)
        with open(fn, 'w') as fp:
            json.dump(data, fp, sort_keys=True, indent=4, separators=(',', ': '))


    def subregion_size(self, subregion):
        if not self._subregion_size:
            if not self.path_exists('subregion_size'):
                progress('(Caching subregions sizes...')
                sizes = {}
                for iid in self.info.imageIds:
                    sizes[iid] = self.info.image_size(iid, '3')
                    progress()
                self._json_dump(sizes, 'subregion_size')
                progress(')')

            self._subregion_size = self._json_load('subregion_size')
        w, h = self._subregion_size[subregion]
        return w//self.scale, h//self.scale


    def load_region(self, region, itype, channel=None, contrast=True):
        """ Glue 25 1km x 1km subregions into one 5km x 5km region """
        image_row = []
        for reg_row in range(0, 5): #FIXME MAGIC NUMBER
            image_column = []
            for reg_col in range(0, 5):
                iid = compose_viewId(region, reg_row, reg_col)
                img = self.info.load_image(iid, itype)
                if channel is not None:
                    img = img[channel, :, :]
                image_column.append(img)
            image_row.append(np.concatenate(image_column, axis=1))
        region_image = np.concatenate(image_row, axis=0)

        if contrast:
            low, high = np.percentile(region_image, (LOW_PERCENT, HIGH_PERCENT))
            region_image = stretch(region_image, low, high, dtype=np.uint8)

        return region_image


    def region_size(self, region):
        std_width, std_height = self.subregion_size(region+'_0_0')
        last_width, last_height = self.subregion_size(region+'_4_4')

        width = (std_width*4 + last_width)
        height = (std_height*4 + last_height)

        return width, height


    def region_slice(self, region):
        width, height = self.region_size(region)

        return REGION_BORDER, REGION_BORDER+ height, REGION_BORDER, REGION_BORDER+width


    def subregion_slice(self, subregion):
        region, reg_row, reg_col = parse_viewId(subregion)[0:3]
        std_width, std_height = self.subregion_size(region+'_0_0')
        width, height = self.subregion_size(subregion)

        rcol_start = REGION_BORDER + reg_col*std_width
        rcol_end = rcol_start + width
        rrow_start = REGION_BORDER + reg_row*std_height
        rrow_end = rrow_start + height

        return rrow_start, rrow_end, rcol_start, rcol_end


    # ===== BUILD =====
    def build(self, regions=None):
        self.build_bands(regions)
        self.build_targets(regions)
        #FIXME: build composites

    def build_targets(self, regions=None):
        """Create target masks from the raw training data"""
        progress('Building category targets...', end='\n')

        if not regions: regions = self.info.regionIds
        gs = self.info.grid_sizes

        for region in regions:
            for iid in self.info.targetIds:
                reg = parse_viewId(iid)[0]
                if reg != region: continue
                progress('    '+iid)

                class_polygons = self.info.wkt.polygons(iid)
                progress()

                dataset = self.load('targets', region, mode='x')
                xmax, ymin = gs[iid]
                width, height = self.subregion_size(iid)

                for ct in self.info.class_types:
                    polygons = class_polygons[ct]

                    mask = polygons_to_mask(polygons, xmax, ymin, width, height)
                    idx = self.info.class_index[ct]
                    rs, re, cs, ce = self.subregion_slice(iid)

                    dataset[idx, rs:re, cs:ce] = mask * 255.   # fixme magic constant

                    progress()
                progress('done')
        progress('done')



    def build_bands(self, regions=None):
        progress('Building bands...', end='\n')

        if not regions: regions = self.info.regionIds

        for region in regions:
            progress('    {} '.format(region))

            data = self.load('bands', region, mode='w+')
            width, height = self.region_size(region)

            for itype in self.info.band_types:
                progress(itype)

                for channel in range(0, self.info.channel_nb[itype]):
                    progress()

                    region_image = self.load_region(region, itype, channel, contrast=True)
                    region_image = cv2.resize(region_image, (width, height),
                                              interpolation=INTERPOLATION)

                    band_index = self.info.band_index[itype][channel]
                    data[band_index] = region_image.mean()

                    rs, re, cs, ce = self.region_slice(region) # row/col, start/end
                    data[band_index, rs:re, cs:ce] = region_image
                progress('  ')
            progress('done')

        progress('done')  # finished all regions


    def build_composites(self, subregions=None):
        progress('Building target composites...', end='\n')

        if not subregions: subregions = self.info.targetIds
        for subregion in subregions:
            progress('    '+subregion)
            fn = self.path('composite', imageId=subregion)
            xmax, ymin = self.info.grid_sizes[subregion]
            class_polygons = self.info.wkt.polygons(subregion)
            width, height = self.subregion_size(subregion)

            polygons_to_composite(class_polygons, xmax, ymin, width, height, fn,
                                  self.info.class_color, self.info.class_zorder)

            progress('done')


    def load_composite(self, subregion):
        fn = self.path('composite', imageId=subregion)
        if not os.path.exists(fn):
            self.build_composites([subregion])

        return Image.open(fn)



    def create_images(self, viewIds, composite=False):
        for view in viewIds:
            for iid in iterate_viewIds(view):
                fn = self.path('image', imageId=iid)
                progress(' creating '+ fn)
                img = self.image(iid, composite)
                img.save(fn)

                progress('done')


    def image(self, viewId, composite=False):
        """Return a single png image. Expects fully specified imageId  """

        region, reg_row, reg_col, itype, channel = parse_viewId(viewId)
        subregion = compose_viewId(region, reg_row, reg_col)

        if channel is None: channel = 0

        if itype == 'C':
            dataset = self.targets(region)
            loc = channel
        else:
            dataset = self.bands(region)
            loc = self.info.band_index[itype][channel]

        if reg_row == 5 and reg_col == 5:
            # whole 5km x 5km region
            chunk = self.region_slice(region)
        else:
            # Single 1km by 1km view
            chunk = self.subregion_slice(subregion)

        rs, re, cs, ce = chunk
        data = dataset[loc, rs:re, cs:ce]
        img = Image.fromarray(data)

        if composite:
            progress('( compositing')
            img1 = img.convert(mode='RGBA')
            img2 = self.load_composite(subregion)
            img = Image.alpha_composite(img1, img2)
            progress(')')

        return img

# ================ Polygon routines ====================

class WktPolygons(object):

    def __init__(self, filename):
        """ Load a WKT polygon CSV file"""
        names = ("imageId", "classType", "wkt")
        dtype = {"imageId":str, "classType":str, "wkt":str}
        data = pandas.read_csv(filename, names=names, dtype=dtype, skiprows=1)
        self.data = data

    def polygons(self, imageId):
        class_types = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        df_image = self.data[self.data.imageId == imageId]
        classPolygons = {}
        for ctype in class_types:
            multipolygon = shapely.wkt.loads(df_image[df_image.classType == ctype].wkt.values[0])

            # At least one polygon in the training data is invalid. Fix (Kudos: @amanbh)
            if not multipolygon.is_valid:
                multipolygon = multipolygon.buffer(0)
                #progress('Fixed invalid multipolygon {} {}'.format(iid, ctype))

            classPolygons[ctype] = multipolygon
        return classPolygons

    def imageIds(self):
        return sorted(self.data.imageId.unique())


def polygons_to_composite(class_polygons, xmax, ymin, width, height, filename,
                          class_color, class_zorder, transparent=True):
    img_width = 1.* width / DPI       # In inches (!)
    img_height = 1.* height / DPI
    fig = plt.figure(figsize=(img_width, img_height), frameon=False)
    axes = plt.Axes(fig, [0., 0, 1, 1]) # One axis, many axes
    axes.set_axis_off()
    fig.add_axes(axes)

    for classType, multipolygon in class_polygons.items():
        progress(classType)
        for polygon in multipolygon:
            patch = PolygonPatch(polygon,
                                 color=class_color[classType],
                                 lw=0.1,     #FIXME: Magic constant
                                 alpha=0.5,  #FIXME: Magic constant
                                 zorder=class_zorder[classType],
                                 antialiased=False,
                                 fill=True)
            axes.add_patch(patch)
            patch = PolygonPatch(polygon,
                                 color="#000000",
                                 lw=0.1,     #FIXME: Magic constant
                                 alpha=0.9,   #FIXME: Magic constant
                                 zorder=class_zorder[classType]+1,
                                 antialiased=True,
                                 fill=False)
            axes.add_patch(patch)

    axes.set_xlim(0, xmax)
    axes.set_ylim(ymin, 0)
    #axes.set_aspect(1)
    plt.axis('off')

    plt.savefig(filename, pad_inches=0, dpi=DPI, transparent=transparent)
    plt.clf()
    plt.close()


def polygons_to_mask(multipolygon, xmax, ymin, width, height, filename=None):
    img_width = 1.* width / DPI       # In inches (!)
    img_height = 1.* height / DPI
    fig = plt.figure(figsize=(img_width, img_height), frameon=False)
    axes = plt.Axes(fig, [0., 0, 1, 1]) # One axis, many axes
    axes.set_axis_off()
    fig.add_axes(axes)

    for polygon in multipolygon:
        patch = PolygonPatch(polygon,
                             color='#000000',
                             lw=0,               # linewidth
                             antialiased=True)
        axes.add_patch(patch)
    axes.set_xlim(0, xmax)
    axes.set_ylim(ymin, 0)
    #axes.set_aspect(1)
    plt.axis('off')

    if filename is None:
        filename = tempfile.NamedTemporaryFile(suffix='.png')
    plt.savefig(filename, pad_inches=0, dpi=DPI, transparent=False)
    plt.clf()
    plt.close()
    a = np.asarray(Image.open(filename))
    a = (1.- a[:, :, 0]/255.)  # convert from B&W to zeros and ones. #FIXME MAGIC CONSTANT
    return a


# Adapted from code by '@shawn'
def mask_to_polygons(mask, xmax, ymin, threshold=0.5, tolerance=1):
    all_polygons = []

    mask[mask >= threshold] = 1
    mask[mask < threshold] = 0

    for shape, _ in rasterio.features.shapes(mask.astype(np.int16),
                                                 mask=(mask == 1),
                                                 transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        all_polygons.append(shapely.geometry.shape(shape))

    all_polygons = shapely.geometry.MultiPolygon(all_polygons)

    # Transform from pixel coordinates to grid coordinates
    height, width = mask.shape
    all_polygons = shapely.affinity.scale(all_polygons, xfact=xmax/(width),
                                          yfact=ymin/(height), origin=(0, 0, 0))

    # simplify the geometry of the masks
    # FIXME: magic constant. 2.7*1e-6 is size of one pixel in grid coordinates
    all_polygons = all_polygons.simplify(tolerance*2.7*1e-6, preserve_topology=True)

    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)

    # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
    # need to keep it a Multi throughout
    if all_polygons.type == 'Polygon':
        all_polygons = shapely.geometry.MultiPolygon([all_polygons])

    return all_polygons


# TODO: Check me.
def polygon_jaccard(actual_polygons, predicted_polygons):
    """
    Calculate the Jaccard similiarity index between two sets of polygons
    Returns jaccard, true_positive, false_positive, false_negative
    """
    true_positive = predicted_polygons.intersection(actual_polygons).area
    false_positive = predicted_polygons.area - true_positive
    false_negative = actual_polygons.area - true_positive
    union = (true_positive+false_positive+false_negative)

    epsilon = 10e-100
    jaccard = (true_positive + epsilon) / (union + epsilon)

    return jaccard, true_positive, false_positive, false_negative



# ---------- Command Line Interface ----------


def _add_argument_version(parser):
    parser.add_argument(
        '--version',
        action='version',
        version=__version__)

def _add_argument_sourcedir(parser):
    parser.add_argument(
        '-s', '--sourcedir',
        action='store',
        dest='sourcedir',
        default=SOURCEDIR,
        metavar='PATH',
        help='Location of input data')

def _add_argument_outputdir(parser):
    parser.add_argument(
        '-o', '--outputdir',
        action='store',
        dest='outputdir',
        default=OUTPUTDIR,
        metavar='PATH',
        help='Location of processed data')

def _add_argument_scale(parser):
    parser.add_argument(
        '-S', '--scale',
        action='store',
        default=SCALE,
        type=int,
        help="Scale of images: '3'-band pixels per image pixel")

def _add_argument_randomize(parser):
    parser.add_argument(
        '-R', '--randomize',
        action='store_true',
        dest='randomize',
        default=False,
        help='Randomly reseed random number generator. (Default: 42)')


def _add_argument_regions(parser):
    parser.add_argument(
        'regions',
        nargs='*',
        metavar='regionIds',
        help="Optinal list of regions to process. Default is all regions.")

#FIXME : Common subregion option

def _cli():
    """DSTLDataset command line interface"""
    
    def add_cmd_build(cmdparser):
        parser = cmdparser.add_parser(
            'build',
            help='Build bands and targets.')
        parser.set_defaults(funcname='build')
        _add_argument_regions(parser)
        

    def add_cmd_build_bands(cmdparser):
        parser = cmdparser.add_parser(
            'build_bands',
            help='Build bands.')
        parser.set_defaults(funcname='build_bands')
        _add_argument_regions(parser)


    def add_cmd_build_targets(cmdparser):
        parser = cmdparser.add_parser(
            'build_targets',
            help='Build targets')
        parser.set_defaults(funcname='build_targets')

        _add_argument_regions(parser)

    def add_cmd_build_composites(cmdparser):
        parser = cmdparser.add_parser(
            'build_composites',
            help='Create composite images of target masks')
        parser.set_defaults(funcname='build_composites')

        parser.add_argument(
            'subregions',
            nargs='*',
            metavar='subregions',
            help="Optinal list of subregions to process. Default is all target subregions.")


    def add_cmd_image(cmdparser):
        parser = cmdparser.add_parser('image',
                                      help='Create images.')
        parser.set_defaults(funcname='create_images')

        parser.add_argument(
            'viewIds',
            nargs='+',
            metavar='viewIds',
            help="One or more imageIds e.g. 6100_1_3_A_3 ")

        parser.add_argument(
            '--composite',
            action='store_true',
            default=False,
            help='')

    parser = argparse.ArgumentParser(
        description=__description__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cmdparser = parser.add_subparsers(title='Commands',
                                      description=None,
                                      help="-h, --help Additional help",)

    _add_argument_version(parser)    
    _add_argument_sourcedir(parser)
    _add_argument_outputdir(parser)
    _add_argument_scale(parser)
    _add_argument_randomize(parser)
    
    add_cmd_build(cmdparser)
    add_cmd_build_bands(cmdparser)
    add_cmd_build_targets(cmdparser)
    add_cmd_build_composites(cmdparser)
    add_cmd_image(cmdparser)

    # Run command
    opts = vars(parser.parse_args())

    sourcedir = opts.pop('sourcedir')
    outputdir = opts.pop('outputdir')
    scale = opts.pop('scale')
    
    funcname = opts.pop('funcname')
    randomize = opts.pop('randomize')
    if randomize:
        random.seed()
        
    dstl = DstlDataset(sourcedir=sourcedir, outputdir=outputdir, scale=scale)
    func = getattr(dstl, funcname)
    func(**opts)



if __name__ == "__main__":
    _cli()
    