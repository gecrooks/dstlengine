
from core_dstl import *
import core_dstl
import h5py
import cv2
#
# Process the raw dstl data.
#

   
def build_init(datadir):
    """input: file path to image datset
       writes directory file structure for .h5 data and 
       writes image data with 
    """
    
    progress('Initilizing dataset...')
  
    import os
    if os.path.exists(datadir): 
        import sys
        sys.exit('Fail: data directory already exists: '+datadir)
    os.makedirs(datadir)
      
    # Create a file for writing. Fails if file exists
    fn = getpath('data', datadir=datadir)
    datafile = h5py.File(fn, mode='w')   
  
    # Create all datasets
    for region in regionIds() :
        dataset = datafile.create_dataset(region, 
               (nb_features,region_height, region_width), 
               dtype = 'uint8', 
               chunks = hdf5_chunks,
               compression = hdf5_compression,)
          
    progress('done')
    datafile.close()



def build_data( datadir,
                regions = None,
                imageType = None,
                channel = None,
                train_only = False,
                ) :      
    progress('Build data...\n')   
    progress('   Initializing data processing...')
    load_dynamic_range() # cached
      
    datafile = h5py.File( getpath('data', datadir=datadir), 'r+') 
    
    if not regions or regions == ['all']:
        regions = regionIds()
 
    if not imageType or imageType == 'all': 
        imageTypes = core_dstl.imageTypes
    else:
        imageTypes = [imageType,]
 
    train = None
    if train_only: 
        progress('(Training images only)')
        train = set( train_imageIds() ) 
 
    progress('done')
 
    for region in regions:
        dataset = datafile[region]
        # Loop over all subregions in 5x5 region    
        for reg_row in range(0,5):
            for reg_col in range (0,5):
                iid = compose_viewId(region, reg_row, reg_col)
                if train_only and iid not in train: continue                
                
                progress('    '+iid)
                for itype in imageTypes : 
                    if channel is not None :
                        channels = [channel,]
                    else :
                        channels = range(0,imageChannels[itype])
                    
                    for chan in channels :
                        img = _process_image(datadir, region, reg_row, reg_col, itype, chan)

                        # Save image data to correct location in region data
                        dataset[ feature_loc[itype][chan], 
                                border + reg_row* std_height: border + std_height*(reg_row+1) ,
                                border + reg_col* std_width: border + std_width*(reg_col+1) 
                                ] = img                                
                        progress()
                progress('done')

    datafile.close()
    progress('done')

def _process_image(datadir,region, reg_row, reg_col, imageType, channel) :  
    # Called by build_data
    # Process a single channel of a single 1km x 1km area.
                               
    iid = compose_viewId(region, reg_row, reg_col)

    data = load_image(iid, imageType)[channel]
    
    # If an edge image, pad to size of _0_0 image
    iid00 = compose_viewId(region, 0, 0)  
    c00, h00, w00,  = load_image(iid00, imageType).shape
    h, w = data.shape
    if h != h00 or w != w00 :
        padded_data = np.zeros( (h00, w00) )
        padded_data[:h, :w] = data
        data = padded_data
        
    # Stetch values so that effective dynamic range is [0,255]
    imageRange = load_dynamic_range()   
    low, high = imageRange[imageType][channel]
    data = stretch_to_uint8( data, low, high)
                                         
    # Rescale all images to standard image size
    img = Image.fromarray(data)          
    img = img.resize( (std_width, std_height),resample = Image.BICUBIC) 
    img = np.asarray(img) 
                       
    return img
 
    
    
def align( datadir,
            regions = None,
            imageType = None,
            channel = None,
            dry_run = False,
            ) :      
    progress('Re-registering images...\n') 
      
    datafile = h5py.File( getpath('data', datadir=datadir), 'r+') 
    
    if not regions or regions == ['all']:
        regions = regionIds()
 
    if not imageType or imageType == 'all': 
        imageTypes = ('A', 'M', 'P')
    else:
        imageTypes = [imageType,]
 
 
    progress('done')
 
    for region in regions:
        progress('    ' + region)
        
        dataset = datafile[region]
                
        for itype in imageTypes : 
            if channel is not None :
                channels = [channel,]
            else :
                channels = range(0,imageChannels[itype])

            for chan in channels :
                img = _align_region(datadir, region, itype, chan, dry_run)
                progress()
                
        progress('done')

    datafile.close()
    progress('done')

def _align_region(datadir, region, itype, chan, dry_run):
    """ """
    
    datafile = h5py.File( getpath('data', datadir=datadir), 'r+')
    # grab dataset for specific region
    dataset = datafile[region]
    
    window_size = 4096
    start = region_height //2 - window_size //2
    end = start + window_size
    
    f1 = feature_loc['3'][0]
    f2 = feature_loc[itype][chan]
    
    img1 = dataset[f1, start:end, start:end].astype(np.float32)
    img2 = dataset[f2, start:end, start:end].astype(np.float32)
    
    warp_mode = cv2.MOTION_TRANSLATION
    #warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2,3, dtype=np.float32)
    
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,  1e-5)
        (cc, warp_matrix) = cv2.findTransformECC (img1, img2, warp_matrix, warp_mode, criteria)
    except cv2.error:
	print('findTransformEEC Failed to converge: {}_5x5_{}_{}'.format(region, itype, chan) )    
        return




    print("register {} {}".format(itype, chan))
    print("cc:{}".format(cc))
    print(warp_matrix)
    
    img1 = dataset[f1, :, :].astype(np.float32)
    img2 = dataset[f2, :, :].astype(np.float32)
    img3 = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]), 
                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    if not dry_run:
        progress('saving aligned image')

        dataset[f2, :, :] = img3.astype(np.uint8)
 
    
    


# Create visualizations of the  data    

 
image_scale_default = 0.2

# Build png images of regions or subregions from dstl.data
def build_images(datadir, imageIds, scale=None, composite=False) :        
    for iid in imageIds:
        region, row, col, imageType, channel = parse_viewId(iid)
        
        if not row: row =5
        if not col: col =5
        
        if not imageType or imageType == '*': 
            itypes = imageTypes
        else:
            itypes = [imageType,]
        
        for itype in itypes:
          
            if channel is None or channel == '*' : 
                channels = range(0, imageChannels[itype])
            else:
                channels = [channel,]
                
            for chan in channels:
                viewId = compose_viewId(region, row, col, itype, chan)
                
                fn = '{datadir}/{iid}.png'.format(datadir=datadir,iid = iid)
   
                progress('creating '+ fn)
                
                img = build_image(datadir,iid, scale, composite) 
                img.save(fn)  
                
                progress('done')
          
region_scale_default = 0.2
image_scale_default = 1.0

# build a single png image
def build_image(datadir, viewId, scale=None, composite=False) :
    # Expects fully specifice imageId 
    region, row, col, imageType, channel = parse_viewId(viewId)
    
    with h5py.File( getpath('data', datadir=datadir), 'r') as datafile :          
        
        dataset = datafile[region]
        feature = feature_loc[imageType][channel]      
        
        if row == 5 and col==5 :
            # whole 5km x 5km region
            data = dataset[feature, border: border+ 5*std_height, border: border+ 5*std_width ]
            if not scale : scale = region_scale_default
        else :
            # Single 1km by 1km view
            row_start = border + int(row)*std_height
            col_start = border + int(col)*std_width
            data = dataset[feature, row_start : row_start+ std_height, col_start: col_start+std_width ]
            if not scale : scale = image_scale_default
                      
        img = Image.fromarray(data) 

        if composite :
            progress('(compositing)')
            fn = getpath('composite', imageId = viewId[0:8], datadir=datadir)
            img1 = img.convert(mode = 'RGBA')
            img2 = Image.open(fn)
            img = Image.alpha_composite(img1, img2)

        if scale != 1.0:
            width, height = img.size
            width = int(round( width*scale ) )
            height = int(round( height* scale))       
            img = img.resize( ( width, height ) , resample = Image.BICUBIC)  
        
    return img


 
 
# Create the target lables from the raw training data
def build_targets(datadir):
    progress('Building category targets...\n')
    progress('   Loading polygons...')
    wkt_polygons = load_wkt_polygons()
    progress('done')
    
    with h5py.File( getpath('data', datadir=datadir), 'r+') as datafile :           
        for iid in train_imageIds() :
            progress('    '+iid)
            class_polygons = wkt_polygons[iid]
            region, reg_row, reg_col = parse_viewId(iid)[0:3]
            dataset = datafile[region]
            for ct in classTypes:
                polygons = class_polygons[ct]
                mask = polygons_to_mask(polygons, std_xmax, std_ymin, std_width, std_height)  
                loc = feature_loc['C'][ct]
                row = border + reg_row * std_height   
                col = border + reg_col * std_width
                dataset[loc, row:row+std_height, col:col+std_width] = mask*255.
                progress()
            progress('done')
    progress('done')
  

def build_composites(datadir, outline = True):
    progress('Building target composite...\n    ')
    wkt_polygons = load_wkt_polygons()          
    
    for iid in train_imageIds() :
        progress('    ') ; progress(iid)
        class_polygons = wkt_polygons[iid]
        fn = getpath('composite', imageId = iid, datadir=datadir)
        polygons_to_composite(class_polygons, std_xmax, std_ymin, std_width, std_height, fn, outline)          
        progress('done')
    progress('done')



def polygons_to_composite(class_polygons, xmax, ymin, width, height, filename, outline=True) :
     """ If outline is true, create transparent outline of classes suitable for layering over other images."""

     width /= float(dpi)
     height /= float(dpi)
     
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



def polygons_to_mask(multipolygon, xmax, ymin, width, height, filename=None) :
     width /= float(dpi)
     height /= float(dpi)    

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



# Adapted from code by 'shawn'
def mask_to_polygons(mask, xmax, ymin, threshold=0.4):
     all_polygons=[]
    
     #print( mask.min(), mask.max()  ) 
     mask[mask >= threshold] = 1
     mask[mask < threshold] = 0
    
     for shape, value in rasterio.features.shapes(mask.astype(np.int16),
                                 mask = (mask==1),
                                 transform = rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):

         all_polygons.append(shapely.geometry.shape(shape))

     all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    
     # simply the geometry of the masks
     all_polygons = all_polygons.simplify(grid_resolution, preserve_topology=False)
    
    
    
     # Transform from pixel coordinates to grid coordinates
     # FIXME
     height, width = mask.shape  
     #width = 1.* width*width/(width+1.)
     #height = 1.* height*height/(height+1.)
     #print(width, height)
     #X = X*X/float(X+1)
     #Y = Y*Y/float(Y+1)  
     all_polygons = shapely.affinity.scale(all_polygons, xfact = xmax/width, yfact = ymin/height, origin=(0,0,0))
    
    
     if not all_polygons.is_valid:
         all_polygons = all_polygons.buffer(0)
     
     # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
     # need to keep it a Multi throughout
     if all_polygons.type == 'Polygon':
        all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    
    
     return all_polygons
        
        
def polygon_jaccard(actual_polygons, predicted_polygons) :
     true_positive  = predicted_polygons.intersection(actual_polygons).area
     false_positive = predicted_polygons.area - true_positive
     false_negative = actual_polygons.area - true_positive
     intersection = (true_positive+false_positive+false_negative)
    
     jaccard = None
     if intersection !=0 : jaccard = true_positive / intersection
    
     return jaccard, true_positive, false_positive, false_negative
    
     
 
 
 

