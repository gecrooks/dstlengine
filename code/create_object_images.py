

from dstl_utils import get_train_ids, classes, class_zorder, class_color, get_training_data

from dstl_utils import load_image, load_grid_sizes, grid_resolution

from descartes.patch import PolygonPatch
import matplotlib.pyplot as plt
from shapely import wkt

def plot_polygons(fig, axis, polygonsList):
    for class_type in polygonsList:
       # if class_type == 5: break
        print('{} : {} \tcount = {}'.format(class_type, classes[class_type], len(polygonsList[class_type])))

        for polygon in polygonsList[class_type]:
            mpl_poly = PolygonPatch(polygon,
                                    color=class_color[class_type],
                                    lw=0,
                                    alpha=0.5,
                                    zorder=class_zorder[class_type],
                                    antialiased =True)
            axis.add_patch(mpl_poly)


def visualize_image(iid):
    '''         
    Plot all images and object-polygons
    
    Parameters
    ----------
    imageId : str
        imageId as used in grid_size.csv
    plot_all : bool, True by default
        If True, plots all images (from three_band/ and sixteen_band/) as subplots.
        Otherwise, only plots Polygons.
    '''  
    
    imageID = iid
           
    df = get_training_data() 
    df_image = df[df.image_id == imageId]
    #xmax, ymin, W, H = get_size(imageId)
    
    # Create new pyplot figure. figsize is in inches. But irrelevant once we save as PNG
    # well should get a 3300 pixel image 
    
    
    
    dpi=512
    
    H = 3403/512
    W = 3348/512
    #3225
  #   W is 3349, and H is 3391
    xmax = 0.009158
    ymin = -0.009043
        
    H = 3*3225/512
    W = - H * xmax /ymin
#    H = -y_min *W/x_max
    
    #gs = load_grid_sizes()
    
    # Make same size as type 3 image
    C,H,W = load_image(iid, '3').shape
    xmax, ymin = load_grid_sizes()[iid]
    W = - H * xmax / ymin
        
        
  #  H = round( xmax/grid_resolution) 
  #  W = round(-ymin/grid_resolution)     
    
    
    print("WH", iid, W, H, xmax, ymin)
    
    
    dpi=512
    H /= dpi
    W /= dpi
    

    fig = plt.figure(figsize=(W,H), frameon=False)
    ax = plt.Axes(fig, [0., 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)  

    print('ImageId : {}'.format(imageId))
    polygonsList = {}
    for class_type in classes.keys():
        polygons = wkt.loads(df_image[df_image.class_type == class_type].wkt.values[0])
        polygonsList[class_type] = polygons
    plot_polygons(fig, ax, polygonsList)
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, 0)
        
    ax.set_aspect(1)
        
    return (fig, None, ax)

for imageId in get_train_ids():
    
    fig, axArr, ax = visualize_image(imageId)
    filename = 'Objects3--' + imageId + '.png'
    plt.axis('off')
    #plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=660)
    plt.savefig(filename, pad_inches=0, dpi=512)
    #plt.savefig('Objects--' + imageId + '.png')
    plt.clf()
