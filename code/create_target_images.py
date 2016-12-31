

from dstl_utils import get_train_ids, classes, class_zorder, class_color, get_training_data

from dstl_utils import load_image, load_grid_sizes, grid_resolution

from descartes.patch import PolygonPatch
import matplotlib.pyplot as plt
from shapely import wkt

from dstl_utils import load_train_iids, dpi, output_dir, load_image
import os

td = get_training_data() 
    
for iid in load_train_iids():
    print(iid)
    filename = '{}_T.png'.format(iid)
    filepath = os.path.join(output_dir,filename)
    
    # Make same size as type 3 image
    C,H,W = load_image(iid, '3').shape
    xmax, ymin = load_grid_sizes()[iid]
    W = - H * xmax / ymin
    
    H /= dpi
    W /= dpi
    
    fig = plt.figure(figsize=(W,H), frameon=False)
    axes = plt.Axes(fig, [0., 0, 1, 1]) # One axis, many axes
    axes.set_axis_off()         
    fig.add_axes(axes)  
    
    df_image = td[td.image_id == iid]
    
    for class_type in classes.keys():
        # if class_type == 5: break
        
        polygons = wkt.loads(df_image[df_image.class_type == class_type].wkt.values[0])
        
        print('{} : {} \tcount = {}'.format(class_type, classes[class_type], len(polygons)))
        
        for polygon in polygons:
            mpl_poly = PolygonPatch(polygon,
                                color=class_color[class_type],
                                lw=0,
                                alpha=0.5,
                                zorder=class_zorder[class_type],
                                antialiased =True)
            axes.add_patch(mpl_poly)

    
    axes.set_xlim(0, xmax)
    axes.set_ylim(ymin, 0)
    axes.set_aspect(1)
    plt.axis('off')
    plt.savefig(filepath, pad_inches=0, dpi=dpi)
    plt.clf()



