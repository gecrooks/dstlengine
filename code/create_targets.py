
# TODO: Rename to create_targets
# TODO: change output orders to tf default of width, height, channels?
# TODO: Rename 'td'

from dstl_utils import get_train_ids, classes, class_zorder, class_color, get_training_data

from dstl_utils import load_image, load_grid_sizes, grid_resolution

from descartes.patch import PolygonPatch
import matplotlib.pyplot as plt
from shapely import wkt

from dstl_utils import load_train_iids, dpi, output_dir, load_image
import os
import numpy as np
from PIL import Image

td = get_training_data() 
    
for iid in load_train_iids():
    print(iid)
    
    # Make same size as type 3 image
    C,H,W = load_image(iid, '3').shape
    xmax, ymin = load_grid_sizes()[iid]
    #W = - H * xmax / ymin
    
    H /= dpi
    W /= dpi
    
    df_image = td[td.image_id == iid]
    
    data = []
    
    # A composite images of all different classes
    composite_fig = plt.figure(figsize=(W,H), frameon=False)
    composite_axes = plt.Axes(composite_fig, [0., 0, 1, 1]) # One axis, many axes
    composite_axes.set_axis_off()         
    composite_fig.add_axes(composite_axes)  
    
    
    for class_type in classes.keys():
        
        fig = plt.figure(figsize=(W,H), frameon=False)
        axes = plt.Axes(fig, [0., 0, 1, 1]) # One axis, many axes
        axes.set_axis_off()         
        fig.add_axes(axes)  
        
        # if class_type == 5: break
        
        polygons = wkt.loads(df_image[df_image.class_type == class_type].wkt.values[0])
        
        print('{} : {} \tcount = {}'.format(class_type, classes[class_type], len(polygons)))
        
        for polygon in polygons:
            patch = PolygonPatch(polygon,
                                color='#000000',
                                lw=0,
                                antialiased =False)
            axes.add_patch(patch)

            patch = PolygonPatch(polygon,
                                color=class_color[class_type],
                                lw=0.2,   
                                alpha=1.0,
                                zorder=class_zorder[class_type],
                                antialiased =True,
                                fill =False)
            composite_axes.add_patch(patch)
    
    
    
        axes.set_xlim(0, xmax)
        axes.set_ylim(ymin, 0)
        axes.set_aspect(1)
        plt.axis('off')
        filepath = os.path.join(output_dir,'{}_C{}.png'.format(iid,class_type))        
        plt.savefig(filepath, pad_inches=0, dpi=dpi, transparent=True)
        plt.clf()
        plt.close()


        a = np.asarray(Image.open(filepath))
        a = (1- a[:,:,0]//255)  # convert from B&W to zeros and ones.
        data.append(a)
   #     print(a.shape)
   #     print(a)
   #     print( np.count_nonzero(a) )

    d = np.stack(data)  
    filename = "{}_labels.npy".format(iid)
    filepath = os.path.join(output_dir, filename)     
    np.save(filepath, d)


    composite_axes.set_xlim(0, xmax)
    composite_axes.set_ylim(ymin, 0)
    composite_axes.set_aspect(1)
    plt.axis('off')
    filepath = os.path.join(output_dir,'{}_C0.png'.format(iid,class_type))        
    plt.savefig(filepath, pad_inches=0, dpi=dpi, transparent=True)
    plt.clf()
    plt.close()
    


