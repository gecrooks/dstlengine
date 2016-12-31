
from dstl_utils import load_train_iids, output_dir
from PIL import Image
import os


for iid in load_train_iids():
    fn1 = os.path.join(output_dir, "{}_P1.png".format(iid) )
    fn2 = os.path.join(output_dir, "{}_C0.png".format(iid) )
    fn3 = os.path.join(output_dir, "{}_comp.png".format(iid) )

    img1 = Image.open(fn1).convert(mode = 'RGBA')
    img2 = Image.open(fn2)
    
    print(iid, img1.size, img2.size)
    img3 = Image.alpha_composite(img1, img2)
    
    img3.save(fn3)
    
    print(img1.mode, img2.mode)