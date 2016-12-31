    
from dstl_utils import *
from PIL import Image

def progress(string = '.') : print(string, end="", flush=True)

td = get_training_data()
image_ids = get_image_ids(td)

for i in image_ids:
    progress(i+' ')
    
    a = load_image(i, '3')    
    C,H,W = a.shape 
    
    for channel in range(0,3) :
        img = Image.fromarray(a.astype('uint8')[channel] )  
        filename = "{}_3{}.png".format(i, channel+1)
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)    
        progress()
    
    # RGB version
    a = np.rollaxis(a, 0, 3) 
    a = a/8 # 10 bit to 8 bit data
    img = Image.fromarray(a.astype('uint8'))  
    filename = "{}_3.png".format(i)
    filepath = os.path.join(output_dir, filename)
    img.save(filepath) 
    progress() 



    a = load_image(i, 'P') 
    a = a/8 # 10 bit to 8 bit data  # CHECK
    img = Image.fromarray(a.astype('uint8')[0] )  
    img = img.resize( (W,H) ) 
    filename = "{}_P.png".format(i)
    filepath = os.path.join(output_dir, filename)
    img.save(filepath)    
    progress()
    
    
    a = load_image(i, 'A')
    a = a/64 # 12 bit to 8 bit data  # CHECK
    for channel in range(0,8) :
        img = Image.fromarray(a.astype('uint8')[channel] ) 
        img = img.resize( (W,H) , resample = Image.LANCZOS)  
        filename = "{}_A{}.png".format(i, channel+1)
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)  
        progress()  
        
    a = load_image(i, 'M')
    a = a/8 # 10 bit to 8 bit data  # CHECK
    for channel in range(0,8) :
        img = Image.fromarray(a.astype('uint8')[channel] ) 
        img = img.resize( (W,H) )  
        filename = "{}_M{}.png".format(i, channel+1)
        filepath = os.path.join(output_dir, filename)
        img.save(filepath) 
        progress()
    
    print()   
        