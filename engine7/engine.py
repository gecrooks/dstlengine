#!/usr/bin/env python

from dstl import *
import dstl as _dstl 

import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import *
import keras.optimizers
from keras.callbacks import *

from functools import partial

#sys.stderr.write('Keras:'+ keras.__version__+'\n')

# ================ Meta ====================
__description__ = 'DSTL Prognostication Engine'


BATCHSIZE = 32

OPTIMIZER = 'rmsprop' 

NB_EPOCH = 1000

THRESHOLDS = [0.4, 0.3, 0.5, 0.4, 0.5, 0.3, 0.5, 0.5, 0.05, 0.2]



def get_optimizer(name=OPTIMIZER, lr_factor=1.0): 
    opt = keras.optimizers.get(name)
    config = opt.get_config()
    config['lr'] = config['lr'] * lr_factor
    opt = keras.optimizers.get(name, config)
    return opt


# rs,re,cs,ce row start, row end, column start, column end

def jaccard(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

def j4(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return jac[4]



def model_tilesize(model) :
    """Return tilesize"""
    return model.output_shape[2]
    
def model_tileborder(model):
    """Return tileborder"""
    return (model.input_shape[2] - model.output_shape[2]) // 2

# TODO: Take class_areas from info
class_areas = np.array([0.03305758113945797,
                        0.007056328714676294,
                        0.008090176443069096,
                        0.03003240318324303,
                        0.10173965514005655,
                        0.27565040186522105,
                        0.004942402053481345,
                        0.0016848797776180405,
                        3.728956501987033e-05,
                        0.00015264662431405042,])

weights_by_name = {
    'area' : K.variable(value=np.reciprocal(class_areas)),
    'uniform' : K.ones(shape=(10)),
    }

def weighted_relent(y_true, y_pred , weights):
    y_true = K.clip(y_true, K.epsilon(), 1.-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1.-K.epsilon())
    loss =  y_true * K.log(y_true / y_pred) + (1.-y_true) * K.log((1.- y_true) / (1.-y_pred) )
    loss = K.mean(loss, axis= [-2,-1])    
    loss *=  weights
    loss = K.mean(loss, axis=-1)
    return loss

def relent(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1.-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1.-K.epsilon())
    loss =  y_true * K.log(y_true / y_pred) + (1.-y_true) * K.log((1.- y_true) / (1.-y_pred) )
    loss = K.mean(loss, axis= [-2,-1])    
    loss *=  K.variable(value=np.reciprocal(class_areas))
    loss = K.mean(loss, axis=-1)
    return loss

# FIXME
TARGET_SELECTIONS = {
    'c10': slice(0,10),
    'trees': (4,),
    }


class DstlEngine(object):
    def __init__(self, sourcedir=SOURCEDIR, outputdir= OUTPUTDIR, scale = SCALE):
        self.dstl = DstlDataset(sourcedir, outputdir, scale)
        self.info = self.dstl.info
    
        self.paths = {
            'models'       : os.path.join('{outputdir}', '{model_name}.hdf5'),
            'predictions'  : os.path.join('{outputdir}', 'pred_{model_name}.csv'),
            'visualization' : os.path.join('{outputdir}', '{subregion}_{model_name}_pred.png'),
            }
            
        self.subsetIds = 'all','test', 'test85', 'train', 'validation' 

    
    def path(self, name, **kwargs) :
        """Return path to various source files"""
        kwargs['outputdir'] = self.dstl.outputdir
        path = self.paths[name].format(**kwargs)
        return path
        
    @property
    def validationIds(self):
        # return ['6140_3_1', '6110_1_2', '6160_2_1', '6170_0_4', '6100_2_2']  # validation set from @ironbar
        return ['6100_1_3']
        
    @property
    def trainIds(self) :
        return list(set(self.info.targetIds) - set(self.validationIds))
      

    def subregions(self, name):
        """
        all
        test
        test85: First fifth of test data. Since public test is 0.18 of data, will
            actually test approx 15 regions. There are no regions from train set in this subset.
        train
        validation
        """       
        
        if name == 'all': 
            return self.info.imageIds
        elif name == 'test':
            return self.info.testIds
        elif name == 'test85' :
            return self.info.testIds[0:85]
        elif name == 'train':
            return self.trainIds             
        elif name == 'validation':
            return self.validationIds
        else :
            sys.exit('Unknown name for subset of ids: {}'.format(name) )
 

    
 
 
    
    def tile(self, viewId, tilesize, train=True):
        # tile up an image into tiles tilesize*tilesize
        # return a list of (region, rrow, rcol), where (row col) are origin point of tiles
     
        if isinstance(viewId, str) : viewId = [viewId,]
        coords = []
        for iid in viewId :
            region, reg_row, reg_col = parse_viewId(iid)[0:3]    
            if reg_row is None and reg_col is None :
                rs,re,cs,ce = self.dstl.region_slice(region)
            else :
                rs,re,cs,ce = self.dstl.subregion_slice(iid)

            height = re-rs 
            width = ce-cs
            
            if train :
                # round down to tilesize.
                # when training we don't want edge tiles since they don't have complete target information
                re = rs + rounddown(height, tilesize)
                ce = cs + rounddown(width, tilesize)    
            else:
                # round up to tilesize, and make sure we cover entire image
                re = rs + roundup(height, tilesize)
                ce = cs + roundup(width, tilesize)

            for row in range( rs, re, tilesize) :
                for col in range( cs, ce, tilesize ) :
                    coords.append( ( region, row, col ) )
    
        return coords
    
    #TODO subsets of targets and bands
    def make_batch(self, batch_coords, tilesize, tileborder, target_selection = None, augment=0.0):
        bands_max = 255. #float(np.iinfo(band_dtype).max)             #fixme
        targets_max = 255. #float(np.iinfo(class_dtype).max)           # fixme
  
        if augment and random.random() < augment :      
            flip = random.random() < 0.5
            rotate = random.randint(0,3)
        else :
            flip = False
            rotate = 0
    
    
        bands = np.stack([ 
            self.dstl.bands(region)[:,x-tileborder:x+tilesize+tileborder,
                y-tileborder:y+tilesize+tileborder] 
            for region, x, y in batch_coords ])
                
        if flip : bands = bands[..., ::-1]
        if rotate: bands = np.rot90(bands,  rotate, (2,3) )
        bands = K.cast_to_floatx(bands)
        bands /= bands_max

        # fixme
        #if target_selection is None:
        #    target_selection = slice(None, None, None)
            
        targets = np.stack([ 
            self.dstl.targets(region)[:, x:x+tilesize, y:y+tilesize]
            for region, x, y in batch_coords ])             # fixme: row col rather than x,y?
        
        if flip : targets = targets[..., ::-1]
        if rotate: targets = np.rot90(targets,  rotate, (2,3) )
        targets = K.cast_to_floatx(targets)
        targets /= targets_max
        
        return bands, targets
        


    def gen_batch(self, coords, tilesize, tileborder, batchsize=BATCHSIZE, augment=0.0, train=True) :
        coords_nb = len(coords)
        if train: 
            coords_nb = rounddown(coords_nb, batchsize)
            
        while 1:
            if train: random.shuffle(coords) 

            for i in range(0, coords_nb, batchsize):
                batch_coords = coords[i:i+batchsize] 
                yield self.make_batch(batch_coords, tilesize, tileborder, augment)
    
    
    def load_model(self, model_name) :
        if '_e' in model_name :
            model_basename, model_epoch = model_name.split('_e')
            model_epoch = int(model_epoch)
            model_fn = self.path('models', model_name=model_basename)
            print('    Loading '+ model_fn) 
            model = keras.models.load_model(model_fn) 
            weights_fn = self.path('models', model_name=model_name)        
            model.load_weights(weights_fn)  
            
        else :
            model_basename = model_name
            model_fn = self.path('models', model_name=model_basename)
            print('    Loading '+ model_fn) 
            model = keras.models.load_model(model_fn)          
            model_epoch = 0

        return model, model_basename, model_epoch



    def train(self, 
              model_name, 
              optimizer=OPTIMIZER, 
              lr_factor=1.0, 
              nb_epoch=NB_EPOCH, 
              batchsize=BATCHSIZE,
              augment=0.0,
              weights = 'uniform',
              scale=SCALE,
              reduce_lr = True,
              ) :
  
        loss = lambda x, y : weighted_relent(x, y, weights_by_name[weights])

        model, model_basename, model_epoch = self.load_model(model_name)

        opt = get_optimizer(optimizer, lr_factor)
        #opt = keras.optimizers.RMSprop(rho=0.99)
        #opt = keras.optimizers.SGD(lr=0.001, momentum=0.99, decay=0.0, nesterov=True)
        print('model: {}, Optimizer: {}, learning factor: {}, batchsize: {}, augment: {}, weights: {}'.format(model_name, 
                    optimizer, lr_factor, batchsize, augment, weights) )
    
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=[jaccard, j4, 'accuracy'])
        model.summary()
 
        tilesize = model_tilesize(model)
        tileborder = model_tileborder(model)

        train_coords = self.tile(self.trainIds, tilesize, train = True)
        val_coords = self.tile(self.validationIds, tilesize, train = True)     
        
        callbacks = []
        if reduce_lr :
            callbacks.append( ReduceLROnPlateau(monitor='val_jaccard', mode='max', factor=0.5, patience=10, verbose=1) )
        
        checkpoint_filepath = 'output/'+model_basename+'_e{epoch:03d}.hdf5'
        callbacks.append(ModelCheckpoint(checkpoint_filepath))
        #checkpoint_cb = ModelCheckpoint(checkpoint_filepath, save_best_only=True, monitor='val_loss', save_weights_only=True, mode='auto')
        
        callbacks.append( CSVLogger('output/'+model_name+'.csv'))
        
 

        model.fit_generator(
            self.gen_batch(train_coords, tilesize, tileborder, batchsize, augment, train=True),
            len(train_coords),
            nb_epoch=nb_epoch+model_epoch,
            validation_data=self.gen_batch(val_coords, tilesize, tileborder, batchsize, train=False),
            nb_val_samples=len(val_coords),
            callbacks = callbacks,
            initial_epoch =model_epoch
            )


    def predict(self, model_name, subregions, batchsize=BATCHSIZE, composite= False) :        
        if not subregions: subregions = ('validation',)
        if len(subregions) ==1 and subregions[0] in self.subsetIds :
            subregions = self.subregions(subregions[0])        
        
        model, model_basename, model_epoch = self.load_model(model_name)
        
        fnout = self.path('predictions', model_name=model_name)
        with open(fnout, 'w') as csvfile:
            column_headings = ["ImageId", "ClassType", "MultipolygonWKT"] #FIXME. this stuff should be in DstlInfo or polygons
            writer = csv.writer(csvfile)
            writer.writerow(column_headings)

            for iid in subregions :
                region = iid[0:4]
                progress('    '+iid)
                
                targets = self.predict_subregion(model, iid, batchsize)

        
                xmax, ymin = self.info.grid_sizes[iid]
                progress(' ')
                poly = {}
                for m in range(0,10) :
                    progress()

                    polys = mask_to_polygons( targets[m,:,:], xmax, ymin, threshold=THRESHOLDS[m])
                    poly[str(m+1)] = polys
                
                    polygons_text = shapely.wkt.dumps(polys, rounding_precision=8)
                    if  polygons_text == "GEOMETRYCOLLECTION EMPTY":            #FIXME
                        polygons_text = "MULTIPOLYGON EMPTY"
                    writer.writerow((iid, m+1, polygons_text))

                if composite :  # fixme visulize
                    view_fp = self.path('visualization', subregion=iid, model_name=model_name)
                    width, height = self.dstl.subregion_size(iid)
                    #print('size', width, height)
                    polygons_to_composite(poly, xmax, ymin, width, height, view_fp, class_color=self.info.class_color, 
                        class_zorder = self.info.class_zorder) #FIXME

                progress('done')
            progress('done')
            
    def predict_subregion(self, model, iid, batchsize):
        width, height = self.dstl.subregion_size(iid)
        tilesize = model_tilesize(model)
        tileborder = model_tileborder(model)
        
        # Add border of tilesize, since tiles don't line up with image coordinates
        targets = np.zeros( (10, height+tilesize*2, width+ tilesize*2) )

        tiles = self.tile(iid, tilesize, train=False)
        
        #FIXME
        L = len(tiles)
        D = roundup(L,batchsize) -L
        tiles.extend(tiles[0:D])

        irow, re, icol, ce = self.dstl.subregion_slice(iid)

        for n in range(0, len(tiles), batchsize) :
            progress()
            batch_coords = tiles[n:n+batchsize]
            data, labels = self.make_batch(batch_coords, tilesize, tileborder, augment=0.0)

            results = model.predict_on_batch(data)

            for c in range(0, batchsize ) :
                region, rrow, rcol = tiles[n+ c] #todo check order row col
                crow = rrow-irow + tilesize
                ccol = rcol-icol + tilesize
        
                targets[:, crow: crow+tilesize, ccol:ccol+tilesize] = results[c]

        # Trim boundary
        targets = targets[:, tilesize:-tilesize, tilesize:-tilesize]
    
        return targets
    

    def evaluate(self, model_name) :
        fin = self.path('predictions', model_name=model_name)
        
        actual = self.info.wkt
        predicted = WktPolygons(fin) 
        imageIds = predicted.imageIds()
    
        tp = {}
        fn = {}
        fp = {}
        for ct in self.info.class_types :
            tp[ct] = 0.0
            fn[ct] = 0.0
            fp[ct] = 0.0
    
    
        print('# class     \t Jaccard\t True Pos. \t False Pos. \t False Neg.')
        for iid in imageIds :
            print('# '+ iid)
            actual_polygons = actual.polygons(iid)
            predicted_polygons = predicted.polygons(iid)
        
            for ct in self.info.class_types :
                jaccard, true_pos, false_pos, false_neg = polygon_jaccard(actual_polygons[ct], predicted_polygons[ct])
                print('{:>2} {:<9}\t{:5.3f}\t{:12.7g}\t{:12.7g}\t{:12.7g}'.format(ct, self.info.class_shortname[ct], jaccard, true_pos, false_pos, false_neg) )
                tp[ct] += true_pos
                fp[ct] += false_pos
                fn[ct] += false_neg
            print()
     
        print('# Class averages' ) 
        avg_jac = 0.0
        for ct in self.info.class_types :
            union = tp[ct] + fn[ct] + fp[ct] + 10e-100 # epsilon
            jac = tp[ct]/union
            avg_jac += jac
            print('{:>2} {:<9}\t{:5.3f}\t{:12.7g}\t{:12.7g}\t{:12.7g}'.format(ct, self.info.class_shortname[ct],  jac, tp[ct], fp[ct], fn[ct]) )
   
        avg_jac /= 10.
        print()
        print('Jaccard\t{:.3f}'.format(avg_jac) )
    
    
    
# ---------- Command Line Interface ----------
def _cli():
      
    def add_cmd_train(cmdparser):
        parser = cmdparser.add_parser(
            'train',
            help='Train models')
        parser.set_defaults(funcname='train')

        parser.add_argument(
            'model_name',
            action='store', 
            metavar = 'MODEL_NAME',
            help = "model to train")

        parser.add_argument(
            '-O', '--optimizer', 
            action='store', 
            dest = 'optimizer',
            default = OPTIMIZER,
            metavar = 'OPTIMIZER',
            help='Optimizer')

        parser.add_argument(
            '-F','--factor', 
            action='store', 
            dest = 'lr_factor',
            default = 1.0,
            metavar = 'FLOAT',
            type = float,
            help='Learning rate factor.')

        parser.add_argument(
            '-B', '--batchsize', 
            action='store', 
            dest = 'batchsize',
            default = BATCHSIZE,
            type = int,
            help='batchsize')

        parser.add_argument(
            '-A', '--augment', 
            action='store', 
            default = 0.0,
            type = float,
            help='Augment data fraction')

        parser.add_argument(
            '-W', '--weights',  
            action='store', 
            type = str,
            default = 'uniform',
            help='Class weights ')
        
 
    def add_cmd_predict(cmdparser) :
        parser = cmdparser.add_parser(
            'predict',
            help='Predict models')
        parser.set_defaults(funcname='predict')
         
        parser.add_argument(
            'model_name',
            metavar = 'NAME',
            help = "Model name.")
        
        parser.add_argument(
            '--subregions',
            nargs='+',
            metavar = 'imageIds',
            default = ('validation',) ,    #FIXME
            help = "List of subregions to process. (Also accepts 'train' and 'test'. ") 

        parser.add_argument(
            '-C', '--composite',
            action='store_true',
            default = False,
            help='Create PNG composite of preditions')

        parser.add_argument(
            '-B', '--batchsize', 
            action='store', 
            dest = 'batchsize',
            default = BATCHSIZE,
            type = int,
            help='batchsize')

    def add_cmd_evaluate(cmdparser):
        parser = cmdparser.add_parser(
            'evaluate',
            help='Validate predictions')
        parser.set_defaults(funcname='evaluate')
         
        parser.add_argument(
            'model_name',
            metavar = 'NAME',
            help = "Model name or previous predictions.")
            
            
        
    parser = argparse.ArgumentParser(
        description=__description__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cmdparser = parser.add_subparsers(
        title='Commands',
        description=None,
        help="-h, --help Additional help",)

    _dstl._add_argument_version(parser)

    _dstl._add_argument_sourcedir(parser)
    _dstl._add_argument_outputdir(parser)
    _dstl._add_argument_scale(parser)
    _dstl._add_argument_randomize(parser)
    
    add_cmd_train(cmdparser)
    add_cmd_predict(cmdparser)
    add_cmd_evaluate(cmdparser)
      
    # Run command
    opts = vars(parser.parse_args())

    sourcedir = opts.pop('sourcedir')
    outputdir = opts.pop('outputdir')
    scale = opts.pop('scale')
    
    funcname = opts.pop('funcname')
    randomize = opts.pop('randomize')
    if randomize: 
        random.seed()

    engine = DstlEngine(sourcedir=sourcedir, outputdir=outputdir, scale=scale)
    func = getattr(engine, funcname)
    func(**opts)


if __name__ == "__main__":
    _cli()




    
