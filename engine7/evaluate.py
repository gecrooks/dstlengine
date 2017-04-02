#!/usr/bin/env python

from dstl import *   
import argparse   


def evaluate(wktfn, sourcedir, outputdir, scale) :
    dstl = DstlDataset(sourcedir=sourcedir, outputdir=outputdir, scale=scale)

    actual = dstl.info.wkt
    predicted = WktPolygons(wktfn) 
    
    imageIds = predicted.imageIds()
    
    tp = {}
    fn = {}
    fp = {}
    for ct in dstl.info.class_types :
        tp[ct] = 0.0
        fn[ct] = 0.0
        fp[ct] = 0.0
    
    
    print('# class     \t Jaccard\t True Pos. \t False Pos. \t False Neg.')
    for iid in imageIds :
    #for iid in ['6120_2_0',]:
        print('# '+ iid)
        actual_polygons = actual.polygons(iid)
        predicted_polygons = predicted.polygons(iid)
        
        for ct in dstl.info.class_types :
            jaccard, true_pos, false_pos, false_neg = polygon_jaccard(actual_polygons[ct], predicted_polygons[ct])
            print('{:>2} {:<9}\t{:5.3f}\t{:12.7g}\t{:12.7g}\t{:12.7g}'.format(ct, dstl.info.class_shortname[ct], jaccard, true_pos, false_pos, false_neg) )
            tp[ct] += true_pos
            fp[ct] += false_pos
            fn[ct] += false_neg
        print()
     
    print('# Class averages' ) 
    avg_jac = 0.0
    for ct in dstl.info.class_types :
        union = tp[ct] + fn[ct] + fp[ct] + 10e-100 # epsilon
        jac = tp[ct]/union
        avg_jac += jac
        print('{:>2} {:<9}\t{:5.3f}\t{:12.7g}\t{:12.7g}\t{:12.7g}'.format(ct, dstl.info.class_shortname[ct],  jac, tp[ct], fp[ct], fn[ct]) )
   
    avg_jac /= 10.
    print()
    print('Jaccard\t{:.3f}'.format(avg_jac) )
            
            
            
def _cli():    

    parser = argparse.ArgumentParser(
                  #  description=__description__,
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ===== Common options =====

    parser.add_argument('wktfn',
                        metavar = 'FILENAME',
                        help = "WKT Polygons filename")

    import dstl as _dstl 
    _dstl._add_argument_version(parser)

    _dstl._add_argument_sourcedir(parser)
    _dstl._add_argument_outputdir(parser)
    _dstl._add_argument_scale(parser)
    #_dstl._add_argument_randomize(parser)

    opts = vars(parser.parse_args())    
    evaluate(**opts)
  
       
            
if __name__ == "__main__":
    _cli()




            