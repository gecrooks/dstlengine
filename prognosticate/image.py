#!/usr/bin/env python
    
from core_dstl import *
from core_dstl import __version__, __description__, _DATADIR, _SOURCEDIR


# ----------   Command Line Interface ----------    
 
def _cli():    
    # Parse command line arguments and run command    
    parser = _arg_parser()
    opts = vars(parser.parse_args())    
    
    datadir = opts.pop('datadir')
    sourcedir = opts.pop('sourcedir')
    verbose = not opts.pop('quite')
    
    with Dstl(datadir=datadir, sourcedir=sourcedir, verbose=verbose) as dstl:
        dstl.view(**opts)

            
def _arg_parser():   
    import argparse  
    main_parser = argparse.ArgumentParser(
                    description=__description__,
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
                    

                                        
    main_parser.add_argument('--version', action='version', version=__version__)
        
    # ===== Common options =====
    main_parser.add_argument('-d', '--data', 
                          action='store', 
                          dest = 'datadir',
                          default = _DATADIR,
                          metavar = 'PATH',
                          help='Location to store and process data')
 
    main_parser.add_argument('-s', '--source', 
                       action='store', 
                       dest = 'sourcedir',
                       default = _SOURCEDIR,
                       metavar = 'PATH',
                       help='Location of input data')
 
 
    main_parser.add_argument('-q', '--quite', 
                       action='store_true', 
                       dest = 'quite',
                       default = False,
                       help='shhh')


    main_parser.add_argument(
        'imageIds',
        nargs='+',
        metavar = 'imageIds',
        help = "Optinal list of regions to process. ")

    main_parser.add_argument(
        '--scale', 
        action='store', 
        type=float,
        help='Rescale image. (Default is 0.2 for regions, 1.0 for subregions) ')
                            
    main_parser.add_argument(
        '--composite', 
        action='store_true', 
        default=False, 
        help='')
    
 
    return main_parser
    # End construction of argument parser

   

if __name__ == "__main__": _cli()