#!/usr/bin/env python
    
from core_dstl import *
from core_dstl import __version__, __description__, _datadir, _sourcedir


# ----------   Command Line Interface ----------    
 
def _cli():    
    # Parse command line arguments and run command    
    parser = _arg_parser()
    opts = vars(parser.parse_args())    
    
    funcname = opts.pop('funcname')

    if funcname == 'help' : 
        parser.print_help() 
        sys.exit()
        
    datadir = opts.pop('datadir')
    sourcedir = opts.pop('sourcedir')
    verbose = not opts.pop('quite')

    
    with Dstl(datadir=datadir, sourcedir=sourcedir, verbose=verbose) as dstl:
        func = getattr(dstl, funcname)
        func(**opts)
        
    
        
def _arg_parser():   
    import argparse  
    main_parser = argparse.ArgumentParser(
                    description=__description__,
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
                    
    subparsers = main_parser.add_subparsers(title='Commands',
                                        description=None,
                                        help="-h, --help Additional help",)

                                        
    main_parser.add_argument('--version', action='version', version=__version__)
    
    # If no subcommand print help message
    main_parser.set_defaults(funcname = 'help')     
    
    # ===== Common options =====
    main_parser.add_argument('-d', '--data', 
                          action='store', 
                          dest = 'datadir',
                          default = _datadir,
                          metavar = 'PATH',
                          help='Location to store and process data')
 
    main_parser.add_argument('-s', '--source', 
                       action='store', 
                       dest = 'sourcedir',
                       default = _sourcedir,
                       metavar = 'PATH',
                       help='Location of input data')
 
 
    main_parser.add_argument('-q', '--quite', 
                       action='store_true', 
                       dest = 'quite',
                       default = False,
                       help='shhh')
  
   
    # === Build ===                     
    parser = subparsers.add_parser('all', 
               help='Build main data structures (data, alignment, targets, composites)')         
    parser.set_defaults(funcname='build')
   
   

    # === build data ===
    parser = subparsers.add_parser('data', 
                    help='Process raw dstl data into a 20 band 5km x 5km array for each of the 18 regions.') 
                       
    parser.set_defaults(funcname ='build_data')
    
    parser.add_argument('regions',
                        nargs='*',
                        metavar = 'regionId',
                        help = "Optinal list of regions to process. "
                                "Defaults to all regions. Will also accept 'all' and 'train'")

    parser.add_argument('-t','--type',
                        action='store',
                        default='all',
                        dest = 'imageType',
                        help='image type (3,P,M,A,all)')


    # === register images ===
    parser = subparsers.add_parser('alignment',
                    help='Register A, M, and P bands to 3-band image')
    parser.set_defaults(funcname='build_register')
    
    parser.add_argument('regions',
                        nargs='*',
                        metavar = 'regionId',
                        help = "Optional list of regions to process. "
                                "Defaults to all regions. Will also accept 'all' ")
                        
    parser.add_argument('-t','--type',
                        action='store',
                        default='all',
                        dest = 'imageType',
                        help='image type (P,M,A,all)')
    
    parser.add_argument('--dry_run',
                        action='store_true',
                        default=False,
                        help='')


    # === build targets ===
    parser = subparsers.add_parser('targets',
                    help='Process raw dstl data to create target data')
    parser.set_defaults(funcname='build_targets')

    parser.add_argument(
        'regions',
        nargs='*',
        metavar = 'regionId',
        help = "Optional list of regions to process. "
              "Defaults to all regions. Will also accept 'all' ")
              

    # === build composites ===
    parser = subparsers.add_parser('composites',
                    help='Create composite images of target masks')
    parser.set_defaults(funcname='build_composites')
    
    parser.add_argument(
        'regions',
        nargs='*',
        metavar = 'regionId',
        help = "Optional list of regions to process. "
              "Defaults to all regions. Will also accept 'all' ")
    
    parser.add_argument(
        '--fill',
        dest = 'outline',
        action='store_false',
        default=True,
        help='Filled masks (rather than outlines) ')  
 
   
    # === create images ===
    parser = subparsers.add_parser('view',
         help='Convert region data to viewable graphics in png format',
         )
    parser.set_defaults(funcname='view')

    parser.add_argument(
        'imageIds',
        nargs='+',
        metavar = 'imageIds',
        help = "Optinal list of regions to process. ")

    parser.add_argument(
        '--scale', 
        action='store', 
        type=float,
        help='Rescale image. (Default is 0.2 for regions, 1.0 for subregions) ')
                            
    parser.add_argument(
        '--composite', 
        action='store_true', 
        default=False, 
        help='')
    
 
    return main_parser
    # End construction of argument parser

   

if __name__ == "__main__": _cli()