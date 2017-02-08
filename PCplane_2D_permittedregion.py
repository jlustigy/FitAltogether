import numpy as np
import sys, getopt
import datetime

RESOLUTION=100

# Specify directory of run to analyze
MCMC_DIR = "mcmc_output/2016-07-13--11-59/"

DIR = "mcmc_output/"

# Specify burn-in index for corner plot
DEFAULT_BURN_INDEX = 0


#===================================================
if __name__ == "__main__":

    # Read command line args
    myopts, args = getopt.getopt(sys.argv[1:],"d:")
    run = ""
    for o, a in myopts:
        # o == option
        # a == argument passed to the o
        # Get MCMC directory timestamp name
        if o == '-d':
            run=a
        else:
            print("Please specify run directory using -d: \n e.g. >python mcmc_physical.py -d 2016-07-13--11-59")
            sys.exit()

    MCMC_DIR = DIR + run + "/"

    # read PCs
    V_nj = np.loadtxt( MCMC_DIR+'PCA_V_jn' ).T
    PC1 = V_nj[0]
    PC2 = V_nj[1]
    n_band = len( PC1 )
    band_ticks = np.arange( n_band )

    ave_j = np.loadtxt( MCMC_DIR+'AVE_j' )

    x_ticks = np.linspace(-0.5,0.5,RESOLUTION)
    y_ticks = np.linspace(-0.5,0.5,RESOLUTION)
    x_mesh, y_mesh, band_mesh = np.meshgrid( x_ticks, y_ticks, band_ticks, indexing='ij' )

    vec_mesh = x_mesh * PC1[ band_mesh ] + y_mesh * PC2[ band_mesh ] + ave_j[ band_mesh ]

    for ii in xrange( len( x_ticks ) ) :
        for jj in xrange( len( y_ticks ) ) :
            if np.any( vec_mesh[ii][jj] < 0. ) :
                print x_ticks[ii], y_ticks[jj], 1
            elif np.any( vec_mesh[ii][jj] > 1. ) :
                print x_ticks[ii], y_ticks[jj], 2
            else :
                print x_ticks[ii], y_ticks[jj], 0
        print ''


