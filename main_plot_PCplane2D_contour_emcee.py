import numpy as np
import sys
import corner
import datetime
import os

import emcee

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import prior
import reparameterize

import PCA
import shrinkwrap

import geometry

LOOP_MAX  = 10000
COUNT_MAX = 100

SEED_in  = 2013

deg2rad = np.pi/180.

LAMBDA_CORR_DEG = 30.
LAMBDA_CORR = LAMBDA_CORR_DEG * deg2rad

RESOLUTION=100

# GEOM = ( lat_o, lon_o, lat_s, lon_s, omega )
# GEOM = ( 0., 0., 0.,  90., 2.*np.pi )

N_SIDE   = 32
INFILE_DIR = 'mockdata/'
# INFILE = 'mockdata_45deg_time23_l`xreplc'
INFILE = 'mockdata_135deg_3types_redAmerica_t12_lc'
OUTFILE_DIR = 'PCplane/'

ALBDFILE = 'mockdata/mockdata_135deg_3types_redAmerica_t12_band_sp'
AREAFILE = 'mockdata/mockdata_135deg_3types_redAmerica_t12_factor'


VARAMP = 0.01

N_TYPE = 3
N_CPU  = 2
N_MCMC = 1000

X_MIN = -0.2
Y_MIN = -0.2
X_MAX = 0.4
Y_MAX = 0.8


#===================================================
# basic functions
#=============================================== ====

np.random.seed(SEED_in)



#---------------------------------------------------
def lnprob( points_kn_flat, inv_cov, det_cov, M_j, U_iq ):


    points_kn = points_kn_flat.reshape([ N_TYPE, N_TYPE-1 ])

#    print 'points_kn', points_kn

#    print 'points', points_kn[2]

    # construct area fraction
    points_kq = np.c_[ points_kn, np.ones( len( points_kn ) ) ]
    

    # if the points do not form a triangle
    if np.linalg.det( points_kq ) == 0. :
        
        term = -1e10

    else :

        # construct albedo
        X_albd_kj = np.dot( points_kn, V_nj ) + M_j

        # if albedo is not between 0 and 1, discard
        if ( np.any( X_albd_kj < 0. ) or np.any( X_albd_kj > 1. ) ) :

            term = -1e10

        # if the data points are not enclosed by the three points, discard
        else :

            x_area_ik  = np.dot( U_iq, np.linalg.inv( points_kq ) )

            if ( np.any( x_area_ik < 0. ) ) :

                term = -1e10

            else :

                x_area_ave = np.average( x_area_ik, axis=0 )        
                x_area_std = np.std( x_area_ik, axis=0 )
                dx_area_ik = ( x_area_ik - x_area_ave )/x_area_std
                term1 = -0.5 * np.sum( np.diag( np.dot( np.dot( dx_area_ik.T, inv_cov ), dx_area_ik ) ) )
                term2 = -0.5 * np.log( det_cov )
                term  = term1 + term2

    if term == np.nan :
        print 'term1, term2', term1, term2

    return term


#---------------------------------------------------
def lnprob_old( points_kn_flat, inv_cov, det_cov, M_j, U_iq ):

    points_kn = points_kn_flat.reshape([ N_TYPE, N_TYPE-1 ])

    # construct albedo
    X_albd_kj = np.dot( points_kn, V_nj ) + M_j

    # construct area fraction
    points_kq = np.c_[ points_kn, np.ones( len( points_kn ) ) ]
    x_area_ik  = np.dot( U_iq, np.linalg.inv( points_kq ) )

    # if albedo is not between 0 and 1, discard
    if ( np.any( X_albd_kj < 0. ) or np.any( X_albd_kj > 1. ) ) :

        term1 = -1.e20

    # if the data points are not enclosed by the three points, discard
    elif ( np.any( x_area_ik < -0. ) ) :

        term1 = -1.e20

    else :

        x_area_ave = np.average( x_area_ik, axis=0 )        
        x_area_std = np.std( x_area_ik, axis=0 )
        dx_area_ik = ( x_area_ik - x_area_ave )/x_area_std

        term1 = -0.5 * np.sum( np.diag( np.dot( np.dot( dx_area_ik.T, inv_cov ), dx_area_ik ) ) )

    return term1

#--------------------------------------------------
def allowed_region( V_nj, ave_j ):

    # read PCs
    PC1 = V_nj[0]
    PC2 = V_nj[1]
    n_band = len( PC1 )
    band_ticks = np.arange( n_band )

    x_ticks = np.linspace(X_MIN,X_MAX,RESOLUTION)
    y_ticks = np.linspace(Y_MIN,Y_MAX,RESOLUTION)
    x_mesh, y_mesh, band_mesh = np.meshgrid( x_ticks, y_ticks, band_ticks, indexing='ij' )
    vec_mesh = x_mesh * PC1[ band_mesh ] + y_mesh * PC2[ band_mesh ] + ave_j[ band_mesh ]

    x_grid, y_grid = np.meshgrid( x_ticks, y_ticks, indexing='ij' )
    prohibited_grid = np.zeros_like( x_grid )

    for ii in xrange( len( x_ticks ) ) :
        for jj in xrange( len( y_ticks ) ) :

            if np.any( vec_mesh[ii][jj] < 0. ) :
                prohibited_grid[ii][jj] = 1
                if np.any( vec_mesh[ii][jj] > 1. ) :
                    prohibited_grid[ii][jj] = 3
            elif np.any( vec_mesh[ii][jj] > 1. ) :
                prohibited_grid[ii][jj] = 2
            else :
                prohibited_grid[ii][jj] = 0

    return x_grid, y_grid, prohibited_grid


def generate_cmap(colors):
    """
    copied from) http://qiita.com/kenmatsu4/items/fe8a2f1c34c8d5676df8
    """
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)


#===================================================
if __name__ == "__main__":


    # Load input data
    Obs_ij = np.loadtxt( INFILE_DIR + INFILE )
    Time_i  = np.arange( len( Obs_ij ) ) / ( 1.0 * len( Obs_ij ) )
    n_band = len( Obs_ij.T )

    # PCA
    print 'Performing PCA...'
    n_pc, V_nj, U_in, M_j = PCA.do_PCA( Obs_ij, E_cutoff=1e-2, output=True )

    if not ( n_pc == N_TYPE - 1 ) :
        print 'ERROR: This code is only applicable for 3 surface types!'
        sys.exit()


    # flipping
#    V_nj[0] = -1. * V_nj[0]
#    U_in.T[0] = -1. * U_in.T[0]
    V_nj[1] = -1. * V_nj[1]
    U_in.T[1] = -1. * U_in.T[1]


    U_iq = np.c_[ U_in, np.ones( len( U_in ) ) ]

    l_dim   = len( Obs_ij )
    cov     = prior.get_cov( 1.0, 0.0, LAMBDA_CORR, l_dim, periodic=False )
    inv_cov = np.linalg.inv( cov )
    det_cov = np.linalg.det( cov )

    print 'det_cov', det_cov
    print 'inv_cov', inv_cov

    params = ( inv_cov, det_cov, M_j, U_iq )

    # Number of walkers
    n_dim     = N_TYPE * ( N_TYPE - 1 )
    n_walkers = 2*n_dim**2



    #--------------------------------------------------------------------------
    # answer
    # projection of 'answer' onto PC plane
    albd_answer_kj  = np.loadtxt( ALBDFILE ).T
    dalbd_answer_kj = albd_answer_kj - M_j
    coeff_kn = np.dot( dalbd_answer_kj, V_nj.T )

    points_kn_flat_answer = coeff_kn[:,0:2].flatten()
    p0_offset = np.tile( points_kn_flat_answer, [ n_walkers, 1 ] )

    p0 = p0_offset + np.random.normal( loc=0.0, scale=0.01, size=( n_walkers, n_dim ) )

    # Initialize emcee EnsembleSampler object
    sampler = emcee.EnsembleSampler( n_walkers, n_dim, lnprob, args=params, threads=N_CPU )
    sampler.run_mcmc( p0, N_MCMC )

    # points_kn = points_kn.reshape( N_TYPE, N_TYPE-1 )
    myrange = [[X_MIN, X_MAX], [Y_MIN, Y_MAX]]
    H, xedges, yedges = np.histogram2d( sampler.flatchain[:,4], sampler.flatchain[:,5], range=myrange, bins=100 )
    X, Y = np.meshgrid( xedges, yedges, indexing='ij' )


    chains = sampler.flatchain[:,4].reshape( [ n_walkers, N_MCMC ] )
    xxx    = np.arange( N_MCMC )
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot( 111 )

    for ii in xrange( len( chains ) ) : 
        ax.plot( xxx, chains[ii] )

    plt.show()

    sys.exit()
 
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot( 111 )

    ax.set_xlabel( 'PC 1' )
    ax.set_xlim([X_MIN, X_MAX])
    ax.set_ylabel( 'PC 2' )
    ax.set_ylim([Y_MIN, Y_MAX])
    ax.set_aspect('equal')

    #--------------------------------------------------------------------------
    # allowed region

    x_grid, y_grid, prohibited_grid = allowed_region( V_nj, M_j )
    mycm = generate_cmap(['white', 'gray'])
    plt.pcolor( x_grid, y_grid, prohibited_grid, cmap=mycm )

    plt.pcolormesh( X, Y, H )
    plt.colorbar()

    plt.savefig( INFILE + '_vege.png' )
