import numpy as np
import sys
import corner
import datetime
import os
import pdb

import emcee

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


from scipy.interpolate import griddata

import prior
import reparameterize

import PCA
import shrinkwrap

import geometry


import multiprocessing
import functools

from numba import jit

LOOP_MAX  = 10000
COUNT_MAX = 100

SEED_in  = 2015

deg2rad = np.pi/180.

LAMBDA_CORR_DEG = 30.
LAMBDA_CORR = LAMBDA_CORR_DEG * deg2rad



# GEOM = ( lat_o, lon_o, lat_s, lon_s, omega )
# GEOM = ( 0., 0., 0.,  90., 2.*np.pi )

N_SIDE   = 32
INFILE_DIR = 'mockdata/'
#INFILE = 'mockdata_90deg_3types_t12_lc'
INFILE = "lc_test1.txt"
OUTFILE_DIR = 'PCplane/'

ALBDFILE = 'mockdata/mockdata_90deg_3types_t12_band_sp'
AREAFILE = 'mockdata/mockdata_90deg_3types_t12_factor'

VARAMP = 0.01

N_TYPE = 3
N_CPU  = 6
N_MCMC = 100

X_MIN = -0.3
X_MAX = 1.5
# X_NUM = 30 + 1
X_NUM = 50#18 + 1
Y_MIN = -1.0
Y_MAX = 0.5
Y_NUM = 50#30 + 1
# Y_NUM = 50 + 1

#===================================================
# basic functions
#=============================================== ====

np.random.seed(SEED_in)


#---------------------------------------------------
#@jit
def function( points_kn, params ):

#    print 'working on ', indx
    inv_cov, det_cov, M_j, U_iq = params

    # construct area fraction
    points_kq = np.c_[ points_kn, np.ones( len( points_kn ) ) ]

    # if the points do not form a triangle
    if np.linalg.det( points_kq ) == 0. :

        term = 0.

    else :

        #print U_iq.shape, points_kq.shape
        x_area_ik  = np.dot( U_iq, np.linalg.inv( points_kq ) )
        #x_area_ik  = np.linalg.solve(points_kq.T, U_iq.T).T

        if ( np.any( x_area_ik < 0. ) ) :

            # print 'x_area_ik', x_area_ik
            # print 'points_kn', points_kn
            term = 0.

        else :

            term = 1.

    return term



#---------------------------------------------------
@jit
def multicore_function( indx, params ):

    print('indx', indx)
#    print 'working on ', indx
    points_valid_pn, inv_cov, det_cov, M_j, U_iq = params

    params2 = ( inv_cov, det_cov, M_j, U_iq )
    point1_n = points_valid_pn[indx]

    terms = 0.
    for p2 in xrange( len( points_valid_pn ) ):

        point2_n = points_valid_pn[p2]

        for p3 in xrange( len( points_valid_pn ) ):

            point3_n = points_valid_pn[p3]
            points_kn = np.vstack( [ point1_n, point2_n, point3_n ] )
            terms += function( points_kn, params2 )

    return terms




#=============================================================================
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



#=============================================================================
def call_multicore( list_index, args ):

    p = multiprocessing.Pool( N_CPU )
    result = p.map( functools.partial( multicore_function, params=args ), list_index )
    return result

@jit
def call_looper( list_index, args ):

    results = np.zeros(len(list_index))
    for i in range(len(list_index)):
        results[i] = multicore_function(list_index[i], args)

    return results


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
    V_nj[0] = -1. * V_nj[0]
    U_in.T[0] = -1. * U_in.T[0]
#    V_nj[1] = -1. * V_nj[1]
#    U_in.T[1] = -1. * U_in.T[1]

    # longitudinal slices
    #--------------------------------------------------------------------------
    # longitudinal colors
    albd_answer_kj  = np.loadtxt( ALBDFILE ).T
    type_kl        = np.loadtxt( 'IGBP_lon.txt' ).T[1:]
    type_lk        = type_kl.T
    color_lj       = np.dot( type_lk, albd_answer_kj )
    U_in           = np.dot( ( color_lj - M_j ), V_nj.T )
    # picking up
    U_in = U_in[::15,:]


    U_iq = np.c_[ U_in, np.ones( len( U_in ) ) ]

    l_dim   = len( Obs_ij )
    cov     = prior.get_cov( 1.0, 0.0, LAMBDA_CORR, l_dim, periodic=False )
    inv_cov = np.linalg.inv( cov )
    det_cov = np.linalg.det( cov )

#    print 'det_cov', det_cov
#    print 'inv_cov', inv_cov

    x_ticks = np.linspace( X_MIN, X_MAX, X_NUM )
    y_ticks = np.linspace( Y_MIN, Y_MAX, Y_NUM )

    area = ( Y_MAX - Y_MIN ) / ( Y_NUM - 1 ) * ( X_MAX - X_MIN ) / ( X_NUM - 1 )

    xindx_ticks = np.arange( X_NUM )
    yindx_ticks = np.arange( Y_NUM )

    x_ticks_mesh, y_ticks_mesh = np.meshgrid( x_ticks, y_ticks, indexing='ij' )

    points_pn = np.c_[ x_ticks_mesh.flatten(), y_ticks_mesh.flatten() ]

    # construct albedo
    albd_pj = np.dot( points_pn, V_nj ) + M_j
    list_indx = np.where( np.all( 0 < albd_pj, axis=1 ) * np.all( albd_pj < 1, axis=1 ) )[0]

    points_valid_pn = points_pn[ list_indx ]

    print 'len(points_pn)', len(points_pn)
    print 'len(points_valid_pn)', len(points_valid_pn)

    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")

    # multiprocessing
    params = ( points_valid_pn, inv_cov, det_cov, M_j, U_iq )
    list_index = np.arange( len( points_valid_pn ) )
    list_term_p = call_multicore( list_index, params )
    #list_term_p = multicore_function( list_index[0], params )
    #import pdb; pdb.set_trace()
    #list_term_p = call_looper( list_index, params )
    terms_p     = np.array( list_term_p )

    print terms_p

    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")

    z_mesh = griddata( points_valid_pn, terms_p, points_pn, method='linear', fill_value=0. )

    z_mesh = z_mesh.reshape( x_ticks_mesh.shape ) / np.sum( z_mesh ) / area



    #--------------------------------------------------------------------------
    # setting up the figure
    fig = plt.figure(figsize=(3, 6) )
    ax = fig.add_subplot( 111, aspect='equal' )
    ax.set_xlabel( 'PC 1' )
    ax.set_xlim([X_MIN, X_MAX])
    #ax.set_xticks([ -0.2, 0.0, 0.2, 0.4 ])
    ax.set_ylabel( 'PC 2' )
    ax.set_ylim([Y_MIN, Y_MAX])

#    #--------------------------------------------------------------------------
#    # longitudinal colors
#    type_kl        = np.loadtxt( 'IGBP_lon.txt' ).T[1:]
#    type_lk        = type_kl.T
#    color_lj       = np.dot( type_lk, albd_answer_kj )
#    U_ln           = np.dot( ( color_lj - M_j ), V_nj.T )
#    ax.plot( U_ln.T[0], U_ln.T[1], color='k'  )

    #--------------------------------------------------------------------------
    # contour
    colormap = ax.pcolor( x_ticks_mesh, y_ticks_mesh, z_mesh, cmap=cm.spectral )
    divider  = make_axes_locatable(ax)
    cax  = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar( colormap, cax=cax )

    #--------------------------------------------------------------------------
    # lightcurves
    ax.plot( U_in.T[0], U_in.T[1], color='white', linewidth=1.5 )

    #--------------------------------------------------------------------------
    # start from answer region
    # projection of 'answer' onto PC plane
    albd_answer_kj  = np.loadtxt( ALBDFILE ).T
    dalbd_answer_kj = albd_answer_kj - M_j
    coeff_kn        = np.dot( dalbd_answer_kj, V_nj.T )
    answer_x, answer_y = coeff_kn[:,0:2].T

    ax.scatter( answer_x[0], answer_y[0], marker='o', c='white', s=40  )
    ax.scatter( answer_x[1], answer_y[1], marker='s', c='white', s=30  )
    ax.scatter( answer_x[2], answer_y[2], marker='^', c='white', s=40  )

    # ax.scatter( answer_x[0], answer_y[0], marker='o', c='blue'  )
    # ax.scatter( answer_x[1], answer_y[1], marker='s', c='red'   )
    # ax.scatter( answer_x[2], answer_y[2], marker='^', c='green' )


    #--------------------------------------------------------------------------
    # allowed region
    x_ticks = np.linspace(X_MIN,X_MAX,100)
    for jj in xrange( len( V_nj.T ) ):

        if V_nj[1][jj] > 0. :
            lower_boundary = (     - 1.0 * x_ticks * V_nj[0][jj] - M_j[jj] ) / V_nj[1][jj]
            upper_boundary = ( 1.0 - 1.0 * x_ticks * V_nj[0][jj] - M_j[jj] ) / V_nj[1][jj]
        else :
            lower_boundary = ( 1.0 - 1.0 * x_ticks * V_nj[0][jj] - M_j[jj] ) / V_nj[1][jj]
            upper_boundary = (     - 1.0 * x_ticks * V_nj[0][jj] - M_j[jj] ) / V_nj[1][jj]
        ax.fill_between( x_ticks, -10., lower_boundary, color='gray', facecolor='gray' )
        ax.fill_between( x_ticks, upper_boundary,  10., color='gray', facecolor='gray' )


#    plt.savefig( INFILE + '_noreg.pdf', bbox_inches='tight' )
    plt.savefig( 'IGBP_lon_noreg.pdf', bbox_inches='tight' )
