import numpy as np
import sys
import corner
import datetime
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.optimize import minimize
import prior
import reparameterize

import PCA
import shrinkwrap

import geometry

LOOP_MAX  = 10000
COUNT_MAX = 100

SEED_in  = 2015

LAMBDA_CORR_DEG = np.array( [ 30., 30., 30. ] )
# LAMBDA_CORR_DEG = np.array( [ 40., 40., 45. ] )
# LAMBDA_CORR_DEG = np.array( [ 35., 35., 45. ] )
# LAMBDA_CORR_DEG = np.array( [ 40., 40., 40. ] )
# LAMBDA_CORR_DEG = np.array( [ 120., 120., 120. ] )
# LAMBDA_CORR_DEG = np.array( [ 30., 30., 30. ] )
# LAMBDA_CORR_DEG = np.array( [ 55., 55., 65. ] )
# LAMBDA_CORR_DEG = np.array( [ 40., 40., 45. ] )
# LAMBDA_CORR_DEG = np.array( [ 50., 50., 60. ] )
LIST_LAMBDA_CORR = LAMBDA_CORR_DEG * ( np.pi/180. )

RESOLUTION=100

GEOM = ( 0., 0., 0.,  90., 2.*np.pi )

N_SIDE   = 32
INFILE_DIR = 'mockdata/'
# INFILE = 'mockdata_45deg_time23_l`xreplc'
INFILE = 'mockdata_90deg_3types_t12_lc'
OUTFILE_DIR = 'PCplane/'

ALBDFILE = 'mockdata/mockdata_90deg_3types_t12_band_sp'
AREAFILE = 'mockdata/mockdata_90deg_3types_t12_factor'

deg2rad = np.pi/180.

#===================================================
# basic functions
#=============================================== ====

np.random.seed(SEED_in)

#---------------------------------------------------
def regularize_area_GP_order( x_area_lk, points_kn, type='squared-exponential', ans=False ):

    sum_term  = 0.
    sum_term1 = 0.
    sum_term2 = 0.

    x_area_ave = np.average( x_area_lk, axis=0 )

    x_area_std = np.std( x_area_lk, axis=0 )
    dx_area_lk = ( x_area_lk - x_area_ave )/x_area_std

    list_arg = np.argsort( x_area_ave )

    print 'x_area_ave', x_area_ave
    print 'x_area_std', x_area_std

#    for kk in xrange( len( points_kn ) ):
    for kk in list_arg[1:] :

        print 'kk', kk
        print'point ', points_kn[kk]

        if ans :
            lambda_angular = LIST_LAMBDA_CORR[kk]
        else :
            if points_kn[kk][1] == np.max( points_kn.T[1] ) :
                lambda_angular = LIST_LAMBDA_CORR[-1]
            else :
                lambda_angular = LIST_LAMBDA_CORR[0]

        l_dim = len( x_area_lk )
        cov = prior.get_cov( 1.0, 0.0, lambda_angular, l_dim, type=type, periodic=False )

        inv_cov = np.linalg.inv( cov )
        det_cov = np.linalg.det( cov )

        if ( det_cov == 0. ) or ( det_cov < 0. ) :
            print 'det_cov', det_cov
            print 'cov', cov

        dx_area_l = dx_area_lk.T[kk]
        term1 = 0.5*np.dot( dx_area_l, np.dot( inv_cov, dx_area_l ) )
        term2 = 0.5 * np.log( det_cov )

        sum_term1 += term1
        sum_term2 += term2

        sum_term  += term1 + term2

    return sum_term, sum_term1, sum_term2



#--------------------------------------------------
def allowed_region( V_nj, ave_j ):

    # read PCs
    PC1 = V_nj[0]
    PC2 = V_nj[1]
    n_band = len( PC1 )
    band_ticks = np.arange( n_band )

    x_ticks = np.linspace(-0.4,0.2,RESOLUTION)
    y_ticks = np.linspace(-0.2,0.4,RESOLUTION)
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




def func( flatten_array, n_slice, Obs_ij, Kernel_il, X_albd_kj ):

    x_area = flatten_array.reshape([ n_slice, 2 ])
    x_area_column3 = np.ones( n_slice ) - np.sum( x_area, axis=1 )
    x_area_lk = np.c_[ x_area, x_area_column3 ]
    if np.any( x_area_lk > 1. ) or np.any( x_area_lk < 0. ):
        return 10.
    else :
        L_ij = np.dot( np.dot( Kernel_il, x_area_lk ), X_albd_kj )
        return np.linalg.norm( Obs_ij - L_ij )

#===================================================
if __name__ == "__main__":


    # Load input data
    Obs_ij = np.loadtxt( INFILE_DIR + INFILE )
    Time_i  = np.arange( len( Obs_ij ) ) / ( 1.0 * len( Obs_ij ) )
    n_band = len( Obs_ij.T )

    # Initialization of Kernel
    print 'Decomposition into time slices...'
    n_slice = len( Time_i )
    Kernel_il = geometry.kernel( Time_i, n_slice, N_SIDE, GEOM )
    print 'Kernel_il', Kernel_il
    Kernel_il[ np.where( Kernel_il < 1e-3 ) ] = 0.
    print 'Kernel_il', Kernel_il

    # PCA
    print 'Performing PCA...'
    n_pc, V_nj, U_in, M_j = PCA.do_PCA( Obs_ij, E_cutoff=1e-2, output=True )

#    V_nj[0] = -1. * V_nj[0]
#    U_in.T[0] = -1. * U_in.T[0]
#    V_nj[1] = -1. * V_nj[1]
#    U_in.T[1] = -1. * U_in.T[1]

    n_type = n_pc + 1
    if n_type != 3 :
        print 'ERROR: This code is only applicable for 3 surface types!'
        sys.exit()

    U_iq = np.c_[ U_in, np.ones( len( U_in ) ) ]

    PC1_limit = [-0.4, 0.2] # manually set for now
    PC2_limit = [-0.1, 0.4] # manually set for now

    points_kn_list     = []
    X_area_lk_list     = []
    X_albd_kj_list     = []
    chi2_list          = []
    ln_prior_area_list = []
    ln_prior_albd_list = []
    regterm_list      = []
    regterm1_list      = []
    regterm2_list      = []


    count = 0
    for loop in xrange( LOOP_MAX ):

        # generate three random points in PC plane ( 3 vertices x 2 PCs )
        points_PC1 = np.random.uniform( PC1_limit[0], PC1_limit[1], 3 )
        points_PC2 = np.random.uniform( PC2_limit[0], PC2_limit[1], 3 )

        points_kn = np.c_[ points_PC1, points_PC2 ]

        # reconstruct albedo
        X_albd_kj = np.dot( points_kn, V_nj ) + M_j

        # if albedo is not between 0 and 1, discard
        # otherwise, proceed
        if not( np.any( X_albd_kj < 0. ) or np.any( X_albd_kj > 1. ) ):

            # construct area fraction
            points_kq = np.c_[ points_kn, np.ones( len( points_kn ) ) ]
#            X_area_lk = np.dot( inv_Kernel_li, np.dot( U_iq, np.linalg.inv( points_kq ) ) )

            X_area_lk = minimize( func, np.zeros( n_slice*2 ), args=( n_slice, Obs_ij, Kernel_il, X_albd_kj ), bounds=[(0,1)]*( n_slice*2 ) )['x'].reshape([ n_slice, 2 ])

            X_area_column3 = np.ones( n_slice ) - np.sum( X_area_lk, axis=1 )
            X_area_lk = np.c_[ X_area_lk, X_area_column3 ]

            print 'X_area_lk', X_area_lk
            
            # If area is not within 0 and 1, discard
            # otherwise, proceed

            if not( np.any( X_area_lk < 0. ) or np.any( X_area_lk > 1. ) ):

                points_kn = np.vstack( [ points_kn, points_kn[0] ] )

                points_kn_list.append( points_kn )
                X_area_lk_list.append( X_area_lk )
                X_albd_kj_list.append( X_albd_kj )

                # chi^2
                Obs_estimate_ij = np.dot( Kernel_il, np.dot( X_area_lk , X_albd_kj ) )
                chi2 = np.sum( ( Obs_ij - Obs_estimate_ij )**2 )
                chi2_list.append( chi2 )
                print chi2

                Y_array = reparameterize.transform_X2Y(X_albd_kj, X_area_lk)

                # flat prior for area fraction
                Y_area_ik = Y_array[n_type*n_band:].reshape([n_slice, n_type-1])
                ln_prior_area = prior.get_ln_prior_area_new( Y_area_ik )
                ln_prior_area_list.append( ln_prior_area )

                # log prior for albedo
                Y_albd_kj = Y_array[0:n_type*n_band].reshape([n_type, n_band])
                ln_prior_albd = prior.get_ln_prior_albd( Y_albd_kj )
                ln_prior_albd_list.append( ln_prior_albd )

                # regularization ?
                # regparam     = ( LAMBDA_CORR )
                # term1, term2 = regularize_area_GP_g( X_area_lk, regparam, type='exponential' )
                term, term1, term2 = regularize_area_GP_order( X_area_lk, points_kn[:-1] )
                regterm_list.append(  term  )
                regterm1_list.append( term1 )
                regterm2_list.append( term2 )

                count = count + 1
                print ''
                print 'count', count
                print 'terms', term1, term2
                if count > COUNT_MAX :
                    break

        loop = loop + 1

    # loop end
    X_area_lk_answer  = np.loadtxt( AREAFILE )

    print 'answer'
    print ''
    print X_area_lk_answer
    print ''
    print 'answer', regularize_area_GP_order( X_area_lk_answer, np.zeros([3,3]), ans=True )
    print ''

    #--------------------------------------------------------------------------
    # set up
    fig = plt.figure(figsize=(3,3)) 
    ax1 = plt.subplot(adjustable='box', aspect=1.0)
    ax1.set_xlim([-0.4,0.2])
    ax1.set_ylim([-0.2,0.4])
    ax1.set_xticks([ -0.4, -0.2, 0.0, 0.2 ])
    ax1.set_yticks([ -0.2, 0.0, 0.2, 0.4 ])

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    dtime = ( LAMBDA_CORR_DEG / 360. )
#    plt.title(r'$\Delta _t =$'+str(dtime) +  " ($\Delta \phi =$" + str(LAMBDA_CORR_DEG) + '$deg$)' )
#    plt.title(r"($\Delta \phi =$" + str(LAMBDA_CORR_DEG) + '$deg$)' )
    plt.title(r'using correct angular separation' )

    #--------------------------------------------------------------------------
    # allowed region
    x_grid, y_grid, prohibited_grid = allowed_region( V_nj, M_j )
    mycm = generate_cmap(['white', 'gray'])
    plt.pcolor( x_grid, y_grid, prohibited_grid, cmap=mycm )

    #--------------------------------------------------------------------------
    # spider graph (?)

#    colorterm = np.array( regterm1_list ) + np.array( regterm2_list )
    colorterm = np.array( regterm_list )
    colorrange = ( np.max( colorterm ) - np.min( colorterm ) )
    colorlevel = ( colorterm - np.min( colorterm ) ) / colorrange

    print ''
    print 'MINIMUM:', np.min( colorterm )
    print 'MAXIMUM:', np.max( colorterm )
    print ''


    colorlevel_sorted = colorlevel[np.argsort( colorlevel )][::-1]
    points_kn_array = np.array( points_kn_list )
    points_kn_array_sorted  = points_kn_array[np.argsort( colorlevel )][::-1]
    for ii in xrange( count ) :
        points_kn = points_kn_array_sorted[ii]
        plt.plot( points_kn.T[0], points_kn.T[1], color=cm.afmhot( colorlevel_sorted[ii] ) )

    #--------------------------------------------------------------------------
    # data
    plt.plot( U_in.T[0], U_in.T[1], 'k' )
#    plt.plot( U_in.T[0], U_in.T[1], marker='.', c="black", label='data' )

    #--------------------------------------------------------------------------
    # answer
    # projection of 'answer' onto PC plane
    albd_answer_kj  = np.loadtxt( ALBDFILE ).T
    dalbd_answer_kj = albd_answer_kj - M_j
    coeff_kn = np.dot( dalbd_answer_kj, V_nj.T )
    answer_x, answer_y = coeff_kn[:,0:2].T

    plt.scatter( answer_x[0], answer_y[0], marker='o', c='blue' )
    plt.scatter( answer_x[1], answer_y[1], marker='s', c='red' )
    plt.scatter( answer_x[2], answer_y[2], marker='^', c='green' )


    dummy_x = np.zeros( len( colorlevel ) ) + 100.
    dummy_y = np.zeros( len( colorlevel ) ) + 100.
    SC = plt.scatter( dummy_x, dummy_y, c=colorterm, cmap=cm.afmhot )
    divider = make_axes_locatable(ax1)
    ax_cb = divider.new_horizontal(size="2%", pad=0.05)
    fig.add_axes(ax_cb)
    plt.colorbar(SC, cax=ax_cb)

    #--------------------------------------------------------------------------
    # save
    filename = OUTFILE_DIR+INFILE+'_fitmap_ratio_regf_PCplane_l' + str(LAMBDA_CORR_DEG[0]) + 'deg_ratio2_' + str(SEED_in) + '.pdf'
    plt.savefig( filename, bbox_inches='tight' )


    #--------------------------------------------------------------------------
    # best

    points_kn_best = points_kn_array[ np.argmin( colorterm ) ][:-1,:]
    
    print 'points_kn_best', points_kn_best
    X_albd_kj_best = np.dot( points_kn_best, V_nj ) + M_j

    points_kq = np.c_[ points_kn_best, np.ones( len( points_kn_best ) ) ]
    X_area_lk_best  = np.dot( U_iq, np.linalg.inv( points_kq ) )

    np.savetxt( OUTFILE_DIR+INFILE+'_l' + str(LAMBDA_CORR_DEG[0]) + 'deg_fitmap_ratio_' + str(SEED_in) + '_X_albd_jk_best', X_albd_kj_best.T )
    np.savetxt( OUTFILE_DIR+INFILE+'_l' + str(LAMBDA_CORR_DEG[0]) + 'deg_fitmap_ratio_' + str(SEED_in) + '_X_area_lk_best', X_area_lk_best )


