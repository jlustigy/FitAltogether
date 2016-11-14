import numpy as np
import math
import sys
from scipy.optimize import minimize

ALPHA_INI = 1.e-4
DIFF_LIM  = 1e-1



#---------------------------------------------------
def init_A_nm( n_PC ) :

    # N ( = n_PC ): number of principle components
    # M ( = n_PC + 1 ) : number of vertices
    # dimension of A_nm     : N x M
    # dimension of A_mn_tmp : M x N just for personal convenience

    A_mn_tmp = np.array( [-1., 1.] ) # each side has length '2'
    for nn in xrange( 1, n_PC ) :

        # new dimension for the previous point(s)
        A_mn_tmp = np.c_[ A_mn_tmp,  [0.]*len( A_mn_tmp ) ]
        # the new dimension for the new point
        distance = np.sqrt( 2.**2 - np.sum( A_mn_tmp[0]**2 ) )
        A_mn_tmp = np.vstack( [ A_mn_tmp, [0.]*( len( A_mn_tmp ) -1 )+[ distance ]  ] )
        # balance point
        balance_n = np.sum( A_mn_tmp, axis=0 ) / len( A_mn_tmp )
        A_mn_tmp  = A_mn_tmp - balance_n

    return A_mn_tmp.T


#---------------------------------------------------
def get_volume ( A_nm, n_PC ) :

    AA_nn = np.delete( A_nm, (0), axis=1 )
    dAA_nn = AA_nn - np.tile( A_nm.T[0] , [ len( AA_nn ), 1 ] ).T
    vol = 1. / math.factorial( n_PC ) * np.absolute( np.linalg.det( dAA_nn ) )
    return vol


#---------------------------------------------------
def get_penalty ( UU_qi, AA_qm ) :

    P_mi = np.dot( np.linalg.inv( AA_qm ) , UU_qi )
    if ( np.all( P_mi > 0. ) and np.all( P_mi < 1. ) ) :
        pnlty = np.sum( 1. / P_mi )
    else :
        pnlty = 1e10
    return pnlty


#---------------------------------------------------
def func_to_minimize ( A_nm_flat, *args ) :

    UU_qi, n_PC, alpha = args
    A_nm = A_nm_flat.reshape( [ n_PC , n_PC+1 ] )

    A_mn = A_nm.T
    AA_qm = np.c_[ A_nm.T, np.ones( n_PC + 1 ) ].T

    # volume to minimize
    volume = get_volume( A_nm, n_PC )

    # penalty to enclose all samples
    penalty = get_penalty( UU_qi, AA_qm )

    value = volume + alpha * penalty 
    return value


#---------------------------------------------------
def do_shrinkwrap ( U_in, n_PC, output=True, run_dir='' ) :

    # N ( = n_PC ): number of principle components
    # M ( = n_PC + 1 ) : number of vertices

    # initialization of data samples
    # UU_mi =  A_nm x P_ni
    UU_qi = np.c_[ U_in, np.ones( len( U_in ) ) ].T

    # initialization of vertices
    # dimension of A_nm : N x M 
    A_nm_ini = init_A_nm( n_PC ) # evenly distributed initial locations

    A_nm_ini_flat = A_nm_ini.flatten()

    alpha = ALPHA_INI * 10.
    vol_old = get_volume ( A_nm_ini, n_PC )

    diff = 1.

    if output :

        with open ( run_dir+'shrinkwrap_log', 'w' ) as f :

            while ( diff > DIFF_LIM ) : 
                
                alpha = alpha / 10.
                f.write( '# ------------------------------------\n' )
                f.write( '# best fit with alpha='+str(alpha)+'\n' )
                params = ( UU_qi , n_PC, alpha )
                A_nm_bestfit_flat = minimize( func_to_minimize, A_nm_ini_flat, args=params, method='L-BFGS-B' )['x']
                A_nm_bestfit      = A_nm_bestfit_flat.reshape( [ n_PC , n_PC+1 ] )

                for mm in xrange( len( A_nm_bestfit.T ) ):
                    for nn in xrange( len( A_nm_bestfit ) ):
                        f.write( str( A_nm_bestfit[nn][mm] )+'\t' ) 
                    f.write( '\n' )
                for nn in xrange( len( A_nm_bestfit ) ):
                    f.write( str( A_nm_bestfit[nn][0] )+'\t' ) 
                f.write( '\n' )
                vol  = get_volume ( A_nm_bestfit, n_PC )
                diff = ( vol_old - vol ) / vol_old # fractional volume change
                f.write( '# vol_old, vol, diff\t' + str(vol_old) + '\t' + str(vol) + '\t' + str(diff) + '\n\n\n' )
                vol_old = vol
                A_nm_ini_flat = A_nm_bestfit_flat
            f.write( '#------------------------------------\n' )

        np.savetxt( run_dir+'shrinkwrap_final', A_nm_bestfit.T )

    else :

        while ( diff > DIFF_LIM ) : 

            alpha = alpha / 10.
            f.write( '# ------------------------------------\n' )
            f.write( '# best fit with alpha='+str(alpha)+'\n' )
            params = ( UU_qi , n_PC, alpha )
            A_nm_bestfit_flat = minimize( func_to_minimize, A_nm_ini_flat, args=params, method='L-BFGS-B' )['x']
            A_nm_bestfit      = A_nm_bestfit_flat.reshape( [ n_PC , n_PC+1 ] )

            for mm in xrange( len( A_nm_bestfit.T ) ):
                for nn in xrange( len( A_nm_bestfit ) ):
                    f.write( str( A_nm_bestfit[nn][mm] )+'\t' ) 
                f.write( '\n' )
            for nn in xrange( len( A_nm_bestfit ) ):
                f.write( str( A_nm_bestfit[nn][0] )+'\t' ) 
            f.write( '\n' )
            vol  = get_volume ( A_nm_bestfit, n_PC )
            diff = ( vol_old - vol ) / vol_old # fractional volume change
            f.write( '# vol_old, vol, diff\t' + str(vol_old) + '\t' + str(vol) + '\t' + str(diff) + '\n\n\n' )
            vol_old = vol
            A_nm_ini_flat = A_nm_bestfit_flat

        f.write( '#------------------------------------\n' )


    AA_qm_bestfit = np.c_[ A_nm_bestfit.T, np.ones( n_PC + 1 ) ].T
    P_mi_bestfit  = np.dot( np.linalg.inv( AA_qm_bestfit ) , UU_qi )

    return A_nm_bestfit.T, P_mi_bestfit.T

