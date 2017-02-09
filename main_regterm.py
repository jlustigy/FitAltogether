import numpy as np
import healpy as hp
import sys

import matplotlib.pyplot as plt

import prior
import reparameterize

deg2rad = np.pi/180.

# AREAFILE='mockdata/mockdata_135deg_3types_t12_factor'
AREAFILE='mockdata/mockdata_90deg_3types_t12_factor'
# AREAFILE='mockdata/mockdata_45deg_3types_t12_factor'

# March 2008
#LAT_S = -0.5857506  # sub-solar latitude
#LON_S = 267.6066184  # sub-solar longitude
#LAT_O = 1.6808370  # sub-observer longitude
#LON_O = 210.1242232 # sub-observer longitude

N_TICKS = 101
PLOT_ON = False

#===================================================
# basic functions
#=============================================== ====

def regularize_area_GP( x_area_lk, lambda_angular, type='squared-exponential' ):

    l_dim = len( x_area_lk )
    cov = prior.get_cov( 1.0, 0.0, lambda_angular, l_dim, type=type, periodic=False )

#    inv_cov = np.linalg.inv( cov )
    inv_cov = np.linalg.solve( cov, np.identity( len( cov ) ) )
    det_cov = np.linalg.det( cov )

#    cov += np.identity( len( cov ) )

    la, v = np.linalg.eig( cov )

#    print ''
#    print 'cov', cov
    print ''
    print 'la', la
#    print 'det_cov', det_cov
    if ( det_cov == 0. ):
        print 'det_cov', det_cov
        print 'cov', cov

#    print '----------------------------------------'
#    print np.dot( inv_cov, cov )
#    print '----------------------------------------'

#    print 'inv_cov', inv_cov
    x_area_ave = np.average( x_area_lk, axis=0 )
    x_area_std = np.std( x_area_lk, axis=0 )

#    dx_area_lk = x_area_lk[:,:-1] - x_area_ave
    dx_area_lk = ( x_area_lk - x_area_ave ) / x_area_std
#    print 'dx_area_lk', dx_area_lk
    term1_all = np.dot( dx_area_lk.T, np.dot( inv_cov, dx_area_lk ) )
#    term1_all = np.dot( x_area_lk.T, np.dot( inv_cov, x_area_lk ) )
    terms = 0.5 * term1_all.diagonal()

    term2 = 0.5 * np.log( det_cov )

    return cov, inv_cov, terms,  term2


#===================================================
if __name__ == "__main__":


    # Determine initial values for fitting parameters
    X0_area_lk = np.loadtxt( AREAFILE )

    l_ticks = np.arange( len( X0_area_lk ) )
    l_mesh, m_mesh = np.meshgrid( l_ticks, l_ticks )
    m_mesh = -1 * m_mesh
#    v = np.linspace(0., 1.0, 10, endpoint=True)

    lambda_ticks = np.linspace( 0., np.pi, N_TICKS )[1:]

    with open( AREAFILE+'_regterm_ratio', 'w' ) as f:

        for lambda_angular in lambda_ticks :

            cov, inv_cov, terms, term2 = regularize_area_GP( X0_area_lk, lambda_angular, type='squared-exponential' )

            if PLOT_ON :

                SC = plt.imshow( cov, interpolation='none' )
                plt.colorbar(SC)
                plt.axes().set_aspect( 'equal', 'datalim' )
                plt.clim(0,1)
                plt.savefig( AREAFILE+'_'+str(lambda_angular/deg2rad)+'_noperiodic_cov.png' )
                plt.clf()
                
                SC = plt.imshow( inv_cov, interpolation='none' )
                plt.colorbar(SC)
                plt.axes().set_aspect( 'equal', 'datalim' )
                plt.savefig( AREAFILE+'_'+str(lambda_angular/deg2rad)+'_noperiodic_invcov.png' )
                plt.clf()
                

            f.write( str(lambda_angular/deg2rad) + '\t' )
            for ii in xrange( len( terms ) ):
                f.write( str( terms[ii] ) + '\t' )
            f.write( str(term2) + '\n' )


