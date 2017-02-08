import numpy as np
import healpy as hp
import sys

import matplotlib.pyplot as plt

import prior
import reparameterize

deg2rad = np.pi/180.

N_TICKS = 11
PLOT_ON = False

#===================================================
# basic functions
#=============================================== ====

def Tikhonov( cov, alpha=1e-4 ):

    Gamma = alpha * np.identity( len( cov ) )
    coeff = np.dot( np.linalg.inv( np.dot( cov.T, cov ) + np.dot( Gamma.T, Gamma ) ), cov.T )
    return np.dot( coeff, np.identity( len( cov ) ) )


#===================================================
if __name__ == "__main__":

    lambda_ticks = np.linspace( 0., np.pi, N_TICKS )[1:]

    for lambda_angular in lambda_ticks :

        l_dim   = 7
        cov     = prior.get_cov( 1.0, 0.0, lambda_angular, l_dim, periodic=False )
#        inv_cov = np.linalg.solve( cov, np.identity( len( cov ) ) )
        inv_cov = Tikhonov( cov )
        det_cov = np.linalg.det( cov )
        
        SC = plt.imshow( cov, interpolation='none' )
        plt.colorbar(SC)
        plt.axes().set_aspect( 'equal', 'datalim' )
        plt.clim(0,1)
        plt.savefig( 'cov/tik_cov_L'+str(l_dim)+'_'+str(lambda_angular/deg2rad)+'_noperiodic.png' )
        plt.clf()
        
        SC = plt.imshow( inv_cov, interpolation='none' )
        plt.colorbar(SC)
        plt.axes().set_aspect( 'equal', 'datalim' )
        plt.savefig( 'cov/tik_cov_L'+str(l_dim)+'_'+str(lambda_angular/deg2rad)+'_noperiodic_inv.png' )
        plt.clf()
        
        print np.dot( cov, inv_cov )

