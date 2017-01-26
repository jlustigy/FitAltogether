import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import font_manager
from copy import deepcopy
import sys
import healpy as hp
from scipy import constants
import geometry
import operator


N_LON = 360.
N_ANGLE = 5

P_ROT = 100.
OMEGA = (2.*np.pi/P_ROT)
T_MAX = 1000.
T_NUM = 1001

deg2rad = (np.pi/180.)



#-----------------------------------------------------------------------------
def map_1slice( grid_phi, width ) :
    """
    1 slice map
    """
    grid_albd = np.zeros_like( grid_phi )

    grid_albd[ np.where( ( 0. < grid_phi ) * ( grid_phi < width ) ) ] = 1.

    return grid_albd


#-----------------------------------------------------------------------------
def make_mockdata( grid_t, width, alpha ):

    grid_phi  = np.linspace( 0, 2.*np.pi, N_LON )

    mesh_t, mesh_phi = np.meshgrid( grid_t, grid_phi, indexing='ij' )

    mesh_albd = map_1slice( mesh_phi, width )

    mesh_weight1  = np.maximum( 0., np.cos( mesh_phi + OMEGA*mesh_t ) )
    mesh_weight2  = np.maximum( 0., np.cos( mesh_phi + OMEGA*mesh_t - alpha ) )
    mesh_integral = mesh_weight1 * mesh_weight2 * mesh_albd
    mesh_integral_norm = mesh_weight1 * mesh_weight2
    grid_lc   = np.trapz( mesh_integral, x=grid_phi, axis=1 )
    grid_norm = np.trapz( mesh_integral_norm, x=grid_phi, axis=1 )

    
    return grid_lc / grid_norm


#-----------------------------------------------------------------------------
if __name__ == "__main__" :

    ticks_alpha = np.linspace( 45, 135,       3 )*deg2rad
    ticks_width = np.linspace( 30, 150, N_ANGLE )*deg2rad

    grid_t = np.linspace( 0, T_MAX, T_NUM )[:-1] # 10 rotations
    dt = T_MAX / ( T_NUM - 1 )


    with open( '1slicemap_lc.txt', 'w' ) as f_lc :

        with open( '1slicemap_corr.txt', 'w' ) as f_corr :

            for alpha in ticks_alpha :

                for width in ticks_width :


                    
                    grid_lc = make_mockdata( grid_t, width, alpha )

                    f_lc.write( '# alpha, width = '+ str(alpha/deg2rad) + str(width/deg2rad) )
                    for ii in xrange( len( grid_t ) ):
                        f_lc.write( str( grid_t[ii] ) + '\t' + str( grid_lc[ii] ) + '\n' )
                    f_lc.write( '\n\n' )

#                    grid_lc = grid_lc - np.average( grid_lc )

                    corrcoeff_i = np.zeros( T_NUM )
                    f_corr.write( '# alpha, width = '+ str(alpha/deg2rad) + str(width/deg2rad) )
                    for ii in xrange( len( grid_lc ) ):
                        corrcoeff_i[ii] = np.dot( grid_lc[:], np.roll( grid_lc, ii ) ) / np.dot( grid_lc, grid_lc )
                        f_corr.write( str( ii*dt/P_ROT*360. ) + '\t' + str( corrcoeff_i[ii] ) + '\n' )
                    f_corr.write( '\n\n' )
