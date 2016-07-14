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

#-----------------------------------------------------------------------------

N_SLICE = 180

# ocean
#target_indx = [17]

# vegetation
# target_indx = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14]
 #target_indx = [1, 2, 3, 4, 5, 6, 10, 14]

# soil
# target_indx = [7,11,13,16]
target_indx = [7,8,9,11,12,13,16]


# all
#target_indx = range(1,18)
#target_indx = range(20)


#--------------------------------------------------------------------
# set-up
#--------------------------------------------------------------------

NCFILE =  'data/SURFACE_TYPE_IGBP_10min.cdf'

#--------------------------------------------------------------------
# read IGBP file
#--------------------------------------------------------------------

ncfile_r = netCDF4.Dataset( NCFILE, 'r', format='NETCDF3_64BIT' )

lat  = ncfile_r.variables['lat'][:] # omega
lon  = ncfile_r.variables['lon'][:]  # omega

surface_type = ncfile_r.variables['surface_type'][:,:] # omega

lon = lon + 180.
#for ll in xrange( len( lon ) ):
#    print ll, np.trunc( lon[ll] ), np.trunc( lon[ll] ).astype(np.int64), lon[ll]

lat_mesh, lon_mesh = np.meshgrid( lat, lon, indexing='ij' )

theta_d = ( 90. - lat_mesh.flatten() ) * np.pi / 180.
phi_d = lon_mesh.flatten()
type_d = surface_type.flatten()

# assignedL_float_d = np.floor( phi_d/( 2.*np.pi / N_SLICE))
assignedL_float_d = phi_d/( 360. / N_SLICE)
assignedL_d = assignedL_float_d.astype(np.int64)
LN_dl = np.zeros([ len( theta_d ), N_SLICE ]) 
LN_dl[ np.arange( len( theta_d ) ), assignedL_d ] = 1.

#for ll in xrange( len( LN_dl.T ) ):
#    print ll, np.count_nonzero( LN_dl.T[ll] )


rad_10min = np.pi / 180. * ( 1. / 6. )
area_d = np.sin( theta_d ) * rad_10min * rad_10min
indx_of_interest = np.in1d( type_d, target_indx )
indx_of_nointerest = map( operator.not_, indx_of_interest )
area_d[np.where( indx_of_nointerest )] = 0.

area_l = np.dot( area_d, LN_dl ) / ( 4. * np.pi / N_SLICE )

for ll in xrange( N_SLICE ) :
    print -180.+360.*(ll+0.5)/N_SLICE, area_l[ll]


