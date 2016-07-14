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

# MONTH = 'March'
MONTH = 'June'

N_SLICE = 24
# soil
# target_indx = [7,11,13,16]

# ocean
#target_indx = [17]

# vegetation
# target_indx = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14]

# all
target_indx = range(1,18)


cldoptd_crit = 5.

TIME_END = 24.0
TSTEP = 1.0
Pspin = 24.0
OMEGA = ( 2. * np.pi / Pspin )

# HealPix
N_side_seed = 3
N_SIDE  = 2*2**N_side_seed

N_pixel = 12*N_SIDE**2

print '# N_pixel', N_pixel


#--------------------------------------------------------------------
# set-up
#--------------------------------------------------------------------

if ( MONTH == 'March' ):
# based on spectroscopic data
#         Sub-Sun Lon/Lat =      97.091       -0.581 /     W longitude, degrees 
#         Sub-SC  Lon/Lat =     154.577        1.678 /     W longitude, degrees
    LAT_S = -0.581  # sub-solar latitude
    LON_S = 262.909  # sub-solar longitude
    LAT_O = 1.678  # sub-observer latitude
    LON_O = 205.423 # sub-observer longitude
    INFILE = "data/raddata_1_norm"
    Time_i = np.arange(25)*1.
    cldfile_frac = "data/cldfrac_EPOXI_March_6.dat"
    cldfile_optd = "data/clddpth_EPOXI_March_6.dat"
elif ( MONTH == 'June' ):
# based on spectroscopic data
#         Sub-Sun Lon/Lat =      79.023       22.531 /     W longitude, degrees
#         Sub-SC  Lon/Lat =     154.535        0.264 /     W longitude, degrees
    LON_S = 280.977
    LAT_S = 22.531
    LON_O = 205.465
    LAT_O = 0.264
#    LON_O = 165.4663412
#    LAT_O = -0.3521857
#    LON_S = 239.1424068
#    LAT_S = 21.6159766
    cldfile_frac = "data/cldfrac_EPOXI_June_6.dat"
    cldfile_optd = "data/clddpth_EPOXI_June_6.dat"
    INFILE = "data/raddata_2_norm"
    Time_i = np.arange(25)*1.

else :
    print 'ERROR: Invalid MONTH'
    sys.exit()


NCFILE =  'data/SURFACE_TYPE_IGBP_10min.cdf'

#--------------------------------------------------------------------
# read cloud file
#--------------------------------------------------------------------
lat, lon, cld_frac = np.loadtxt( cldfile_frac ).T
lat, lon, cld_optd = np.loadtxt( cldfile_optd ).T

lon[ np.where( lon < 0. )] = 360. + lon[ np.where( lon < 0. )]

theta_data = ( 90. - lat ) * np.pi / 180.
phi_data = ( lon ) * np.pi / 180.
pixel_data = hp.pixelfunc.ang2pix( N_SIDE, theta_data, phi_data )

#cldoptd_data = cld_optd
#cldfrac_data =

cld_frac_data = np.zeros( N_pixel )
cld_optd_data = np.zeros( N_pixel )
for ipix in xrange( N_pixel ):

    # cloud optical thickness
    cld_optd_tmp = cld_optd[np.where( pixel_data == ipix )]
    histo_data = np.histogram( cld_optd_tmp, bins=np.logspace(-1,2,31) )[0]
    cld_optd_data[ipix] = np.argmax( histo_data )

    # cloud optical thickness
    cld_frac_tmp = cld_frac[np.where( pixel_data == ipix )]
    histo_data = np.histogram( cld_frac_tmp, bins=np.arange(0,101) )[0]
    cld_frac_data[ipix] = np.argmax( histo_data )

cld_frac_data = cld_frac_data * 0.01


cld_frac_data[ np.where( cld_optd_tmp <  cldoptd_crit ) ] = 0.



#--------------------------------------------------------------------
# read IGBP file
#--------------------------------------------------------------------

ncfile_r = netCDF4.Dataset( NCFILE, 'r', format='NETCDF3_64BIT' )

lat  = ncfile_r.variables['lat'][:] # omega
lon  = ncfile_r.variables['lon'][:]  # omega
surface_type = ncfile_r.variables['surface_type'][:,:] # omega

lon[ np.where( lon < 0. )] = 360. + lon[ np.where( lon < 0. )]
lat_mesh, lon_mesh = np.meshgrid( lat, lon, indexing='ij' )

theta_data = ( 90. - lat_mesh.flatten() ) * np.pi / 180.
phi_data = ( lon_mesh.flatten() ) * np.pi / 180.

surface_data = surface_type.flatten()
pixel_data = hp.pixelfunc.ang2pix( N_SIDE, theta_data, phi_data )

type_n = np.zeros( N_pixel, dtype=int )
for ipix in xrange( N_pixel ):
    type_tmp_data = surface_data[np.where( pixel_data == ipix )]
    histo_data = np.histogram( type_tmp_data, bins=np.arange(19)-0.5 )[0]
    type_n[ipix] = np.argmax( histo_data )


#--------------------------------------------------------------------
# loop in time
#--------------------------------------------------------------------


theta_n, phi_n   = hp.pixelfunc.pix2ang( N_SIDE, np.arange( N_pixel ) )
phi_n[ phi_n < np.pi ]  = phi_n[ phi_n < np.pi ] + 2. * np.pi 
assignedL_float_n = np.floor(( phi_n - np.pi)/( 2.*np.pi / N_SLICE ))
assignedL_n = assignedL_float_n.astype(np.int64)
LN_nl = np.zeros([ N_pixel, N_SLICE ]) 
LN_nl[ np.arange( N_pixel ), assignedL_n ] = 1.


area_n = np.ones( N_pixel ) / ( N_pixel*1. )

indx_of_interest = np.in1d( type_n, target_indx )
indx_of_nointerest = map( operator.not_, indx_of_interest )
area_n[np.where( indx_of_nointerest )] = 0.

area_l = np.dot( area_n, LN_nl ) * N_SLICE

for ll in xrange( N_SLICE ) :
    print -180.+360.*(ll+0.5)/N_SLICE, area_l[ll]


