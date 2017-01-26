import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import font_manager
from copy import deepcopy
import sys
import healpy as hp
from scipy import constants
import geometry_forward
import operator
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap


deg2rad = np.pi/180.

#-----------------------------------------------------------------------------
# Specifying the Classification
#-----------------------------------------------------------------------------

# 3 surface type
target_indx_list = [[ 17 ], # ocean
                   [ 7, 11, 13, 15, 16 ], # soil
                   [ 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14 ]] # vegetation

# 2 surface types
# target_indx_list = [[ 17 ], # ocean
#                   [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ]] # continents

#-----------------------------------------------------------------------------

TAG='135deg_3types_redAmerica_t12'

#--------------------------------------------------------------------
# set-up
#--------------------------------------------------------------------

LON_S = 135.
LAT_S = 0.
LON_O = 0.
LAT_O = 0.
Time_i = np.arange(12)*1./12.

NCFILE =  'data/SURFACE_TYPE_IGBP_10min.cdf'

Pspin = 1.0
OMEGA = ( 2. * np.pi / Pspin )

# HealPix
N_side_seed = 3
N_SIDE  = 2*2**N_side_seed

N_pixel = 12*N_SIDE**2

print '# N_pixel', N_pixel


#--------------------------------------------------------------------
# read IGBP file
#--------------------------------------------------------------------

ncfile_r = netCDF4.Dataset( NCFILE, 'r', format='NETCDF3_64BIT' )

lat  = ncfile_r.variables['lat'][:] # omega
lon  = ncfile_r.variables['lon'][:]  # omega
nlat = len( lat )
nlon = len( lon )
lat_mesh, lon_mesh = np.meshgrid( lat, lon, indexing='ij' )

theta_data = ( 90. - lat_mesh.flatten() ) * np.pi / 180.
phi_data = ( lon_mesh.flatten() ) * np.pi / 180.
phi_data[ np.where( phi_data < 0. )] = 2.*np.pi + phi_data[ np.where( phi_data < 0. )]

surface_type = ncfile_r.variables['surface_type'][:,:] # omega
surface_type_flat = surface_type.flatten()
surface_data = np.zeros_like( surface_type_flat )
for kk in xrange( len( target_indx_list ) ):
    target_indx = target_indx_list[kk]
    indx_of_interest = np.in1d( surface_type_flat, target_indx )
    surface_data[ indx_of_interest ] = kk

# modified to have RED AMERICA    
surface_data[np.where( ( np.pi < phi_data ) * ( phi_data < (330./360.)*2.*np.pi ) * ( surface_data != 0 ) )] = 1

# save map
with open( 'mockdata/mockdata_'+TAG+'_map', 'w') as f:
    for ii in xrange( 0, len( theta_data ), 1000 ):
        f.write( str(theta_data[ii] )+'\t' )
        f.write( str(phi_data[ii] )+'\t' )
        f.write( str(surface_data[ii])+'\n' )

pixel_data = hp.pixelfunc.ang2pix( N_SIDE, theta_data, phi_data )



# save map
# ax = plt.subplot(111, projection = 'mollweide')
# ax.grid(True)
#print 'lon_mesh.T', lon_mesh.T
#print 'lat_mesh.T', lat_mesh.T

m = Basemap(projection='moll',lon_0=0,resolution='c')
m.contourf( lon_mesh.T, lat_mesh.T, surface_data.reshape([nlat, nlon]).T, 4, cmap=plt.cm.brg,latlon=True)
# ax.contour( lon_mesh.T*deg2rad, lat_mesh.T*deg2rad, surface_data.reshape([nlat, nlon]).T, 10 )
plt.savefig( 'mockdata/mockdata_'+TAG+'_map.png' )

type_data = np.zeros( N_pixel, dtype=int )

for ipix in xrange( N_pixel ):

    type_tmp_data = surface_data[np.where( pixel_data == ipix )]
    histo_data = np.histogram( type_tmp_data, bins=np.arange( len(target_indx_list)+1 )-0.5 )[0]
    type_data[ipix]  = np.argmax( histo_data )

#--------------------------------------------------------------------
# read spectra
#--------------------------------------------------------------------

bands = np.array( [[ 0.4, 0.5 ],
                  [ 0.5, 0.6 ],
                  [ 0.6, 0.7 ],
                  [ 0.7, 0.8 ]] )

def band_ave( wl_array, sp_array ):
    sp_ave = np.zeros( len( bands ) )
    for bb in xrange( len( bands ) ):
        sp_ave[bb] = np.average( sp_array[ np.where( ( bands[bb][0] < wl_array ) * ( bands[bb][1] > wl_array ) ) ] )
    return sp_ave

wl_ocean, sp_ocean = np.loadtxt( 'data/ocean.alb.short', unpack=True )
wl_soil, sp_soil = np.loadtxt( 'data/jhu.becknic.soil.entisol.quartzipsamment.coarse.87P706.spectrum.txt', unpack=True )
wl_soil = wl_soil
sp_soil = sp_soil*0.01
# wl_vege, sp_vege = np.loadtxt( '/Users/yuka/Dropbox/data/ASTER/grass', unpack=True )
wl_vege, sp_vege = np.loadtxt( 'data/jhu.becknic.vegetation.grass.green.solid.gras.spectrum.txt', unpack=True )
wl_vege = wl_vege[::-1]
sp_vege = sp_vege[::-1]*0.01

# 3 surface types
band_sp = np.zeros( [ 3, len(bands) ] )
# 2 surface types
# band_sp = np.zeros( [ 2, len(bands) ] )
band_sp[0] = band_ave( wl_ocean, sp_ocean )
band_sp[1] = band_ave( wl_soil,  sp_soil  )
band_sp[2] = band_ave( wl_vege,  sp_vege  )

np.savetxt( 'mockdata/mockdata_'+TAG+'_band_sp', band_sp.T )

#--------------------------------------------------------------------
# loop in time
#--------------------------------------------------------------------

param_geometry = ( LAT_O, LON_O, LAT_S, LON_S, OMEGA )

sp = np.zeros( [ len(Time_i), len(bands) ] )
contribution_factor = np.zeros( [ len(Time_i),  len( target_indx_list ) ] )

for ii in xrange( len(Time_i) ) :

    time = Time_i[ii]
    weight_array0 = geometry_forward.weight( time, N_SIDE, param_geometry )
    norm = np.sum( weight_array0 )

    for kk in xrange( len( target_indx_list ) ):

        weight_array = deepcopy(weight_array0)
        weight_array[np.where( type_data != kk ) ] = 0.

        areafrac = np.sum( weight_array ) / norm
        contribution_factor[ii][kk] = areafrac

#    print 'contribution_factor', contribution_factor
sp = np.dot( contribution_factor, band_sp )

np.savetxt( 'mockdata/mockdata_'+TAG+'_factor', contribution_factor )
np.savetxt( 'mockdata/mockdata_'+TAG+'_lc', sp )
