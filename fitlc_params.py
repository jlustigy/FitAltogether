import numpy as np

#--------------------------------------------------------------------
# Parameters
#--------------------------------------------------------------------

<<<<<<< HEAD
NUM_MCMC = 20000
=======
NUM_MCMC = 5000
>>>>>>> upstream/master
NUM_MCMC_BURNIN = 0
SEED_AMP = 0.1

SIGMA_Y  = 100.0
NOISELEVEL = 0.01

#REGULARIZATION = 'GP'
#REGULARIZATION = 'GP2'
REGULARIZATION = 'GP4'
#REGULARIZATION = 'Tikhonov'
#REGULARIZATION = None

SLICE_TYPE = 'time'
# SLICE_TYPE = 'longitude'
N_SLICE_LONGITUDE = 5

# lat_o, lon_o, lat_s, lon_s, omega
GEOM = ( 0., 0., 0.,  90., 2.*np.pi )

deg2rad = np.pi/180.

N_SIDE   = 32
# INFILE = '/Users/yuka/Dropbox/Project/11_RotationalUnmixing/mockdata_quadrature_lc'
# INFILE = 'data/raddata_1_norm'
# INFILE = 'mockdata/simpleIGBP_quadrature_lc'
INFILE = 'mockdata_90deg_13time_lcscat'
# INFILE = 'mockdata_90deg_lc'

KNOWN_ANSWER=False
ALBDFILE = 'mockdata/simpleIGBP_quadrature_bandsp'
AREAFILE = 'mockdata/simpleIGBP_quadrature_factor'
#INFILE = "data/raddata_12_norm"
<<<<<<< HEAD
##INFILE = "data/raddata_2_norm"
INFILE = "data/raddata_1_norm"
#INFILE = "mockdata/mock_simple_1_data"
#INFILE = "mockdata/mock_simple_3types_1_data"
# INFILE = 'mockdata/mock_simple_1_scattered0.01_data_with_noise'
#INFILE = 'mockdata/simpleIGBP_quadrature_lc'

#WAVEBAND_CENTERS = np.array([550., 650., 750., 850.])
#WAVEBAND_WIDTHS = np.array([100., 100., 100., 100.])
WAVEBAND_CENTERS = np.array([350., 450., 550., 650., 750., 850., 950.])
WAVEBAND_WIDTHS = np.array([100., 100., 100., 100., 100., 100., 100.])

HDF5_COMPRESSION = 'lzf'
=======
#INFILE = "data/raddata_2_norm"
#INFILE = "data/raddata_2_norm"
#INFILE = "mockdata/mock_simple_1_data"
#INFILE = "mockdata/mock_simple_3types_1_data"
# INFILE = 'mockdata/mock_simple_1_scattered0.01_data_with_noise'
>>>>>>> upstream/master

def calculate_walkers(n_dim):
    return 2*n_dim**2
 
