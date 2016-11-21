import numpy as np

#--------------------------------------------------------------------
# Parameters
#--------------------------------------------------------------------

NUM_MCMC = 5000
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
#INFILE = "data/raddata_2_norm"
#INFILE = "data/raddata_2_norm"
#INFILE = "mockdata/mock_simple_1_data"
#INFILE = "mockdata/mock_simple_3types_1_data"
# INFILE = 'mockdata/mock_simple_1_scattered0.01_data_with_noise'

def calculate_walkers(n_dim):
    return 2*n_dim**2
 
