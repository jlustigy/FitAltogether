import numpy as np

#--------------------------------------------------------------------
# Parameters
#--------------------------------------------------------------------

NUM_MCMC = 100
NUM_MCMC_BURNIN = 0
SEED_AMP = 0.5

REGULARIZATION = None
#REGULARIZATION = 'GP'
#REGULARIZATION = 'GP2'
#REGULARIZATION = 'Tikhonov'

SIGMA_Y  = 3.0
NOISELEVEL = 0.01

FLAG_REG_AREA = False
FLAG_REG_ALBD = False

#n_slice = 4
N_TYPE  = 3

deg2rad = np.pi/180.

N_SIDE   = 32
#INFILE = "data/raddata_12_norm"
INFILE = "data/raddata_2_norm"
#INFILE = "mockdata/mock_simple_1_data"
#INFILE = "mockdata/mock_simple_3types_1_data"
# INFILE = 'mockdata/mock_simple_1_scattered0.01_data_with_noise'

HDF5_COMPRESSION = 'lzf'

def calculate_walkers(n_dim):
    return 10*n_dim
