import numpy as np

#--------------------------------------------------------------------
# Parameters
#--------------------------------------------------------------------

#REGULARIZATION = None
REGULARIZATION = 'GP'
#REGULARIZATION = 'GP2'
#REGULARIZATION = 'Tikhonov'

N_TYPE  = 3
N_SLICE = 9

MONTH = 'March'
WAVEBAND_CENTERS = np.array([350., 450., 550., 650., 750., 850., 950.])
WAVEBAND_WIDTHS = np.array([100., 100., 100., 100., 100., 100., 100.])

NOISELEVEL = 0.01

NUM_MCMC = 20000
NUM_MCMC_BURNIN = 0
SEED_AMP = 0.5

N_side_seed = 5
N_SIDE  = 2*2**N_side_seed

Pspin = 24.0
OMEGA = ( 2. * np.pi / Pspin )

HDF5_COMPRESSION = 'lzf'

def calculate_walkers(n_dim):
    return 10*n_dim#**2
