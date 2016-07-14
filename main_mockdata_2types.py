import numpy as np
import healpy as hp
from scipy.optimize import minimize
import scatter
import geometry

N_SLICE = 13
N_BAND  = 3
N_TYPE  = 2

NOISELEVEL = 0.01

deg2rad = np.pi/180.

TSTEP = 1.0

OUTFILE = "mockdata/mock_simple_JuneKernel_scattered0.01"

N_side_seed = 3
N_SIDE  = 2*2**N_side_seed

# geometry of June
LON_S = 280.977
LAT_S = 22.531
LON_O = 205.465
LAT_O = 0.264
Pspin = 24.
OMEGA = ( 2. * np.pi / Pspin )
Time_i = np.arange(25)*1.

#===================================================
if __name__ == "__main__":

# mock simple data 1
    X_albd_kj = np.zeros([N_TYPE, N_BAND])
#    X_albd_kj[0,:] = 0.1+0.2*np.arange(N_BAND)
#    X_albd_kj[1,:] = 0.9-0.3*np.arange(N_BAND)
    X_albd_kj[0,:] = [0.1, 0.3, 0.35]
    X_albd_kj[1,:] = [0.9, 0.5, 0.6]

    X_area_lk = np.zeros([N_SLICE, N_TYPE])
    X_area_lk[:,0] = [0.2, 0.55, 0.6, 0.9, 0.95, 0.8, 0.7, 0.3, 0.4, 0.5, 0.56, 0.43, 0.32]
    X_area_lk[:,1] = 1.0 - X_area_lk[:,0]


    param_geometry = ( LAT_O, LON_O, LAT_S, LON_S, OMEGA )
    Kernel_il = geometry.kernel( Time_i, N_SLICE, N_SIDE, param_geometry )
#    Model_lj = np.zeros([N_TIME, N_BAND])

    # making matrix...
    Model_ij = np.dot( Kernel_il, np.dot(X_area_lk, X_albd_kj))
    with open(OUTFILE+"_data", 'wb') as f:
        f.write("# data vector ( longitude slice x band )\n")
        np.savetxt( f, Model_ij )

    Model_ij_noise = scatter.add_gaussnoise( Model_ij, NOISELEVEL )
    with open(OUTFILE+"_data_with_noise", 'wb') as f:
        f.write("# data vector ( longitude slice x band )\n")
        np.savetxt( f, Model_ij_noise )


    with open(OUTFILE+"_albd_area", 'wb') as f:

        f.write("\n\n")
        f.write("# albedo ( band x surface type )\n")
        np.savetxt(f, X_albd_kj.T)

        f.write("\n\n")
        f.write("# area fraction ( longitude slice x suface type )\n")
        np.savetxt(f, X_area_lk)
