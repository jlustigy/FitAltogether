import numpy as np
import healpy as hp
from scipy.optimize import minimize
import scatter

N_SLICE = 4
N_BAND  = 3
N_TYPE  = 2

NOISELEVEL = 0.01

deg2rad = np.pi/180.

TSTEP = 6.0

OUTFILE = "mockdata/mock_simple_1_scattered0.01"



#===================================================
if __name__ == "__main__":

# mock simple data 1
    X_albd_kj = np.zeros([N_TYPE, N_BAND])
#    X_albd_kj[0,:] = 0.1+0.2*np.arange(N_BAND)
#    X_albd_kj[1,:] = 0.9-0.3*np.arange(N_BAND)
    X_albd_kj[0,:] = [0.1, 0.3, 0.35]
    X_albd_kj[1,:] = [0.9, 0.5, 0.6]

    X_area_lk = np.zeros([N_SLICE, N_TYPE])
    X_area_lk[:,0] = [0.2, 0.55, 0.6, 0.9]
    X_area_lk[:,1] = 1.0 - X_area_lk[:,0]

    Time_i   = np.arange(int(24/TSTEP))*TSTEP

#    Model_lj = np.zeros([N_TIME, N_BAND])

    # making matrix...
    Model_lj = np.dot(X_area_lk, X_albd_kj)
    with open(OUTFILE+"_data", 'wb') as f:
        f.write("# data vector ( longitude slice x band )\n")
        np.savetxt(f, Model_lj)

    Model_lj_noise = scatter.add_gaussnoise( Model_lj, NOISELEVEL )
    with open(OUTFILE+"_data_with_noise", 'wb') as f:
        f.write("# data vector ( longitude slice x band )\n")
        np.savetxt(f, Model_lj_noise)


    with open(OUTFILE+"_albd_area", 'wb') as f:

        f.write("\n\n")
        f.write("# albedo ( band x surface type )\n")
        np.savetxt(f, X_albd_kj.T)

        f.write("\n\n")
        f.write("# area fraction ( longitude slice x suface type )\n")
        np.savetxt(f, X_area_lk)
