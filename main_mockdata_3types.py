import numpy as np
import healpy as hp
from scipy.optimize import minimize

N_SLICE = 6
N_BAND  = 5
N_TYPE  = 3

deg2rad = np.pi/180.

TSTEP = 6.0

OUTFILE = "mockdata/mock_simple_3types_1"



#===================================================
if __name__ == "__main__":

 
# mock simple data 1
    X_albd_kj = np.zeros([N_TYPE, N_BAND])
#    X_albd_kj[0,:] = 0.1+0.2*np.arange(N_BAND)
#    X_albd_kj[1,:] = 0.9-0.3*np.arange(N_BAND)
    X_albd_kj[0,:] = [0.1, 0.3, 0.35, 0.4, 0.45]
    X_albd_kj[1,:] = [0.9, 0.5, 0.4, 0.5, 0.6]
    X_albd_kj[2,:] = [0.8, 0.75, 0.7, 0.5, 0.3]

    X_area_lk = np.zeros([N_SLICE, N_TYPE])
    X_area_lk[:,0] = [0.2, 0.55, 0.3, 0.4, 0.8, 0.1]
    X_area_lk[:,1] = [0.3, 0.15, 0.3, 0.2, 0.1, 0.7]
    X_area_lk[:,2] = 1.0 - np.sum(X_area_lk, axis=1)

    # making matrix...
    Model_lj = np.dot(X_area_lk, X_albd_kj)

    with open(OUTFILE+"_data", 'wb') as f:

        f.write("# data vector ( longitude slice x band )\n")
        np.savetxt(f, Model_lj)


    with open(OUTFILE+"_albd_area", 'wb') as f:

        f.write("\n\n")
        f.write("# albedo ( band x surface type )\n")
        np.savetxt(f, X_albd_kj.T)

        f.write("\n\n")
        f.write("# area fraction ( longitude slice x suface type )\n")
        np.savetxt(f, X_area_lk)
