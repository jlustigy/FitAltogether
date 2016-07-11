import numpy as np
import healpy as hp
from scipy.optimize import minimize

# March 2008
LAT_S = -0.5857506  # sub-solar latitude
LON_S = 267.6066184  # sub-solar longitude
LAT_O = 1.6808370  # sub-observer longitude
LON_O = 210.1242232 # sub-observer longitude

# June 2008
#LON_O = 165.4663412
#LAT_O = -0.3521857
#LON_S = 239.1424068
#LAT_S = 21.6159766

#LAMBDA = 1.0
#KAPPA  = 2.0
OMEGA = (2.0*np.pi/24.0) # [rad/hr]
N_SLICE = 4
N_BAND  = 2

deg2rad = np.pi/180.

N_TYPE  = 2
N_SIDE   = 32

TSTEP = 6.0

OUTFILE = "mockdata/mock7"


#===================================================
# basic functions
#===================================================
def latlon2cart(lat, lon):
    x = np.sin((90.-lat)*deg2rad)*np.cos(lon*deg2rad)
    y = np.sin((90.-lat)*deg2rad)*np.sin(lon*deg2rad)
    z = np.cos((90.-lat)*deg2rad)
    return np.array([x,y,z])


def weight(time):
    """
    Weight of pixel at (lat_r, lon_r) assuming Lambertian surface
    """
    EO_vec = latlon2cart(LAT_O, LON_O-OMEGA*time/deg2rad)
    ES_vec = latlon2cart(LAT_S, LON_S-OMEGA*time/deg2rad)
    ER_vec_array = np.array(hp.pix2vec(N_SIDE, np.arange(hp.nside2npix(N_SIDE))))
    cosTH0_array = np.dot(ES_vec, ER_vec_array)
    cosTH1_array = np.dot(EO_vec, ER_vec_array)
    return np.clip(cosTH0_array, 0., 1.)*np.clip(cosTH1_array, 0., 1.)


def kernel(Time_i, n_slice):
    """
    Kernel!
    """
    Kernel_il = np.zeros([len(Time_i), n_slice])
    N_pix = hp.nside2npix(N_SIDE)
    for ii in xrange(len(Time_i)):
        position_theta, position_phi   = hp.pixelfunc.pix2ang(N_SIDE, np.arange(N_pix))
        for nn in xrange(len(position_phi)):
            if (position_phi[nn] > np.pi):
                position_phi[nn] = position_phi[nn]-2.0*np.pi
        Weight_n = weight(Time_i[ii])
        assigned_l = np.trunc(position_phi/(2.*np.pi/n_slice))
        Count_l = np.zeros(n_slice)
        for nn in xrange(N_pix):
            Kernel_il[ii][assigned_l[nn]] += Weight_n[nn]
            Count_l[assigned_l[nn]]       += 1
        Kernel_il[ii] = Kernel_il[ii]/Count_l
        Kernel_il[ii] = Kernel_il[ii]/np.sum(Kernel_il[ii])
    return Kernel_il




#===================================================
if __name__ == "__main__":


    """
    Misfit-function to be minimized
    """
    X_albd_kj = np.zeros([N_TYPE, N_BAND])

# mock data 2
#    X_albd_kj[0,:] = np.array([0.67, 0.52, 0.40, 0.35, 0.33, 0.33, 0.32])
#    X_albd_kj[1,:] = np.array([0.34, 0.28, 0.28, 0.30, 0.40, 0.48, 0.37])
#    X_albd_kj[2,:] = np.array([0.37, 0.28, 0.15, 0.13, 0.12, 0.10, 0.03])
#    X_area_lk = np.zeros([N_SLICE, N_TYPE])
#    X_area_lk[:,0] = 0.2 + 0.1*np.cos(2.0*np.pi*np.arange(N_SLICE)/4.52)
#    X_area_lk[:,1] = 0.3 + 0.15*np.cos(2.0*np.pi*np.arange(N_SLICE)/7.1)
#    X_area_lk[:,2] = 1.0 - X_area_lk[:,0] - X_area_lk[:,1]


# mock data 3
#    X_albd_kj[,:] = 0.3*np.zeros(N_BAND)
    X_albd_kj[0,:] = 0.1+0.2*np.arange(N_BAND)
    X_albd_kj[1,:] = 0.9-0.3*np.arange(N_BAND)
    X_area_lk = np.zeros([N_SLICE, N_TYPE])
    X_area_lk[:,0] = 0.2 + 0.1*np.cos(2.0*np.pi*np.arange(N_SLICE)/4.52)
#    X_area_lk[:,1] = 0.3 + 0.15*np.cos(2.0*np.pi*np.arange(N_SLICE)/7.1)
#    X_area_lk[:,2] = 1.0 - X_area_lk[:,0] - X_area_lk[:,1]
    X_area_lk[:,1] = 1.0 - X_area_lk[:,0]

    Time_i   = np.arange(int(24/TSTEP))*TSTEP

    Kernel_il = kernel(Time_i, N_SLICE)


#    Model_ij = np.zeros([N_TIME, N_BAND])

    # making matrix...
    Model_ij = np.dot(Kernel_il, np.dot(X_area_lk, X_albd_kj))


    lcdata = np.c_[Time_i, Model_ij]

    with open(OUTFILE+"_lc", 'wb') as f:

        f.write("# LAT_S ="+str(LAT_S)+"\n")
        f.write("# LON_S ="+str(LON_S)+"\n")
        f.write("# LAT_O ="+str(LAT_O)+"\n")
        f.write("# LON_O ="+str(LON_O)+"\n")
        f.write("\n")
        f.write("# apparent albedo ( time x band )\n")
        np.savetxt(f, lcdata)


    with open(OUTFILE+"_albd_area", 'wb') as f:

        f.write("\n\n")
        f.write("# albedo ( band x surface type )\n")
        np.savetxt(f, X_albd_kj.T)

        f.write("\n\n")
        f.write("# area fraction ( longitude slice x suface type )\n")
        np.savetxt(f, X_area_lk)
