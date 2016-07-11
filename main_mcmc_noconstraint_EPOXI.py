import numpy as np
import healpy as hp
from scipy.optimize import minimize
import emcee
import matplotlib.pyplot as pl


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

NUM_MCMC = 100
#LAMBDA = 1.0
#KAPPA  = 2.0
OMEGA = (2.0*np.pi/24.0) # [rad/hr]
N_SLICE = 21

deg2rad = np.pi/180.

N_TYPE  = 3

N_SIDE   = 32
#INFILE = "/Users/yuka/data/EPOXI/Earth/raddata_1_norm"
INFILE = "/Users/yuka/Dropbox/Exocartographer/lc_yf/mock3_lc"


#===================================================
# basic functions
#===================================================

#---------------------------------------------------
def latlon2cart(lat, lon):
    x = np.sin((90.-lat)*deg2rad)*np.cos(lon*deg2rad)
    y = np.sin((90.-lat)*deg2rad)*np.sin(lon*deg2rad)
    z = np.cos((90.-lat)*deg2rad)
    return np.array([x,y,z])


#---------------------------------------------------
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


#---------------------------------------------------
def kernel(Time_i, n_slice):
    """
    Kernel!
    """
    Kernel_il = np.zeros([len(Time_i), n_slice])
    N_pix = hp.nside2npix(N_SIDE)
    for ii in xrange(len(Time_i)):
        position_theta, position_phi   = hp.pixelfunc.pix2ang(N_SIDE, np.arange(N_pix))
        Weight_n = weight(Time_i[ii])
        assigned_l = np.trunc(position_phi/(2.*np.pi/n_slice))
        Count_l = np.zeros(n_slice)
        for nn in xrange(N_pix):
            Kernel_il[ii][assigned_l[nn]] += Weight_n[nn]
            Count_l[assigned_l[nn]]       += 1
        Kernel_il[ii] = Kernel_il[ii]/Count_l
        Kernel_il[ii] = Kernel_il[ii]/np.sum(Kernel_il[ii])
    return Kernel_il


#---------------------------------------------------
def sigma(sigma, kappa, dim, periodic=False):

#    kappa0 = np.log(output["x"][-1]) - np.log(360.0 - output["x"][-1])
    kappa0 = np.exp(kappa)/(1 + np.exp(kappa))
    Sigma_ll = np.zeros([dim, dim])
    for l1 in xrange(dim):
        for l2 in xrange(dim):

            if periodic:
                diff = min(abs(l2-l1), dim-abs(l2-l1))
            else:
                diff = abs(l2-l1)

            Sigma_ll[l1][l2] = np.exp(-diff/kappa0)

#    print "Sigma_ll", Sigma_ll
#    print np.dot(Sigma_ll, inv_Sigma_ll)
#    Sigma2_ll = sigma**2*(Sigma_ll + np.identity(N_SLICE))
    Sigma2_ll = sigma**2*Sigma_ll
    return Sigma2_ll 

#---------------------------------------------------
def misfitfunc(Y_array, *args):
    """
    Misfit-function to be minimized
    """

    Obs_ij, Obsnoise_ij, Kernel_il  = args
    n_band = len(Obs_ij[0])

    # parameter conversion
    X_albd_kj, X_area_lk = transform_Y2X(Y_array, len(Obs_ij[0]))

    # making matrix...
    Model_ij = np.dot(Kernel_il, np.dot(X_area_lk, X_albd_kj))
    Diff_ij = ( Obs_ij - Model_ij ) / Obsnoise_ij
    Chi2_i  = np.diag(np.dot( Diff_ij, Diff_ij.T ))
    chi2    = np.sum(Chi2_i)

    # return
    return chi2


#---------------------------------------------------
def transform_Y2X(Y_array, n_band):

    X_albd_kj = Y_array[0:N_TYPE*n_band].reshape([N_TYPE, n_band])
    X_area_lk = Y_array[N_TYPE*n_band:].reshape([N_SLICE, N_TYPE])
#    X_area_lk = Y_array[N_TYPE*n_band:].reshape([N_SLICE, N_TYPE-1])
#    X_area_add = 1. - np.sum( X_area_lk[:,:], axis=1 )
#    X_area_lk = np.c_[X_area_lk, X_area_add]
    return X_albd_kj, X_area_lk


def transform_X2Y(X_albd_kj, X_area_lk):

#    return np.concatenate([X_albd_kj.flatten(), X_area_lk[:,:-1].flatten()])
    return np.concatenate([X_albd_kj.flatten(), X_area_lk.flatten()])



#===================================================
if __name__ == "__main__":

    # input data
    Obsdata = np.loadtxt(INFILE)
    Time_i   = Obsdata[:,0]*1.0
#    Obs_ij = Obsdata[:,1:-1:2]
#    Obsnoise_ij = np.sqrt(Obsdata[:,1:-1:2])
    Obs_ij = Obsdata[:,1:]
#    print  Obs_ij
#    Obsnoise_ij = 0.05*Obs_ij
    Obsnoise_ij = 1.0 + 0.*Obs_ij
    n_time = len(Obs_ij)    
    n_band = len(Obs_ij[0])

    # set kernel
    Kernel_il = kernel(Time_i, N_SLICE)
#    Sigma_ll = np.identity(N_SLICE)

#    set initial condition
    X0_albd_kj = 0.3+np.zeros([N_TYPE, n_band])
    X0_area_lk = 0.1+np.zeros([N_SLICE, N_TYPE])

#    X0_area_add = 1. - np.sum( X0_area_lk[:,:], axis=1 )
#    X0_area_lk = np.c_[X0_area_lk, X0_area_add]

    Y0_array = transform_X2Y(X0_albd_kj, X0_area_lk)

    X_albd_kj, X_area_lk =  transform_Y2X(Y0_array, n_band)

    # minimize
    data = (Obs_ij, Obsnoise_ij, Kernel_il)
    output = minimize(misfitfunc, Y0_array, args=data, method="Nelder-Mead")

    X_albd_kj, X_area_lk =  transform_Y2X(output["x"], n_band)
    np.savetxt("X_area_lk", X_area_lk)
    np.savetxt("X_albd_kj_T", X_albd_kj.T)

    print "best-fit", output["x"]
    X_albd_kj, X_area_lk =  transform_Y2X(output["x"], n_band)
    np.savetxt("X_area_lk", X_area_lk)
    np.savetxt("X_albd_kj_T", X_albd_kj.T)



    print "MCMC..."
    n_dim = len(Y0_array)
    n_walkers = 2*n_dim**2
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, misfitfunc, args=data)
    pos = 0.01*np.random.rand(n_dim * n_walkers).reshape((n_walkers, n_dim)) + output["x"]
    sampler.run_mcmc(pos, NUM_MCMC)

    np.savetxt("tmp1.txt", sampler.flatchain[:,0])
    np.savetxt("tmp15.txt", sampler.flatchain[:,15])

#    X_albd_kj_list = []
#    X_area_lk_list = []
    histo1 = []
    histo2 = []
    for nn in xrange(50,len(sampler.flatchain)):
        X_albd_kj, X_area_lk = transform_Y2X(sampler.flatchain[nn], n_band)
        histo1.append(X_albd_kj[0][0])
        histo2.append(X_area_lk[0][0])
#       X_albd_kj_list.append(X_albd_kj)
#       X_area_lk_list.append(X_albd_kj)

    pl.figure()
    pl.hist(histo1, 100, color="k", histtype="step", range=(-10,10))
    pl.show()

    pl.hist(histo2, 100, color="k", histtype="step", range=(0,1))
#    pl.title("Dimension {0:d}".format(i))
    pl.show()


