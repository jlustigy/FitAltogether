# http://dan.iel.fm/emcee/current/user/quickstart/
import numpy as np
import emcee
import matplotlib.pyplot as pl
from scipy.optimize import minimize
import healpy as hp


LAT_S = -0.5857506  # sub-solar latitude
LON_S = 267.6066184  # sub-solar longitude
LAT_O = 1.6808370  # sub-observer longitude
LON_O = 210.1242232 # sub-observer longitude


OMEGA = (2.0*np.pi/24.0) # [rad/hr]

deg2rad = np.pi/180.

TSTEP = 1.0 # hr
N_SLICE = 20
N_TYPE  = 3

N_SIDE   = 32
INFILE = "/Users/yuka/data/EPOXI/Earth/raddata_1_norm"


#============================================
# parmaeters for MCMC
#============================================


#===================================================
# functions
#===================================================
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
def gamma():

    Gamma_ll = np.zeros([N_SLICE, N_SLICE])
    for l1 in xrange(N_SLICE):
        for l2 in xrange(N_SLICE):
            Gamma_ll[l1][l2] = np.exp(-1.0*KAPPA*abs(l2-l1))
            
    print "Gamma_ll", Gamma_ll
    return Gamma_ll


#---------------------------------------------------
def transform_Y2X(Y_array, n_band):

    Y_albd_kj = Y_array[0:N_TYPE*n_band].reshape([N_TYPE, n_band])
    X_albd_kj = np.exp( Y_albd_kj )/( 1 + np.exp( Y_albd_kj ) )
    Y_area_lk = Y_array[N_TYPE*n_band:].reshape([N_SLICE, N_TYPE-1])
    X_area_lk = np.zeros([len(Y_area_lk), len(Y_area_lk[0]) + 1 ])
    for kk in xrange( len(Y_area_lk[0]) ):
        X_area_lk[:,kk] = ( 1. - np.sum( X_area_lk[:,:kk], axis=1 ) ) * np.exp(Y_area_lk[:,kk]) / ( 1 + np.exp(Y_area_lk[:,kk]) )
    X_area_lk[:,-1] = 1. - np.sum( X_area_lk[:,:-1], axis=1 )
    return X_albd_kj, X_area_lk


#---------------------------------------------------
def transform_X2Y(X_albd_kj, X_area_lk):

    print "X_area_lk", X_area_lk
    Y_albd_kj = np.log(X_albd_kj) - np.log(1.-X_albd_kj)
    print "Y_albd_kj", Y_albd_kj
    Y_area_lk = np.zeros([N_SLICE, N_TYPE-1])

    for kk in xrange(N_TYPE-1):
        print np.sum(X_area_lk[:,:kk+1],axis=1)
        Y_area_lk[:,kk] = np.log(X_area_lk[:,kk]) - np.log(1.-np.sum(X_area_lk[:,:kk+1], axis=1))
    print "Y_area_lk", Y_area_lk
    return np.concatenate([Y_albd_kj.flatten(), Y_area_lk.flatten()])




#---------------------------------------------------
def lnlikelihood(Y_array, *args):
    """
    Misfit-function to be minimized
    """

    Obs_ij, Sigma_ij, Kernel_il, Gamma_ll = args

    # parameter conversion
    X_albd_kj, X_area_lk = transform_Y2X(Y_array, len(Obs_ij[0]))

    # making matrix...
    Model_ij = np.dot(Kernel_il, np.dot(X_area_lk, X_albd_kj))
    Diff_ij = ( Obs_ij - Model_ij ) / Sigma_ij
    Chi2_i  = np.diag(np.dot( Diff_ij, Diff_ij.T ))
    chi2    = np.sum(Chi2_i)
    addterm_k = np.diag(np.dot(np.dot(X_area_lk.T, Gamma_ll), X_area_lk))
    addterm   = np.sum(addterm_k)
    # return
#    print "chi2", chi2
    return chi2 + LAMBDA*addterm


#---------------------------------------------------
def prior(Y_array):
    Y_array_0 = 0.1
    Diff_m = Y_array - Y_array_0
    return np.dot(Diff_m, Diff_m)



#---------------------------------------------------
def lnprob(Y_array, *arg):
    """
    Log Probability
    """
#    data, param0, icov0 = arg
    return lnlikelihood(Y_array, *arg) + prior(Y_array)




#===================================================
# main
#===================================================
if __name__ == "__main__":



    # input data
    Obsdata = np.loadtxt(INFILE)
    Time_i   = Obsdata[:,0]
#    Obs_ij = Obsdata[:,1:-1:2]
#    Sigma_ij = np.sqrt(Obsdata[:,1:-1:2])
    Obs_ij = Obsdata[:,1:]
    print  Obs_ij
    Sigma_ij = 0.05*Obs_ij
    n_time = len(Obs_ij)    
    n_band = len(Obs_ij[0])

    # set kernel
    Kernel_il = kernel(Time_i, N_SLICE)

    # set initial condition
    X0_albd_kj = 0.3+np.zeros([N_TYPE, n_band])
    X0_area_lk = 0.3+np.zeros([N_SLICE, N_TYPE-1])
    X0_albd_kj[0,0:7] = np.array([0.35, 0.28, 0.28, 0.32, 0.40, 0.47, 0.35])
    X0_albd_kj[1,0:7] = np.array([0.37, 0.28, 0.28, 0.30, 0.40, 0.48, 0.37])
    X0_albd_kj[2,0:7] = np.array([0.67, 0.52, 0.40, 0.35, 0.33, 0.33, 0.32])
    Y0_array = transform_X2Y(X0_albd_kj, X0_area_lk)

    data = (Obs_ij, Sigma_ij, Kernel_il)
    result = minimize(lnprob, Y0_array, args=data, method="SLSQP")


    n_dim = len(Y0_array)
    n_walkers = 2*n_dim**2
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob, args=data)
    pos = np.random.rand(n_dim * n_walkers).reshape((n_walkers, n_dim))-0.5 + result['x']
    sampler.run_mcmc(pos, 200)


    i = 0
    print sampler.flatchain.shape
    np.savetxt("tmp.txt", sampler.flatchain[:,i])

    pl.figure()
    pl.hist(sampler.flatchain[:,i], 200, color="k", histtype="step", range=(-50,50))
    pl.title("Dimension {0:d}".format(i))
    pl.show()
