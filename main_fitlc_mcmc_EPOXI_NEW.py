import numpy as np
import healpy as hp
import emcee
import matplotlib.pyplot as pl
from scipy.optimize import minimize
import sys
import corner
import datetime
import multiprocessing
import os
import pdb


# March 2008
#LAT_S = -0.5857506  # sub-solar latitude
#LON_S = 267.6066184  # sub-solar longitude
#LAT_O = 1.6808370  # sub-observer longitude
#LON_O = 210.1242232 # sub-observer longitude

#NUM_MCMC = 2
#NUM_MCMC_BURNIN = 1

NUM_MCMC = 1000
NUM_MCMC_BURNIN = 600
SEED_AMP = 0.5

# Set the number of CPUs on current machine for the MCMC
NCPU = multiprocessing.cpu_count()

SIGMA_Y  = 3.0
NOISELEVEL = 0.01

FLAG_REG_AREA = False
FLAG_REG_ALBD = False

#n_slice = 4
N_TYPE  = 2

deg2rad = np.pi/180.

N_SIDE   = 32
#INFILE = "data/raddata_12_norm"
#INFILE = "data/raddata_2_norm"
INFILE = "mockdata/mock_simple_1_data"
#INFILE = "mockdata/mock_simple_3types_1_data"
# INFILE = 'mockdata/mock_simple_1_scattered0.01_data_with_noise'


#===================================================
# basic functions
#=============================================== ====

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
#    Sigma2_ll = sigma**2*(Sigma_ll + np.identity(n_slice))
    Sigma2_ll = sigma**2*Sigma_ll
    return Sigma2_ll



#---------------------------------------------------
def get_ln_prior_albd( y_albd_kj ):
    prior_kj = np.exp( y_albd_kj ) / ( 1 + np.exp( y_albd_kj ) )**2
    ln_prior = np.sum( np.log( prior_kj ) )
    return ln_prior

#---------------------------------------------------
def get_ln_prior_area( y_area_lk, x_area_lk):
    # x_area_lk is a dummy arg (can be deleted)
    prior_lk = np.exp( y_area_lk ) / ( 1 + np.exp( y_area_lk ) )**2
    ln_prior = np.log( np.prod( prior_lk ) )
    return ln_prior


"""
#---------------------------------------------------
def get_ln_prior_area( y_area_lk, x_area_lk ):

    l_dim = len( y_area_lk )
    k_dim = len( y_area_lk.T )
    kk_dim = len( y_area_lk.T )

    sumF = np.cumsum( x_area_lk, axis=1 )

    # when kk < k
    l_indx, k_indx, kk_indx = np.meshgrid( np.arange( l_dim ), np.arange( k_dim ), np.arange( kk_dim ), indexing='ij' )
    dgdF = np.zeros( [ l_dim, k_dim, kk_dim  ] )
    dgdF[ l_indx, k_indx, kk_indx ] = x_area_lk[ l_indx, kk_indx ] / x_area_lk[ l_indx, k_indx ] / ( 1 - sumF[ l_indx, k_indx ] )

    # when kk > k
    k_tmp, kk_tmp   = np.triu_indices(k_dim)
    l_indx, k_indx  = np.meshgrid( np.arange( l_dim ), k_tmp,  indexing='ij' )
    l_indx, kk_indx = np.meshgrid( np.arange( l_dim ), kk_tmp, indexing='ij' )
    dgdF[ l_indx, k_indx, kk_indx ] = 0.

    # when kk = k
    k_tmp, kk_tmp = np.diag_indices(k_dim)
    l_indx, k_indx  = np.meshgrid( np.arange( l_dim ), k_tmp,  indexing='ij' )
    l_indx, kk_indx = np.meshgrid( np.arange( l_dim ), kk_tmp, indexing='ij' )
    dgdF[ l_indx, k_indx, kk_indx ] = 1./x_area_lk[l_indx,k_indx]*(1. - sumF[l_indx, k_indx-1]) / ( 1 - sumF[ l_indx, k_indx ] )

    dgdF_factor = np.linalg.det( dgdF )
    ln_prior = np.sum( np.log( dgdF_factor ) )

    return ln_prior
"""


#---------------------------------------------------
def get_ln_prior_area_old( y_area_lj, x_area_lj ):

    dydx_det = 1.
    for ll in xrange( len( y_area_lj ) ):
        dydx = np.zeros( [ len( y_area_lj.T ), len( y_area_lj.T ) ] )
        for ii in xrange( len( dydx ) ):
            jj = 0
            # jj < ii
            while ( jj < ii ):
                g_i = y_area_lj[ll,ii]
                f_i = x_area_lj[ll,ii]
                f_j = x_area_lj[ll,jj]
                sum_fi = np.sum( x_area_lj[ll,:ii+1] )
                dydx[ii][jj] = 1. / ( 1. - sum_fi )
                jj = jj + 1
            # jj == ii
            g_i = y_area_lj[ll,ii]
            f_i = x_area_lj[ll,ii]
            f_j = x_area_lj[ll,jj]
            sum_fi = np.sum( x_area_lj[ll,:ii+1] )
            dydx[ii][jj] = 1. / f_i * ( 1. - sum_fi + f_i ) / ( 1 - sum_fi )

#        print "dydx", dydx
#        print "det", np.linalg.det( dydx )
        dydx_det = dydx_det * np.linalg.det( dydx )
    dxdy_det = 1. / dydx_det

    if ( dxdy_det <= 0. ):
        print "ERROR! ln_prior_area is NaN"
        print "     ", dxdy_det
        sys.exit()

    ln_prior = np.log( dxdy_det )
    return ln_prior


#---------------------------------------------------
def lnprob(Y_array, *args):
    """
    Misfit-function to be minimized
    """

    Obs_ij, Obsnoise_ij, Kernel_il, n_param, flip, verbose  = args
    n_slice = len(Obs_ij)
    n_band = len(Obs_ij[0])


    # parameter conversion
    if (n_param > 0):
        X_albd_kj, X_area_lk = transform_Y2X(Y_array[:-1*n_param], len(Obs_ij[0]))
    else:
        X_albd_kj, X_area_lk = transform_Y2X(Y_array, len(Obs_ij[0]))

    # making matrix...
    Model_ij = np.dot(Kernel_il, np.dot(X_area_lk, X_albd_kj))
    Diff_ij = ( Obs_ij - Model_ij ) / Obsnoise_ij
    Chi2_i  = np.diag(np.dot( Diff_ij, Diff_ij.T ))
    chi2    = np.sum(Chi2_i)

    # flat prior for albedo
    Y_albd_kj = Y_array[0:N_TYPE*n_band].reshape([N_TYPE, n_band])
    ln_prior_albd = get_ln_prior_albd( Y_albd_kj )

    # flat prior for area fraction
    Y_area_lk = Y_array[N_TYPE*n_band:].reshape([n_slice, N_TYPE-1])
    ln_prior_area = get_ln_prior_area( Y_area_lk, X_area_lk[:,:-1] )

    if verbose :
        print 'chi2', chi2 - ln_prior_albd - ln_prior_area, chi2, ln_prior_albd, ln_prior_area
        print 'chi2/d.o.f.', chi2 / (len(Y_array)*1.-1.), len(Y_array)

    # regularization term for area fraction
    if FLAG_REG_AREA :
        if FLAG_REG_ALBD :
            origin = -4
        else:
            origin = -2
        Sigma_ll      = sigma(Y_array[origin], Y_array[origin+1], n_slice, periodic=True)
        inv_Sigma_ll  = np.linalg.inv(Sigma_ll)
        addterm_k     = np.diag(np.dot(np.dot(X_area_lk.T, inv_Sigma_ll), X_area_lk))
        r_areafrac_1  = np.sum(addterm_k)
        r_areafrac_2 = np.log(np.linalg.det(Sigma_ll))

    else:
        r_areafrac_1 = 0.
        r_areafrac_2 = 0.

    # refularization term for albedo
    if FLAG_REG_ALBD :
        Sigma_jj      = sigma(Y_array[-2], Y_array[-1], n_band)
    #    print "Sigma_jj", Sigma_jj
        inv_Sigma_jj  = np.linalg.inv(Sigma_jj)
        addterm_k     = np.diag(np.dot(np.dot(X_albd_kj, inv_Sigma_jj), X_albd_kj.T))
        r_albd_1 = np.sum(addterm_k)
        r_albd_2 = np.log(np.linalg.det(Sigma_jj))
    else:
        r_albd_1  = 0.
        r_albd_2  = 0.


    r_y  = 1.0/SIGMA_Y**2*np.dot(Y_array-0.5, Y_array-0.5)

    # return
#    print "chi2", chi2
#    return chi2 + r_areafrac_1 + r_areafrac_2 + r_albd_1 + r_albd_2 + r_y
    if flip :
        return chi2 - ln_prior_albd - ln_prior_area
    else :
        return - chi2 + ln_prior_albd + ln_prior_area

#---------------------------------------------------
def transform_Y2X(Y_array, n_band):

    Y_array = np.maximum(Y_array, -10)
    Y_array = np.minimum(Y_array, 10)

    Y_albd_kj = Y_array[0:N_TYPE*n_band].reshape([N_TYPE, n_band])
    X_albd_kj = np.exp( Y_albd_kj )/( 1 + np.exp( Y_albd_kj ) )
    Y_area_lk = Y_array[N_TYPE*n_band:].reshape([n_slice, N_TYPE-1])
    X_area_lk = np.zeros([len(Y_area_lk), len(Y_area_lk[0]) + 1 ])
    for kk in xrange( len(Y_area_lk[0]) ):
        X_area_lk[:,kk] = ( 1. - np.sum( X_area_lk[:,:kk], axis=1 ) ) * np.exp(Y_area_lk[:,kk]) / ( 1 + np.exp(Y_area_lk[:,kk]) )
    X_area_lk[:,-1] = 1. - np.sum( X_area_lk[:,:-1], axis=1 )
    return X_albd_kj, X_area_lk


#---------------------------------------------------
def transform_X2Y(X_albd_kj, X_area_lk, n_slice):
    """
    Re-parameterization for convenience -- now Y can take on any value.
    """

#    print "X_area_lk", X_area_lk
    Y_albd_kj = np.log(X_albd_kj) - np.log(1.-X_albd_kj)
#    print "Y_albd_kj", Y_albd_kj
    Y_area_lk = np.zeros([n_slice, N_TYPE-1])
    print 'X_area_lk', X_area_lk.shape
    print 'Y_area_lk', Y_area_lk.shape
    for kk in xrange(N_TYPE-1):
#        print np.sum(X_area_lk[:,:kk+1],axis=1)
        Y_area_lk[:,kk] = np.log(X_area_lk[:,kk]) - np.log(1.-np.sum(X_area_lk[:,:kk+1], axis=1))
#    print "Y_area_lk", Y_area_lk
    return np.concatenate([Y_albd_kj.flatten(), Y_area_lk.flatten()])

#---------------------------------------------------
def generate_tex_names(n_type, n_band, n_slice):
    """
    Generate an array of Latex strings for each parameter in the
    X and Y vectors.

    Returns
    -------
    Y_names : array
        Non-physical fitting parameters
    X_names : array
        Physical parameters for Albedo and Surface Area Fractions
    """
    # Create list of strings for Y parameter names
    btmp = []
    gtmp = []
    for i in range(n_type):
        for j in range(n_band):
            btmp.append(r"b$_{"+str(i+1)+","+str(j+1)+"}$")
    for j in range(n_type - 1):
        for i in range(n_slice):
            gtmp.append(r"g$_{"+str(i+1)+","+str(j+1)+"}$")
    Y_names = np.concatenate([np.array(btmp), np.array(gtmp)])

    # Create list of strings for X parameter names
    Atmp = []
    Ftmp = []
    for i in range(n_type):
        for j in range(n_band):
            Atmp.append(r"A$_{"+str(i+1)+","+str(j+1)+"}$")
    for j in range(n_type):
        for i in range(n_slice):
            Ftmp.append(r"F$_{"+str(i+1)+","+str(j+1)+"}$")
    X_names = np.concatenate([np.array(Atmp), np.array(Ftmp)])

    return Y_names, X_names

#---------------------------------------------------
def run_initial_optimization(lnlike, data, guess, method="Nelder-Mead", run_dir=""):

    print "Finding initial best-fit values using %s method..." %method

    # Decompose data
    Obs_ij = data[0]
    n_slice = len(Obs_ij)
    n_band = len(Obs_ij[0])

    # Run optimization
    output = minimize(lnlike, guess, args=data, method=method)

    # Get best-fitting params
    best_fit = output["x"]
    print "initial best-fit:", best_fit

    # Get best-lnlike and BIC
    lnprob_bestfit = lnlike( output['x'], *data )
    BIC = 2.0 * lnprob_bestfit + len( output['x'] ) * np.log( len(Obs_ij.flatten()) )
    print 'BIC: ', BIC

    # Transform back to physical params
    X_albd_kj, X_area_lk =  transform_Y2X(output["x"], n_band)
    X_albd_kj_T = X_albd_kj.T

    # Flatten best-fitting physical parameters
    bestfit = np.r_[ X_albd_kj.flatten(), X_area_lk.T.flatten() ]

    # Calculate residuals
    residuals = Obs_ij - np.dot( X_area_lk, X_albd_kj )
    print "residuals", residuals

    # Save initialization run as npz
    print "Saving:", run_dir+"initial_minimize.npz"
    np.savez(run_dir+"initial_minimize.npz", data=data, best_fity=best_fit, \
        lnprob_bestfit=lnprob_bestfit, BIC=BIC, X_area_lk=X_area_lk, \
        X_albd_kj_T=X_albd_kj_T, residuals=residuals, best_fitx =bestfit)

    return best_fit

#---------------------------------------------------
def run_emcee(lnlike, data, guess, N=500, run_dir="", seed_amp=0.01, *args):

    print "MCMC until burn-in..."

    # Number of dimensions is number of free parameters
    n_dim = len(guess)
    # Number of walkers
    n_walkers = 2*n_dim**2

    # Initialize emcee EnsembleSampler object
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob, args=data, threads=NCPU)

    # Set starting guesses as gaussian noise ontop of intial optimized solution
    # note: consider using emcee.utils.sample_ball(p0, std) (std: axis-aligned standard deviation.)
    #       to produce a ball of walkers around an initial parameter value.
    p0 = seed_amp * np.random.rand(n_dim * n_walkers).reshape((n_walkers, n_dim)) + best_fit

    # Run MCMC
    sampler.run_mcmc( p0, N )

    original_samples = sampler.chain

    print "Saving:", run_dir+"mcmc_samples.npz"
    np.savez(run_dir+"mcmc_samples.npz", data=data, samples=original_samples, Y_names=Y_names, X_names=X_names, N_TYPE=N_TYPE, p0=p0)


#===================================================
if __name__ == "__main__":

    # print start time
    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")

    # Create directory for this run
    startstr = now.strftime("%Y-%m-%d--%H-%M")
    run_dir = "mcmc_output/" + startstr + "/"
    os.mkdir(run_dir)
    print "Created directory:", run_dir

    # Save THIS file for reproducibility!
    thisfile = os.path.basename(__file__)
    newfile = run_dir + thisfile
    commandString = "cp " + thisfile + " " + newfile
    os.system(commandString)

    # input data
    Obs_ij = np.loadtxt(INFILE)
    n_slice = len(Obs_ij)

    n_band = len(Obs_ij[0])
    Time_i = np.arange( n_slice )

    Obsnoise_ij = ( NOISELEVEL * Obs_ij )

    # set kernel
#    Kernel_il = kernel(Time_i, n_slice)
    Kernel_il = np.identity( n_slice )
#    Sigma_ll = np.identity(n_slice)

#    print 1/0
#    set initial condition
#    Y0_array = np.ones(N_TYPE*n_band+n_slice*(N_TYPE-1))
    X0_albd_kj = 0.3+np.zeros([N_TYPE, n_band])
    X0_area_lk = 0.1+np.zeros([n_slice, N_TYPE-1])
    """ # Load perfect starting position from file
    temp = np.load("mockdata/mock_simple_3types_1_albd_area.npz")
    X0_albd_kj = temp["X0_albd_kj"]
    X0_area_lk = temp["X0_area_lk"]
    """

    # Create list of strings for Y & X parameter names
    Y_names, X_names = generate_tex_names(N_TYPE, n_band, n_slice)

    Y0_array = transform_X2Y(X0_albd_kj, X0_area_lk, n_slice)
    n_dim = len(Y0_array)
    print '# of parameters', n_dim

    n_param = 0
    if FLAG_REG_AREA:
        Y0_array = np.append(Y0_array, [1.0, 1.0])
        n_param += 2
    if FLAG_REG_ALBD:
        Y0_array = np.append(Y0_array, [1.0, 1.0])
        n_param += 2

#    Y0_albd_kj = np.zeros([N_TYPE,  len(Obs_ij[0])])
#    Y0_area_lk = np.zeros([n_slice, N_TYPE-1])
#    Y0_area_lk[:,0] = 1.
#    Y0_list = [Y0_albd_kj, Y0_area_lk]
#    print "Y0_array", Y0_array

    if (n_param > 0):
        X_albd_kj, X_area_lk =  transform_Y2X(Y0_array[:-1*n_param], n_band)
    else:
        X_albd_kj, X_area_lk =  transform_Y2X(Y0_array, n_band)

#    print "X_area_lk", X_area_lk
#    print "X_albd_kj", X_albd_kj

    ########## use optimization for mcmc initial guesses ##########

    data = (Obs_ij, Obsnoise_ij, Kernel_il, n_param, True, False)

    best_fit = run_initial_optimization(lnprob, data, Y0_array, method="Nelder-Mead", run_dir=run_dir)

    """
    # minimize
    print "finding best-fit values..."
    data = (Obs_ij, Obsnoise_ij, Kernel_il, n_param, True, False)
    output = minimize(lnprob, Y0_array, args=data, method="Nelder-Mead")

    best_fit = output["x"]
    print "best-fit", best_fit

    data = (Obs_ij, Obsnoise_ij, Kernel_il, n_param, True, False)
    lnprob_bestfit = lnprob( output['x'], *data )
    BIC = 2.0 * lnprob_bestfit + len( output['x'] ) * np.log( len(Obs_ij.flatten()) )
    print 'BIC: ', BIC

    # Transform back to physical params
    X_albd_kj, X_area_lk =  transform_Y2X(output["x"], n_band)
    X_albd_kj_T = X_albd_kj.T
    #np.savetxt("X_area_lk", X_area_lk)
    #np.savetxt("X_albd_kj_T", X_albd_kj.T)
    bestfit = np.r_[ X_albd_kj.flatten(), X_area_lk.T.flatten() ]

    # Calculate residuals
    residuals = Obs_ij - np.dot( X_area_lk, X_albd_kj )
    print "residuals", residuals

    # Save initialization run as npz
    print "Saving:", run_dir+"initial_minimize.npz"
    np.savez(run_dir+"initial_minimize.npz", data=data, best_fity=best_fit, \
        lnprob_bestfit=lnprob_bestfit, BIC=BIC, X_area_lk=X_area_lk, \
        X_albd_kj_T=X_albd_kj_T, residuals=residuals, best_fitx = bestfit)
    """

    ########## Run MCMC ##########

    # Data tuple to pass to emcee
    data = (Obs_ij, Obsnoise_ij, Kernel_il, n_param, False, False)

    #run_emcee(lnprob, data, best_fit, run_dir=run_dir)

    # Number of dimensions is number of free parameters
    n_dim = len(Y0_array)
    # Number of walkers
    n_walkers = 2*n_dim**2

    # Data tuple to pass to emcee
    data = (Obs_ij, Obsnoise_ij, Kernel_il, n_param, False, False)

    # Initialize emcee EnsembleSampler object
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob, args=data, threads=NCPU)

    # Set starting guesses as gaussian noise ontop of intial optimized solution
    # note: consider using emcee.utils.sample_ball(p0, std) (std: axis-aligned standard deviation.)
    #       to produce a ball of walkers around an initial parameter value.
    p0 = SEED_AMP*np.random.rand(n_dim * n_walkers).reshape((n_walkers, n_dim)) + best_fit

    if NUM_MCMC_BURNIN > 0:
        print "MCMC until burn-in..."
        # Run MCMC
        pos, prob, state = sampler.run_mcmc( p0, NUM_MCMC_BURNIN )
        # Save initial positions of chain[n_walkers, steps, n_dim]
        burnin_chain = sampler.chain[:, :, :].reshape((-1, n_dim))
        # Save chain[n_walkers, steps, n_dim] as npz
        now = datetime.datetime.now()
        print "Finished Burn-in MCMC:", now.strftime("%Y-%m-%d %H:%M:%S")
        print "Saving:", run_dir+"mcmc_burnin.npz"
        np.savez(run_dir+"mcmc_burnin.npz", pos=pos, prob=prob, burnin_chain=burnin_chain)
        print "MCMC from burn-in..."
        # Set initial starting position to the current state of chain
        p0 = pos
        # Reset sampler for production run
        sampler.reset()
    else:
        print "MCMC from initial optimization..."

    # Run MCMC
    sampler.run_mcmc( p0, NUM_MCMC )

    original_samples = sampler.chain

    print "Saving:", run_dir+"mcmc_samples.npz"
    np.savez(run_dir+"mcmc_samples.npz", data=data, samples=original_samples, Y_names=Y_names, X_names=X_names, N_TYPE=N_TYPE, p0=p0)

    sys.exit()
