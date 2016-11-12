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

from fitlc_params import NUM_MCMC, NUM_MCMC_BURNIN, SEED_AMP, SIGMA_Y, NOISELEVEL, \
    REGULARIZATION, deg2rad, N_SIDE, INFILE, calculate_walkers, KNOWN_ANSWER, \
    ALBDFILE, AREAFILE, SLICE_TYPE, N_SLICE_LONGITUDE, GEOM

import prior
import reparameterize

import PCA
import shrinkwrap

import geometry

NCPU = multiprocessing.cpu_count()

# March 2008
#LAT_S = -0.5857506  # sub-solar latitude
#LON_S = 267.6066184  # sub-solar longitude
#LAT_O = 1.6808370  # sub-observer longitude
#LON_O = 210.1242232 # sub-observer longitude


#===================================================
# basic functions
#=============================================== ====

N_REGPARAM = 0
if REGULARIZATION is not None:
    if REGULARIZATION == 'Tikhonov' :
        N_REGPARAM = 1
    elif REGULARIZATION == 'GP' :
        N_REGPARAM = 3
    elif REGULARIZATION == 'GP2' :
        N_REGPARAM = 2
    elif REGULARIZATION == 'GP3' :
        N_REGPARAM = 2
    elif REGULARIZATION == 'GP4' :
        N_REGPARAM = 0
else :
    N_REGPARAM = 0


#---------------------------------------------------
def lnprob(Y_array, *args):
    """
    Misfit-function to be minimized
    """

    Obs_ij, Obsnoise_ij, Kernel_il, N_REGPARAM, n_type, flip, verbose  = args
    #n_slice = len(Obs_ij)
    n_slice = len(Kernel_il[0])
    n_band = len(Obs_ij[0])
 
    # parameter conversion
    if (N_REGPARAM > 0):
        X_albd_kj, X_area_lk = reparameterize.transform_Y2X(Y_array[:-1*N_REGPARAM], n_type, n_band, n_slice )
    else:
        X_albd_kj, X_area_lk = reparameterize.transform_Y2X(Y_array, n_type, n_band, n_slice )

    # making matrix...
    Model_ij = np.dot(Kernel_il, np.dot(X_area_lk, X_albd_kj))
    Diff_ij = ( Obs_ij - Model_ij ) / Obsnoise_ij
    Chi2_i  = np.diag(np.dot( Diff_ij, Diff_ij.T ))
    chi2    = np.sum(Chi2_i)

    # flat prior for albedo
    Y_albd_kj = Y_array[0:n_type*n_band].reshape([n_type, n_band])
    ln_prior_albd = prior.get_ln_prior_albd( Y_albd_kj )

    # flat prior for area fraction
    if (N_REGPARAM > 0):
        Y_area_lk = Y_array[n_type*n_band:-1*N_REGPARAM].reshape([n_slice, n_type-1])
    else:
        Y_area_lk = Y_array[n_type*n_band:].reshape([n_slice, n_type-1])
    ln_prior_area = prior.get_ln_prior_area_new( Y_area_lk, X_area_lk[:,:-1] )

    # regularization
    # ---Tikhonov Regularization
    if REGULARIZATION is not None:
        if ( REGULARIZATION == 'Tikhonov' ):
            regparam = Y_array[-1*N_REGPARAM]
            regterm_area = prior.regularize_area_tikhonov( X_area_lk, regparam )
    # ---Gaussian Process
        elif ( REGULARIZATION == 'GP' ):
            regparam = ( Y_array[-1*N_REGPARAM], Y_array[-1*N_REGPARAM+1], Y_array[-1*N_REGPARAM+2] )
            regterm_area = prior.regularize_area_GP( X_area_lk, regparam )
    # ---Gaussian Process without constraint
        elif ( REGULARIZATION == 'GP2' ):
            regparam = ( Y_array[-1*N_REGPARAM], Y_array[-1*N_REGPARAM+1] )
            regterm_area = prior.regularize_area_GP2( X_area_lk, regparam )
    # ---Gaussian Process
        elif ( REGULARIZATION == 'GP3' ):
            regparam = ( Y_array[-1*N_REGPARAM], Y_array[-1*N_REGPARAM+1] )
            regterm_area = prior.regularize_area_GP3( X_area_lk, regparam )
    # ---Gaussian Process
        elif ( REGULARIZATION == 'GP4' ):
            regparam = ( SIGMA_Y, -10., np.pi/180.*120. )
#            regparam = ( 0.01, -10., np.pi/180.*120. )
#            regparam = ( 0.0001, -10., np.pi/180.*60. )
            regterm_area = prior.regularize_area_GP( X_area_lk, regparam )
    # ---Others
    else :
        regterm_area = 0.

    # verbose
    if verbose :
        print 'chi2', chi2 - ln_prior_albd - ln_prior_area, chi2, ln_prior_albd, ln_prior_area
        print 'chi2/d.o.f.', chi2 / (len(Y_array)*1.-1.), len(Y_array)

    answer = - chi2 + ln_prior_albd + ln_prior_area + regterm_area
    if flip :
        return -1. * answer
    else :
         return answer


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
    np.savez(run_dir+"mcmc_samples.npz", data=data, samples=original_samples, Y_names=Y_names, X_names=X_names, n_type=n_type, p0=p0)


#===================================================
if __name__ == "__main__":


    # Print start time
    now = datetime.datetime.now()
    # print now.strftime("%Y-%m-%d %H:%M:%S")
    print ''

    # Create directory for this run
    startstr = now.strftime("%Y-%m-%d--%H-%M")
    run_dir = "mcmc_output/" + startstr + "/"
    os.mkdir(run_dir)
    print "Created directory:", run_dir

    # Save THIS file and the param file for reproducibility!
    thisfile = os.path.basename(__file__)
    paramfile = "fitlc_params.py"
    priorfile = "prior.py"
    newfile = run_dir + thisfile
    commandString1 = "cp " + thisfile + " " + newfile
    commandString2 = "cp "+paramfile+" " + run_dir+paramfile
    commandString3 = "cp "+priorfile+" " + run_dir+priorfile
    os.system(commandString1)
    os.system(commandString2)
    os.system(commandString3)
    print "Saved :", thisfile, " &", paramfile

    # Load input data
    Obs_ij = np.loadtxt(INFILE)
    Obsnoise_ij = ( NOISELEVEL * Obs_ij )
#    Time_i = np.arange( len( Obs_ij ) )*1.
    Time_i  = np.arange( len( Obs_ij ) ) / ( 1.0 * len( Obs_ij ) )
    n_band = len( Obs_ij.T )

    # Initialization of Kernel
    if SLICE_TYPE == 'time' :

        print 'Decomposition into time slices...'
        n_slice = len( Time_i )
        Kernel_il = np.identity( n_slice )

    elif SLICE_TYPE == 'longitude' :

        print 'Decomposition into longitudinal slices...'
        n_slice = N_SLICE_LONGITUDE
        # (Time_i, n_slice, n_side, param_geometry):
        Kernel_il = geometry.kernel( Time_i, n_slice, N_SIDE, GEOM )

    else : 

        print '\nERROR : Unknown slice type\n'
        sys.exit()

    # Determine initial values for fitting parameters
    if KNOWN_ANSWER :

        print 'Initial values from known answers'
        X0_albd_kj = np.loadtxt( ALBDFILE ).T
        X0_area_lk = np.loadtxt( AREAFILE )

    else:

        # PCA
        print 'Performing PCA...'
        n_pc, V_nj, U_in, M_j = PCA.do_PCA( Obs_ij, E_cutoff=1e-2, run_dir=run_dir )
        n_type = n_pc + 1

        # shrinkwrap
        print 'Perfoming shrink-wrapping...'
        # N ( = n_PC ): number of principle components
        # M ( = n_PC + 1 ) : number of vertices
        A_mn, P_im   = shrinkwrap.do_shrinkwrap( U_in, n_pc, run_dir )
        X0_albd_kj   = np.dot( A_mn, V_nj )
        X0_albd_kj   = X0_albd_kj + M_j
        if ( SLICE_TYPE=='time' ) :
            X0_area_lk   = P_im
        else :
            X0_area_lk = np.ones( n_slice*n_type ).reshape([n_slice, n_type])/(n_type*1.0)

    # Save initial condutions
    np.savetxt( run_dir+'X0_albd_jk', X0_albd_kj.T )
    np.savetxt( run_dir+'X0_area_lk', X0_area_lk )


    Y0_array = reparameterize.transform_X2Y(X0_albd_kj, X0_area_lk)
    if ( N_REGPARAM > 0 ):
        Y0_array = np.append( Y0_array, np.ones( N_REGPARAM )*10. )
    n_dim = len(Y0_array)
    print '   # of parameters', n_dim

    # Create list of strings for Y & X parameter names
    Y_names, X_names = generate_tex_names( n_type, n_band, n_slice)

    ##########           old (outdated) version          ##########
    ########## use optimization for mcmc initial guesses ##########
    ### data = (Obs_ij, Obsnoise_ij, Kernel_il, N_REGPARAM, True, False)
    ### best_fit = run_initial_optimization(lnprob, data, Y0_array, method="Nelder-Mead", run_dir=run_dir)


    ########## Run MCMC ##########
    # Number of walkers
    n_walkers = calculate_walkers(n_dim)

#    Y0_array = reparameterize.transform_X2Y( X0_albd_kj, X0_area_lk )

    # Data tuple to pass to emcee
    data = (Obs_ij, Obsnoise_ij, Kernel_il, N_REGPARAM, n_type, False, False)

    # Initialize emcee EnsembleSampler object
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob, args=data, threads=NCPU)

    # Set starting guesses as gaussian noise ontop of intial optimized solution
    # note: consider using emcee.utils.sample_ball(p0, std) (std: axis-aligned standard deviation.)
    #       to produce a ball of walkers around an initial parameter value.
    p0 = SEED_AMP*np.random.rand(n_dim * n_walkers).reshape((n_walkers, n_dim)) + Y0_array


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
    np.savez(run_dir+"mcmc_samples.npz", data=data, samples=original_samples, \
             Y_names=Y_names, X_names=X_names, N_TYPE=n_type, N_SLICE=n_slice, p0=p0)

    sys.exit()
