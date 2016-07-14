import numpy as np
import matplotlib.pyplot as pl
import sys
import datetime
import multiprocessing
from scipy.optimize import minimize

import healpy as hp
import emcee
import corner

import geometry
import prior
from reparameterize import *

#--------------------------------------------------------------------
# Parameters
#--------------------------------------------------------------------

REGULARIZATION = 'GP'
#REGULARIZATION = 'GP2'
#REGULARIZATION = 'Tikhonov'

#n_slice = 4
N_TYPE  = 3
N_SLICE = 13

MONTH = 'June'

NOISELEVEL = 0.01

NUM_MCMC = 10
NUM_MCMC_BURNIN = 100
NCPU = multiprocessing.cpu_count()

N_side_seed = 2
N_SIDE  = 2*2**N_side_seed

Pspin = 24.0
OMEGA = ( 2. * np.pi / Pspin )



#--------------------------------------------------------------------
# set-up
#--------------------------------------------------------------------

if ( MONTH == 'March' ):
# from spectroscopic data
#         Sub-Sun Lon/Lat =      97.091       -0.581 /     W longitude, degrees 
#         Sub-SC  Lon/Lat =     154.577        1.678 /     W longitude, degrees
    LAT_S = -0.581  # sub-solar latitude
    LON_S = 262.909  # sub-solar longitude
    LAT_O = 1.678  # sub-observer latitude
    LON_O = 205.423 # sub-observer longitude
    INFILE = "data/raddata_1_norm"
    Time_i = np.arange(25)*1.

elif ( MONTH == 'June' ):
# from spectroscopic data
#         Sub-Sun Lon/Lat =      79.023       22.531 /     W longitude, degrees
#         Sub-SC  Lon/Lat =     154.535        0.264 /     W longitude, degrees
    LON_S = 280.977
    LAT_S = 22.531
    LON_O = 205.465
    LAT_O = 0.264
#    LON_O = 165.4663412
#    LAT_O = -0.3521857
#    LON_S = 239.1424068
#    LAT_S = 21.6159766
    INFILE = "data/raddata_2_norm"
    Time_i = np.arange(25)*1.

else :
    print 'ERROR: Invalid MONTH'
    sys.exit()

N_REGPARAM = 0
if 'REGULARIZATION' in globals():
    if REGULARIZATION == 'Tikhonov' :
        N_REGPARAM = 1
    elif REGULARIZATION == 'GP' :
        N_REGPARAM = 3
    elif REGULARIZATION == 'GP2' :
        N_REGPARAM = 2
else :
    N_REGPARAM = 0

#--------------------------------------------------------------------
# log ( posterior probability  )
#--------------------------------------------------------------------

def lnprob(Y_array, *args):
    """
    Misfit-function to be minimized
    """

    Obs_ij, Obsnoise_ij, Kernel_il, n_regparam, flip, verbose  = args
    n_band = len(Obs_ij[0])

    # parameter conversion
    if ( n_regparam > 0 ):
        X_albd_kj, X_area_lk = transform_Y2X(Y_array[:-1*n_regparam], N_TYPE, n_band, N_SLICE)
    else:
        X_albd_kj, X_area_lk = transform_Y2X(Y_array, N_TYPE, n_band, N_SLICE)

    # making matrix...
    Model_ij = np.dot(Kernel_il, np.dot(X_area_lk, X_albd_kj))
    Diff_ij = ( Obs_ij - Model_ij ) / Obsnoise_ij
    Chi2_i  = np.diag(np.dot( Diff_ij, Diff_ij.T ))
    chi2    = np.sum( Chi2_i )

    # flat prior for albedo
    Y_albd_kj = Y_array[0:N_TYPE*n_band].reshape([N_TYPE, n_band])
    ln_prior_albd = prior.get_ln_prior_albd( Y_albd_kj )

    # flat prior for area fraction
    Y_area_lk = Y_array[N_TYPE*n_band:N_TYPE*n_band+N_SLICE*(N_TYPE-1)].reshape([N_SLICE, N_TYPE-1])
    ln_prior_area =  prior.get_ln_prior_area_new( Y_area_lk, X_area_lk[:,:-1] )

    # regularization
    # ---Tikhonov Regularization
    if 'REGULARIZATION' in globals():
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



#===================================================
if __name__ == "__main__":

    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")

    # input data
    Obs_ij = np.loadtxt(INFILE)
    Obsnoise_ij = ( NOISELEVEL * Obs_ij )
    n_band = len(Obs_ij[0])

    # set kernel
    param_geometry = ( LAT_O, LON_O, LAT_S, LON_S, OMEGA )
    Kernel_il = geometry.kernel( Time_i, N_SLICE, N_SIDE, param_geometry )
 
    # initialize the fitting parameters
    X0_albd_kj = 0.3+np.zeros([N_TYPE, n_band])
    X0_area_lk = 0.2+np.zeros([N_SLICE, N_TYPE])
    Y0_array = transform_X2Y(X0_albd_kj, X0_area_lk)
    if ( N_REGPARAM > 0 ) :
        Y0_array = np.append(Y0_array, np.array([10.]*N_REGPARAM) )
    n_dim = len(Y0_array)
    print 'Y0_array', Y0_array
    print '# of parameters', n_dim
    print 'N_REGPARAM', N_REGPARAM
    if (N_REGPARAM > 0):
        X_albd_kj, X_area_lk =  transform_Y2X(Y0_array[:-1*N_REGPARAM], N_TYPE, n_band, N_SLICE)
    else:
        X_albd_kj, X_area_lk =  transform_Y2X(Y0_array, N_TYPE, n_band, N_SLICE)

    # minimize
    print "finding best-fit values..."
    data = (Obs_ij, Obsnoise_ij, Kernel_il, N_REGPARAM, True, False)
    output = minimize(lnprob, Y0_array, args=data, method="Nelder-Mead")
#    output = minimize(lnprob, Y0_array, args=data, method="L-BFGS-B" ) 
    best_fit = output["x"]
    print "best-fit", best_fit

    # more information about the best-fit parameters
    data = (Obs_ij, Obsnoise_ij, Kernel_il, N_REGPARAM, True, False)
    lnprob_bestfit = lnprob( output['x'], *data )

    # best-fit values for physical parameters
    if N_REGPARAM > 0:
        X_albd_kj, X_area_lk =  transform_Y2X(output["x"][:-1*N_REGPARAM], N_TYPE, n_band, N_SLICE)
    else :
        X_albd_kj, X_area_lk =  transform_Y2X(output["x"], N_TYPE, n_band, N_SLICE)
    np.savetxt("X_area_lk", X_area_lk)
    np.savetxt("X_albd_kj_T", X_albd_kj.T)

    # best-fit values for regularizing parameters
    if 'REGULARIZATION' in globals():
        if REGULARIZATION == 'Tikhonov' :
            print 'sigma', best_fit[-1] 
        elif REGULARIZATION == 'GP' :
            print 'overall_amp', best_fit[-3]
            print 'wn_rel_amp', np.exp( best_fit[-2] ) / ( 1. + np.exp( best_fit[-2] ) )
            print 'lambda _angular', best_fit[-1] * ( 180. / np.pi )
        elif REGULARIZATION == 'GP2' :
            print 'overall_amp', best_fit[-2]
            print 'lambda _angular', best_fit[-1]* ( 180. / np.pi )
 

    sys.exit()

    print "MCMC until burn-in..."
    n_dim = len(Y0_array)
    n_walkers = 2*n_dim**2
    data = (Obs_ij, Obsnoise_ij, Kernel_il, N_REGPARAM, False, False)
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob, args=data, threads=NCPU)
#    pos = 0.01*np.random.rand(n_dim * n_walkers).reshape((n_walkers, n_dim)) + output["x"]
    p0 = 0.1*np.random.rand(n_dim * n_walkers).reshape((n_walkers, n_dim)) + output["x"]
    pos, prob, state = sampler.run_mcmc( p0, NUM_MCMC_BURNIN )
    np.savetxt( 'tmp.txt', sampler.chain[0,:,1] )

    print "MCMC from burn-in..."
    sampler.reset()
    sampler.run_mcmc( pos, NUM_MCMC )
    samples = sampler.chain[:, :, :].reshape((-1, n_dim)) # trial x n_dim
    X_array = np.zeros( [ len( samples ), N_TYPE*n_band + N_SLICE*N_TYPE ] )

    print 'accumulation...'
    print len( samples )


    X_albd_kj, X_area_lk =  transform_Y2X( samples[0], N_TYPE, n_band, N_SLICE)
    X_array[0] = np.r_[ X_albd_kj.flatten(), X_area_lk.T.flatten() ]

    X_albd_kj_stack = X_albd_kj
    X_area_lk_stack = X_area_lk

    for ii in xrange( 1, len( samples ) ):
        if ii % 1000 == 0 :
            print ii

        X_albd_kj, X_area_lk =  transform_Y2X( samples[ii], N_TYPE, n_band, N_SLICE)
        X_array[ii] = np.r_[ X_albd_kj.flatten(), X_area_lk.T.flatten() ]

        X_albd_kj_stack = np.dstack([ X_albd_kj_stack, X_albd_kj ])
        X_area_lk_stack = np.dstack([ X_area_lk_stack, X_area_lk ])

    print 'evaluation...'

    X_albd_error = np.percentile( X_albd_kj_stack, [16, 50, 84], axis=2)

#    X_albd_error = map(lambda v: np.array([v[1], v[2]-v[1], v[1]-v[0]]),
#                       zip(*np.percentile( X_albd_kj_stack, [16, 50, 84], axis=2)))

    print X_albd_error
#    print X_albd_error.shape

    for jj in  xrange( n_band ):
        for kk in  xrange( N_TYPE ):
            print X_albd_error[1][kk][jj], X_albd_error[2][kk][jj], X_albd_error[0][kk][jj],
        print ''

    print ''
    print ''

    X_area_error = np.percentile( X_area_lk_stack, [16, 50, 84], axis=2)
#    X_area_error = map(lambda v: np.array([v[1], v[2]-v[1], v[1]-v[0]]),
#                       zip(*np.percentile( X_area_lk_stack, [16, 50, 84], axis=2)))

    for ll in xrange( N_SLICE ):
        for kk in  xrange( N_TYPE ):
            print X_area_error[1][ll][kk], X_area_error[2][ll][kk], X_area_error[0][ll][kk],
        print ''


#    for jj in xrange( n_band ):
#        print jj
#        for kk in xrange( N_TYPE ):
#            print np.average( X_albd_kj_stack[kk][jj] ), np.sqrt( np.var(X_albd_kj_stack[kk][jj]) ),
#        print ''
#
#    print ''
#    print ''
 #
#    for ll in xrange( n_slice ):
#        print ll,
#        for kk in xrange( N_TYPE ):
#            print np.average( X_area_lk_stack[ll][kk] ), np.sqrt( np.var( X_area_lk_stack[ll][kk] ) ),
#        print ''

    sys.exit()

    print ''
    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")
    print 'PLOTTING ALBEDO PROBABILITY MAP...'
    myrange = np.tile( [0., 1.,], ( N_TYPE*n_band, 1 ) )
    fig = corner.corner( X_array[:,:N_TYPE*n_band], labels=range(N_TYPE*n_band), truths=best_fit[:N_TYPE*n_band], range=myrange, bins=100 )
    fig.savefig(INFILE+"_corner_albd.png")

    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")

    print ''
    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")
    print 'PLOTTING AREA FRACTION PROBABILITY MAP...'
    myrange = np.tile( [0., 1.,], ( N_SLICE, 1 ) )
    fig = corner.corner( X_array[:,N_TYPE*n_band:N_TYPE*n_band+N_SLICE], labels=range(N_SLICE), truths=best_fit[N_TYPE*n_band:N_TYPE*n_band+N_SLICE], range=myrange, bins=100 )
    fig.savefig(INFILE+"_corner_area1.png")

    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")

    print ''
    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")
    print 'PLOTTING AREA FRACTION PROBABILITY MAP...'
    myrange = np.tile( [0., 1.,], ( N_SLICE, 1 ) )
    fig = corner.corner( X_array[:,N_TYPE*n_band+N_SLICE:N_TYPE*n_band+2*N_SLICE], labels=range(N_SLICE), truths=best_fit[N_TYPE*n_band+N_SLICE:N_TYPE*n_band+2*N_SLICE], range=myrange, bins=100 )
    fig.savefig(INFILE+"_corner_area2.png")

    print 'X_array.shape', X_array[:,N_TYPE*n_band+2*N_SLICE:].shape
    print X_array[:,N_TYPE*n_band+2*N_SLICE:]

    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")

    print ''
    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")
    print 'PLOTTING AREA FRACTION PROBABILITY MAP...'
    myrange = np.tile( [0., 1.,], ( N_SLICE, 1 ) )
    print 'myrange', myrange
    print 'range(N_SLICE)', range(N_SLICE)
    fig = corner.corner( X_array[:,N_TYPE*n_band+2*N_SLICE:], labels=range(N_SLICE), truths=best_fit[N_TYPE*n_band+2*N_SLICE:], range=myrange, bins=100 )
    fig.savefig(INFILE+"_corner_area3.png")

    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")

