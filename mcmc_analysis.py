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


# Specify directory of run to analyze
MCMC_DIR = "mcmc_output/2016-07-11--15-08/"

def estimate_burnin(samples):
    # chain[n_walkers, steps, n_dim]
    # Determine time of burn-in by calculating first time median is crossed
    # Algorithm by Eric Agol 2016
    #
    # Parameters
    # ----------
    # par_mcmc : Array{Float64,3}
    #    Array containing history of all walkers for each parameter
    # nwalkers : Int
    #    Number of walkers used in MCMC
    # nparam : Int
    #    Number of params fit for in MCMC
    # nsteps : Int
    #    Number of MCMC steps
    #
    # Returns
    # -------
    # iburn : Int
    #    Index corresponding to the step where the burn in approximately ended

    nwalkers = samples.shape[0]
    nsteps = samples.shape[1]
    nparam = samples.shape[2]

    iburn = 0

    # Calculate median for each parameter
    med_params = np.array([np.median(samples[:,:,i]) for i in range(nparam)])

    # For each parameter
    for i in range(nparam):
        med_param = np.median(samples[:,:,i])
        # For each walker
        for j in range(nwalkers):
            istep=2
            while ((samples[j,istep,i] > med_param) == (samples[j,istep-1,i] > med_param)) & (istep < nsteps):
                istep=istep+1
            if istep >= iburn:
                iburn = istep

    return iburn

def plot_trace():

    return

def plot_corner():
    return

#===================================================
if __name__ == "__main__":

    # Load in MCMC save files
    temp = np.load(MCMC_DIR+"mcmc_results.npz")
    # Extract info
    samples=temp["samples"]
    original_samples = temp["original_samples"]
    X_albd_kj_stack=temp["X_albd_kj_stack"]
    X_area_lk_stack=temp["X_area_lk_stack"]
    X_albd_error=temp["X_albd_error"]
    X_area_error=temp["X_area_error"]

    print estimate_burnin(original_samples)

    pdb.set_trace()

    X_array = np.zeros( [ len( samples ), N_TYPE*n_band + n_slice*N_TYPE ] )

    print 'accumulation...'
    print len( samples )


    X_albd_kj, X_area_lk =  transform_Y2X( samples[0], n_band)
    X_array[0] = np.r_[ X_albd_kj.flatten(), X_area_lk.T.flatten() ]

    X_albd_kj_stack = X_albd_kj
    X_area_lk_stack = X_area_lk

    for ii in xrange( 1, len( samples ) ):
        if ii % 1000 == 0 :
            print ii

        X_albd_kj, X_area_lk =  transform_Y2X( samples[ii], n_band)
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

    for ll in xrange( n_slice ):
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


    # Save results
    print "Saving:", run_dir+"mcmc_results.npz"
    np.savez(run_dir+"mcmc_results.npz", samples, X_albd_kj_stack, X_area_lk_stack, X_albd_error, X_area_error)

    sys.exit()

    print ''
    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")
    print 'PLOTTING ALBEDO PROBABILITY MAP...'
    myrange = np.tile( [0., 1.,], ( N_TYPE*n_band, 1 ) )
    fig = corner.corner( X_array[:,:N_TYPE*n_band], labels=range(N_TYPE*n_band), truths=bestfit[:N_TYPE*n_band], range=myrange, bins=100 )
    fig.savefig(INFILE+"_corner_albd.png")

    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")

    print ''
    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")
    print 'PLOTTING AREA FRACTION PROBABILITY MAP...'
    myrange = np.tile( [0., 1.,], ( n_slice, 1 ) )
    fig = corner.corner( X_array[:,N_TYPE*n_band:N_TYPE*n_band+n_slice], labels=range(n_slice), truths=bestfit[N_TYPE*n_band:N_TYPE*n_band+n_slice], range=myrange, bins=100 )
    fig.savefig(INFILE+"_corner_area1.png")

    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")

    print ''
    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")
    print 'PLOTTING AREA FRACTION PROBABILITY MAP...'
    myrange = np.tile( [0., 1.,], ( n_slice, 1 ) )
    fig = corner.corner( X_array[:,N_TYPE*n_band+n_slice:N_TYPE*n_band+2*n_slice], labels=range(n_slice), truths=bestfit[N_TYPE*n_band+n_slice:N_TYPE*n_band+2*n_slice], range=myrange, bins=100 )
    fig.savefig(INFILE+"_corner_area2.png")

    print 'X_array.shape', X_array[:,N_TYPE*n_band+2*n_slice:].shape
    print X_array[:,N_TYPE*n_band+2*n_slice:]

    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")

    print ''
    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")
    print 'PLOTTING AREA FRACTION PROBABILITY MAP...'
    myrange = np.tile( [0., 1.,], ( n_slice, 1 ) )
    print 'myrange', myrange
    print 'range(n_slice)', range(n_slice)
    fig = corner.corner( X_array[:,N_TYPE*n_band+2*n_slice:], labels=range(n_slice), truths=bestfit[N_TYPE*n_band+2*n_slice:], range=myrange, bins=100 )
    fig.savefig(INFILE+"_corner_area3.png")

    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")
