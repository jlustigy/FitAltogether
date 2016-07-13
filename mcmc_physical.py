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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
import pdb
mpl.rc('font', family='Times New Roman')
mpl.rcParams['font.size'] = 25.0

# Specify directory of run to analyze
MCMC_DIR = "mcmc_output/2016-07-12--15-04/"

# Specify burn-in index for corner plot
BURN_INDEX = 15000

#---------------------------------------------------
#---------------------------------------------------
def transform_Y2X(Y_array, n_band, n_slice):

    Y_array = np.maximum(Y_array, -10)
    Y_array = np.minimum(Y_array, 10)

    # 'albedo' part
    Y_albd_kj_flat = Y_array[0:N_TYPE*n_band]
    X_albd_kj_flat = np.exp( Y_albd_kj_flat )/( 1 + np.exp( Y_albd_kj_flat ) )

    # 'area fraction' part
    Y_area_lk = Y_array[N_TYPE*n_band:].reshape([n_slice, N_TYPE-1])
    expY_area_lk = np.exp( Y_area_lk )
    expYY_area_lk = 1./( 1 + expY_area_lk )
    cumprodY_area_lk = np.cumprod( expYY_area_lk, axis=1 )
    X_area_lk = expY_area_lk * cumprodY_area_lk
    X_area_lk = np.c_[X_area_lk, 1. - np.sum( X_area_lk, axis=1 )]
    X_area_lk_flat = X_area_lk.flatten()

    X_array = np.r_[ X_albd_kj_flat, X_area_lk_flat ]
    return X_array

def decomposeX(x,n_band,n_slice,n_type):
    alb = x[0:n_band * n_type].reshape((n_type,n_band))
    area = xmed[n_band * n_type:].reshape((n_slice , n_type))
    return alb, area

def plot_median(med_alb, std_alb, med_area, std_area, directory=""):

    print "Plotting Median, Std..."

    fig = plt.figure(figsize=(16,8))
    gs = gridspec.GridSpec(1,2)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax0.set_ylabel("Area Fraction")
    ax0.set_xlabel("Slice #")
    ax1.set_ylabel("Albedo")
    ax1.set_xlabel("Band")

    c = ["purple", "orange", "green", "lightblue"]

    xarea = np.arange(n_slice)
    xalb = np.arange(n_band)

    for i in range(N_TYPE):
        ax0.plot(xarea, med_area[:,i], "o-", label="Surface %i" %(i+1), color=c[i])
        ax0.fill_between(xarea, med_area[:,i] - std_area[:,i], med_area[:,i] + std_area[:,i], alpha=0.3, color=c[i])
        ax1.plot(xalb, med_alb[i,:], "o-", color=c[i])
        ax1.fill_between(xalb, med_alb[i,:] - std_alb[i,:], med_alb[i,:] + std_alb[i,:], alpha=0.3 ,color=c[i])

    ax0.set_ylim([-0.05, 1.05])
    ax0.set_xlim([-0.05, 3.05])

    ax1.set_xlim([-0.05, 2.05])

    leg=ax0.legend(loc=0, fontsize=14)
    leg.get_frame().set_alpha(0.0)

    fig.tight_layout()

    fig.savefig(directory+"xmed_std.pdf")

#===================================================
if __name__ == "__main__":

    # Load MCMC samples from numpy archive
    temp = np.load(MCMC_DIR+"mcmc_samples.npz")

    # Extract info from numpy archive
    samples=temp["samples"]
    data = temp["data"]
    N_TYPE = temp["N_TYPE"]
    p0 = temp["p0"]
    X_names = temp["X_names"]

    # MCMC dimensions
    nwalkers = samples.shape[0]
    nsteps = samples.shape[1]
    nparam = samples.shape[2]

    # Data dimensions
    Obs_ij = data[0]
    n_slice = len(Obs_ij)
    n_band = len(Obs_ij[0])

    # Flatten chains that go beyond burn-in (aka sampling the posterior)
    samples = samples[:,BURN_INDEX:,:].reshape((-1, nparam))

    # Transform all samples to physical units
    xsam = np.array([transform_Y2X(samples[i], n_band, n_slice) for i in range(len(samples))])

    # Find median & standard deviation
    xmed = np.median(xsam, axis=0)
    xstd = np.std(xsam, axis=0)

    # Decompose into useful 2d arrays
    med_alb, med_area = decomposeX(xmed, n_band, n_slice, N_TYPE)
    std_alb, std_area = decomposeX(xstd, n_band, n_slice, N_TYPE)

    print "Median:", xmed
    print "Std:", xstd

    # Plot median
    plot_median(med_alb, std_alb, med_area, std_area, directory=MCMC_DIR)

    sys.exit()

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

    # Save results
    print "Saving:", MCMC_DIR+"mcmc_physical.npz"
    np.savez(MCMC_DIR+"mcmc_physical.npz", \
        X_albd_kj_stack=X_albd_kj_stack, X_area_lk_stack=X_area_lk_stack, \
        X_albd_error=X_albd_error, X_area_error=X_area_error)

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
