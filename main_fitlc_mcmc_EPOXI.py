import numpy as np
import healpy as hp
import emcee
import matplotlib.pyplot as pl
from scipy.optimize import minimize
import sys
import corner
import datetime
import multiprocessing
import geometry
import prior
from reparameterize import *

# March 2008
#LAT_S = -0.5857506  # sub-solar latitude
#LON_S = 267.6066184  # sub-solar longitude
#LAT_O = 1.6808370  # sub-observer longitude
#LON_O = 210.1242232 # sub-observer longitude

#NUM_MCMC = 2
#NUM_MCMC_BURNIN = 1

NUM_MCMC = 10
NUM_MCMC_BURNIN = 1000

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
#INFILE = "data/raddata_1_norm"
#INFILE = 'mockdata/mock_simple_1_scattered0.01_data_with_noise'
INFILE = "mockdata/mock_simple_1_data"

#===================================================
 # basic functions
#=============================================== ====

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
        X_albd_kj, X_area_lk = transform_Y2X( Y_array[:-1*n_param], N_TYPE, n_band, n_slice, flatten=False )
    else:
        X_albd_kj, X_area_lk = transform_Y2X( Y_array, N_TYPE, n_band, n_slice, flatten=False )

    # making matrix...
    Model_ij = np.dot(Kernel_il, np.dot(X_area_lk, X_albd_kj))
    Diff_ij = ( Obs_ij - Model_ij ) / Obsnoise_ij
    Chi2_i  = np.diag(np.dot( Diff_ij, Diff_ij.T ))
    chi2    = np.sum(Chi2_i)

    # flat prior for albedo
    Y_albd_kj = Y_array[0:N_TYPE*n_band].reshape([N_TYPE, n_band])
    ln_prior_albd = prior.get_ln_prior_albd( Y_albd_kj )

    # flat prior for area fraction
    Y_area_lk = Y_array[N_TYPE*n_band:].reshape([n_slice, N_TYPE-1])
    ln_prior_area = prior.get_ln_prior_area( Y_area_lk, X_area_lk[:,:-1] )

    if verbose :
        print ''
        print 'chi2', chi2 - ln_prior_albd - ln_prior_area, chi2, -1*ln_prior_albd, -1*ln_prior_area
        print 'chi2/d.o.f.', chi2 / (len(Y_array)*1.-1.), len(Y_array)
        print ''

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



#===================================================
if __name__ == "__main__":

    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")

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
    X0_area_lk = 0.1+np.zeros([n_slice, N_TYPE])

## albedo ( band x surface type )
#    X0_albd_kj = np.array( [[1.000000000000000056e-01, 9.000000000000000222e-01],
#                            [2.999999999999999889e-01, 5.000000000000000000e-01],
#                            [3.499999999999999778e-01, 5.999999999999999778e-01]]).T
#
#
## area fraction ( longitude slice x suface type )
#    X0_area_lk = np.array([[2.000000000000000111e-01, 8.000000000000000444e-01],
#                           [5.500000000000000444e-01, 4.499999999999999556e-01],
#                           [5.999999999999999778e-01, 4.000000000000000222e-01],
#                           [9.000000000000000222e-01, 9.999999999999997780e-02]])

#    X0_array = np.r_[ X0_albd_kj.flatten(), X0_area_lk.T[0].flatten() ]

#    X0_albd_kj[0,0:7] = np.array([0.35, 0.28, 0.28, 0.32, 0.40, 0.47, 0.35])
#    X0_albd_kj[1,0:7] = np.array([0.37, 0.28, 0.28, 0.30, 0.40, 0.48, 0.37])
#    X0_albd_kj[2,0:7] = np.array([0.67, 0.52, 0.40, 0.35, 0.33, 0.33, 0.32])
#    X0_albd_kj[0,0:7] = np.array([0.35, 0.28, 0.28, 0.32, 0.40, 0.47])
#    X0_albd_kj[1,0:7] = np.array([0.37, 0.28, 0.28, 0.30, 0.40, 0.48])
#    X0_albd_kj[2,0:7] = np.array([0.67, 0.52, 0.40, 0.35, 0.33, 0.33])
#    X0_area_lk[0,0] = np.array([0.3])
    Y0_array = transform_X2Y(X0_albd_kj, X0_area_lk)
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
        X_array =  transform_Y2X(Y0_array[:-1*n_param], N_TYPE, n_band, n_slice, flatten=True)
    else:
        X_array =  transform_Y2X(Y0_array, N_TYPE, n_band, n_slice, flatten=True)

#    print "X_area_lk", X_area_lk
#    print "X_albd_kj", X_albd_kj

    # minimize
    print "finding best-fit values..."
    data = (Obs_ij, Obsnoise_ij, Kernel_il, n_param, True, False)
    output = minimize(lnprob, Y0_array, args=data, method="Nelder-Mead")


    print "best-fit", output["x"]

    data = (Obs_ij, Obsnoise_ij, Kernel_il, n_param, True, True)
    lnprob_bestfit = lnprob( output['x'], *data )
    BIC = 2.0 * lnprob_bestfit + len( output['x'] ) * np.log( len(Obs_ij.flatten()) )
    print 'BIC: ', BIC

    X_albd_kj, X_area_lk = transform_Y2X(output["x"], N_TYPE, n_band, n_slice, flatten=False)
#    X_area_lk = transform_Y2X(output["x"], n_band, n_slice)
    np.savetxt("X_area_lk", X_area_lk)
    np.savetxt("X_albd_kj_T", X_albd_kj.T)
    bestfit = np.r_[ X_albd_kj.flatten(), X_area_lk.T.flatten() ]

    print "residuals", Obs_ij - np.dot( X_area_lk, X_albd_kj )

    print "MCMC until burn-in..."
    n_dim = len(Y0_array)
    n_walkers = 2*n_dim**2
    data = (Obs_ij, Obsnoise_ij, Kernel_il, n_param, False, False)
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob, args=data, threads=NCPU)
#    pos = 0.01*np.random.rand(n_dim * n_walkers).reshape((n_walkers, n_dim)) + output["x"]
    p0 = 0.01*np.random.rand(n_dim * n_walkers).reshape((n_walkers, n_dim)) + output["x"]
    pos, prob, state = sampler.run_mcmc( p0, NUM_MCMC_BURNIN )
    np.savetxt( 'tmp.txt', sampler.chain[0,:,1] )

    print "MCMC from burn-in..."
    sampler.reset()
    sampler.run_mcmc( pos, NUM_MCMC )
    samples = sampler.chain[:, :, :].reshape((-1, n_dim)) # trial x n_dim
    X_array = np.zeros( [ len( samples ), N_TYPE*n_band + n_slice*N_TYPE ] )

    print 'accumulation...'
    print len( samples )


    X_albd_kj, X_area_lk =  transform_Y2X( samples[0], N_TYPE, n_band, n_slice, flatten=False)
    X_array[0] = np.r_[ X_albd_kj.flatten(), X_area_lk.T.flatten() ]

    X_albd_kj_stack = X_albd_kj
    X_area_lk_stack = X_area_lk

    for ii in xrange( 1, len( samples ) ):
        if ii % 1000 == 0 :
            print ii

        X_albd_kj, X_area_lk =  transform_Y2X( samples[ii], N_TYPE, n_band, n_slice, flatten=False)
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
