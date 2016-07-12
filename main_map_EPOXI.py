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

#n_slice = 4
N_TYPE  = 3
N_SLICE = 9

MONTH = 'June'
SIGMA_Y  = 3.0
NOISELEVEL = 0.01

NUM_MCMC = 10
NUM_MCMC_BURNIN = 100
NCPU = multiprocessing.cpu_count()

FLAG_REG_AREA = False
FLAG_REG_ALBD = False

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
    LAT_S = 280.977
    LON_S = 22.531
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


#===================================================
 # basic functions
#=============================================== ====

#---------------------------------------------------
def lnprob(Y_array, *args):
    """
    Misfit-function to be minimized
    """

    Obs_ij, Obsnoise_ij, Kernel_il, n_param, flip, verbose  = args
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
    ln_prior_albd = prior.get_ln_prior_albd( Y_albd_kj )

    # flat prior for area fraction
    Y_area_lk = Y_array[N_TYPE*n_band:].reshape([N_SLICE, N_TYPE-1])
    ln_prior_area = prior.get_ln_prior_area( Y_area_lk, X_area_lk[:,:-1] )

    if verbose :
        print 'chi2', chi2 - ln_prior_albd - ln_prior_area, chi2, ln_prior_albd, ln_prior_area
        print 'chi2/d.o.f.', chi2 / (len(Y_array)*1.-1.), len(Y_array)

    # regularization term for area fraction
    if FLAG_REG_AREA :
        if FLAG_REG_ALBD :
            origin = -4
        else:
            origin = -2
        regterm_area = prior.regularize_area( x_area_lk, Y_array[origin], Y_array[origin+1], periodic=True)

    else:
        regterm_area = 0.

    # refularization term for albedo
    if FLAG_REG_ALBD :
        Sigma_jj      = sigma(Y_array[-2], Y_array[-1], n_band)
    #    print "Sigma_jj", Sigma_jj
        inv_Sigma_jj  = np.linalg.inv(Sigma_jj)
        addterm_k     = np.diag(np.dot(np.dot(X_albd_kj, inv_Sigma_jj), X_albd_kj.T))
        regterm_albd = np.sum(addterm_k) + np.log(np.linalg.det(Sigma_jj))
    else:
        regterm_albd = 0.

    r_y  = 1.0/SIGMA_Y**2*np.dot(Y_array-0.5, Y_array-0.5)

    answer = - chi2 + ln_prior_albd + ln_prior_area + regterm_area + regterm_albd
    if flip :
        return -1. * answer
    else :
        return answer


#---------------------------------------------------
def transform_Y2X(Y_array, n_band):

    Y_array = np.maximum(Y_array, -10)
    Y_array = np.minimum(Y_array, 10)

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
    """
    Re-parameterization for convenience -- now Y can take on any value.
    """

#    print "X_area_lk", X_area_lk
    Y_albd_kj = np.log(X_albd_kj) - np.log(1.-X_albd_kj)
#    print "Y_albd_kj", Y_albd_kj
    Y_area_lk = np.zeros([N_SLICE, N_TYPE-1])
    print 'X_area_lk', X_area_lk.shape
    print 'Y_area_lk', Y_area_lk.shape
    for kk in xrange(N_TYPE-1):
#        print np.sum(X_area_lk[:,:kk+1],axis=1)
        Y_area_lk[:,kk] = np.log(X_area_lk[:,kk]) - np.log(1.-np.sum(X_area_lk[:,:kk+1], axis=1))
#    print "Y_area_lk", Y_area_lk
    return np.concatenate([Y_albd_kj.flatten(), Y_area_lk.flatten()])



#===================================================
if __name__ == "__main__":

    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")

    # input data
    Obs_ij = np.loadtxt(INFILE)

    n_band = len(Obs_ij[0])

    Obsnoise_ij = ( NOISELEVEL * Obs_ij )

    # set kernel
    param_geometry = ( LAT_O, LON_O, LAT_S, LON_S, OMEGA )
    Kernel_il = geometry.kernel( Time_i, N_SLICE, N_SIDE, param_geometry )
    for ii in xrange( len( Kernel_il ) ):
        for ll in xrange( len( Kernel_il[0] ) ):
            print Kernel_il[ii][ll],
        print '' 

    X0_albd_kj = 0.3+np.zeros([N_TYPE, n_band])
    X0_area_lk = 0.1+np.zeros([N_SLICE, N_TYPE-1])

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
#    Y0_1<list = [Y0_albd_kj, Y0_area_lk]
#    print "Y0_array", Y0_array

    if (n_param > 0):
        X_albd_kj, X_area_lk =  transform_Y2X(Y0_array[:-1*n_param], n_band)
    else:
        X_albd_kj, X_area_lk =  transform_Y2X(Y0_array, n_band)

#    print "X_area_lk", X_area_lk
#    print "X_albd_kj", X_albd_kj

    # minimize
    print "finding best-fit values..."
    data = (Obs_ij, Obsnoise_ij, Kernel_il, n_param, True, False)
    output = minimize(lnprob, Y0_array, args=data, method="Nelder-Mead")

    print "best-fit", output["x"]

    data = (Obs_ij, Obsnoise_ij, Kernel_il, n_param, True, False)
    lnprob_bestfit = lnprob( output['x'], *data )

#    BIC = 2.0 * lnprob_bestfit + len( output['x'] ) * np.log( len(Obs_ij.flatten()) )
#    print 'BIC: ', BIC

    X_albd_kj, X_area_lk =  transform_Y2X(output["x"], n_band)
    np.savetxt("X_area_lk", X_area_lk)
    np.savetxt("X_albd_kj_T", X_albd_kj.T)
    bestfit = np.r_[ X_albd_kj.flatten(), X_area_lk.T.flatten() ]


    print "MCMC until burn-in..."
    n_dim = len(Y0_array)
    n_walkers = 2*n_dim**2
    data = (Obs_ij, Obsnoise_ij, Kernel_il, n_param, False, False)
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
    fig = corner.corner( X_array[:,:N_TYPE*n_band], labels=range(N_TYPE*n_band), truths=bestfit[:N_TYPE*n_band], range=myrange, bins=100 )
    fig.savefig(INFILE+"_corner_albd.png")

    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")

    print ''
    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")
    print 'PLOTTING AREA FRACTION PROBABILITY MAP...'
    myrange = np.tile( [0., 1.,], ( N_SLICE, 1 ) )
    fig = corner.corner( X_array[:,N_TYPE*n_band:N_TYPE*n_band+N_SLICE], labels=range(N_SLICE), truths=bestfit[N_TYPE*n_band:N_TYPE*n_band+N_SLICE], range=myrange, bins=100 )
    fig.savefig(INFILE+"_corner_area1.png")

    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")

    print ''
    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")
    print 'PLOTTING AREA FRACTION PROBABILITY MAP...'
    myrange = np.tile( [0., 1.,], ( N_SLICE, 1 ) )
    fig = corner.corner( X_array[:,N_TYPE*n_band+N_SLICE:N_TYPE*n_band+2*N_SLICE], labels=range(N_SLICE), truths=bestfit[N_TYPE*n_band+N_SLICE:N_TYPE*n_band+2*N_SLICE], range=myrange, bins=100 )
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
    fig = corner.corner( X_array[:,N_TYPE*n_band+2*N_SLICE:], labels=range(N_SLICE), truths=bestfit[N_TYPE*n_band+2*N_SLICE:], range=myrange, bins=100 )
    fig.savefig(INFILE+"_corner_area3.png")

    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")

