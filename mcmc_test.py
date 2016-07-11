# http://dan.iel.fm/emcee/current/user/quickstart/
import numpy as np
import emcee
import matplotlib.pyplot as pl
from scipy.optimize import minimize

INFILE = "mkdata.txt"
NWALKERS = 100
NDIM = 3
PARAM0 = np.array([10,10,10])
COV0  = 50 - np.random.rand(NDIM**2).reshape((NDIM, NDIM))
COV0  = np.triu(COV0)
COV0  += COV0.T - np.diag(COV0.diagonal())
COV0  = np.dot(COV0,COV0)
ICOV0 = np.linalg.inv(COV0)*0.01
print ICOV0
R0=1.0

def lnlikelihood_gaussian(param, data):
    """
    Define likelihood
    """
    x, obs, sigma = data.T
    theory = param[0] + param[1]*x + param[2]*x**2
#    print "th", theory, "obs", obs
    metric = (theory-obs)/sigma
#    print "metric", metric
    return np.dot(metric, metric)


def prior_gaussian(param, param0, icov0):
    """
    Define prior
    input : parameters
    """
    nparam = param - param0
    return np.dot(np.dot(icov0,nparam), nparam)


def prior_flat(param):
    return 1.0


def lnprob(param, *arg):
    """
    Log Probability
    """
    data, param0, icov0 = arg
#    print data
#    print lnlikelihood_gaussian(param, data)
    return lnlikelihood_gaussian(param, data)+prior_gaussian(param, param0, icov0)

#    return -np.dot(diff,np.dot(iCOV,diff))/2.0


##############################################################################
if __name__ == "__main__":


    data = np.loadtxt(INFILE)

    array_x0 = np.zeros(3)+10.0
    result = minimize(lnprob, array_x0, args=(data,PARAM0,ICOV0), method='SLSQP', options={'disp': True})
    print result['x']
    sampler = emcee.EnsembleSampler(NWALKERS, NDIM, lnprob, args=(data,PARAM0,ICOV0))
    pos = R0*np.random.rand(NDIM * NWALKERS).reshape((NWALKERS, NDIM))-0.5 + result['x']
    sampler.run_mcmc(pos, 200)

    i = 2
    print sampler.flatchain.shape
    np.savetxt("tmp.txt", sampler.flatchain[:,i])

    pl.figure()
    pl.hist(sampler.flatchain[:,i], 200, color="k", histtype="step", range=(-50,50))
    pl.title("Dimension {0:d}".format(i))
    pl.show()
