# http://dan.iel.fm/emcee/current/user/quickstart/

import numpy as np
import emcee
import matplotlib.pyplot as pl

def lnprob(x, mu, icov):
    diff = x-mu
    return -np.dot(diff,np.dot(icov,diff))/2.0

nwalkers = 250
ndim = 5
means = np.random.rand(ndim)
cov = 0.5 - np.random.rand(ndim**2).reshape((ndim, ndim))
cov = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov = np.dot(cov,cov)
icov = np.linalg.inv(cov)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[means, icov])

p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()

sampler.run_mcmc(pos, 100)

print "means:", means
print "cov:"
for k in xrange(ndim) :
    print cov[k,k]

i = 0
pl.figure()
pl.hist(sampler.flatchain[:,i], 100, color="k", histtype="step")
pl.title("Dimension {0:d}".format(i))
pl.show()
