import numpy as np
import healpy as hp
import emcee
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
import pdb
mpl.rc('font', family='Times New Roman')
mpl.rcParams['font.size'] = 25.0
from scipy.optimize import minimize
import sys
import corner
import datetime
import multiprocessing
import os
import pdb


# Specify directory of run to analyze
MCMC_DIR = "mcmc_output/2016-07-12--16-57/"

# Specify burn-in index for corner plot
BURN_INDEX = 0

def estimate_burnin1(samples):
    # Determine time of burn-in by calculating first time median is crossed
    # Algorithm by Eric Agol 2016

    # Calculate the median for each parameter across all walkers and steps
    med_params = np.array([np.median(samples[:,:,i]) for i in range(nparam)])

    # Subtract off the median
    diff = samples - med_params

    # Determine where the sign changes occur
    asign = np.sign(diff)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)

    # For each walker determine index where the first sign change occurs
    first_median_crossing = np.argmax(signchange>0, axis=1)

    # Now return the index of the last walker to cross its median
    return np.amax(first_median_crossing)

def plot_trace(samples, directory="", X_names=None):

    print "Plotting Trace..."

    nwalkers = samples.shape[0]
    nsteps = samples.shape[1]
    nparam = samples.shape[2]

    # Flatten chains for histogram
    samples_flat = samples[:, :, :].reshape((-1, nparam))

    # Loop over all parameters making trace plots
    for i in range(nparam):
        if X_names is None:
            pname = ""
        else:
            pname = X_names[i]
        fig = plt.figure(figsize=(13,5))
        gs = gridspec.GridSpec(1,2, width_ratios=(1,.3))
        ax0 = plt.subplot(gs[0])
        ax0.plot(samples[:,:,i].T, lw=0.5)
        ax0.set_xlabel("Iteration")
        ax0.set_ylabel(pname+" Value")
        ax1 = plt.subplot(gs[1], sharey=ax0)
        bins = np.linspace(np.min(samples_flat[:,i]), np.max(samples_flat[:,i]), 25, endpoint=True)
        h = ax1.hist(samples_flat[:,i], bins, orientation='horizontal', color="k", alpha=0.5)
        ax1.set_xticks([])
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        plt.setp(ax0.get_xticklabels(), fontsize=18, rotation=45)
        plt.setp(ax0.get_yticklabels(), fontsize=18, rotation=45)
        plt.setp(ax1.get_xticklabels(), fontsize=18, rotation=45)
        plt.setp(ax1.get_yticklabels(), fontsize=18, rotation=45)
        fig.subplots_adjust(wspace=0)
        fig.savefig(directory+"trace"+str(i)+".pdf")
    return

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

    nwalkers = samples.shape[0]
    nsteps = samples.shape[1]
    nparam = samples.shape[2]

    # Flatten chains
    samples_flat = samples[:,BURN_INDEX:,:].reshape((-1, nparam))

    if "trace" in str(sys.argv):

        # Create directory for trace plots
        trace_dir = MCMC_DIR+"trace_plots/"
        try:
            os.mkdir(trace_dir)
            print "Created directory:", trace_dir
        except OSError:
            print trace_dir, "already exists."

        # Make trace plots
        plot_trace(samples, X_names=X_names, directory=trace_dir)

    if "corner" in str(sys.argv):
        print "Making Corner Plot..."

        # Make corner plot
        fig = corner.corner(samples_flat, plot_datapoints=True, plot_contours=False, plot_density=False, labels=X_names)
        fig.savefig(MCMC_DIR+"xcorner.png")
