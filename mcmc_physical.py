import numpy as np
import healpy as hp
import emcee
from scipy.optimize import minimize
import sys, getopt
import corner
import datetime
import multiprocessing
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
import h5py
import pdb

from colorpy import colormodels, ciexyz

from reparameterize import transform_Y2X
from map_utils import save2hdf5

mpl.rc('font', family='Times New Roman')
mpl.rcParams['font.size'] = 25.0

DIR = "mcmc_output/"
EYECOLORS = False
EPOXI = True

# Specify burn-in index for corner plot
DEFAULT_BURN_INDEX = 0

#---------------------------------------------------
def decomposeX(x,n_band,n_slice,n_type):
    alb = x[0:n_band * n_type].reshape((n_type,n_band))
    area = x[n_band * n_type:].reshape((n_slice , n_type))
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

    xarea = np.arange(n_slice)
    xalb = np.arange(n_band)

    if n_slice == n_times:
        ax0.set_xlabel("Time [hrs]")
        ax0.set_xlim([np.min(xarea)-0.05, np.max(xarea)+0.05])
    else:
        ax0.set_xlabel("Slice Longitude [deg]")
        xarea = np.array([-180. + (360. / n_slice) * (i + 0.5) for i in range(n_slice)])
        ax0.set_xlim([-185, 185])
        ax0.set_xticks([-180, -90, 0, 90, 180])

    if EPOXI:
        epoxi_bands = np.loadtxt("data/EPOXI_band")
        wl = epoxi_bands[:,1]
        xalb = wl
        ax1.set_xlabel("Wavelength [nm]")
        ax1.set_xlim([300,1000])
    else:
        ax1.set_xlabel("Band")
        ax1.set_xlim([np.min(xalb)-0.05, np.max(xalb)+0.05])

    if EYECOLORS:
        epoxi_bands = np.loadtxt("data/EPOXI_band")
        wl = epoxi_bands[:,1]
        c = [convolve_with_eye(wl, med_alb[i,:]) for i in range(N_TYPE)]
    else:
        c = ["purple", "orange", "green", "lightblue"]

    for i in range(N_TYPE):
        ax0.plot(xarea, med_area[:,i], "o-", label="Surface %i" %(i+1), color=c[i])
        ax0.fill_between(xarea, med_area[:,i] - std_area[:,i], med_area[:,i] + std_area[:,i], alpha=0.3, color=c[i])
        ax1.plot(xalb, med_alb[i,:], "o-", color=c[i])
        ax1.fill_between(xalb, med_alb[i,:] - std_alb[i,:], med_alb[i,:] + std_alb[i,:], alpha=0.3 ,color=c[i])

    ax0.set_ylim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])

    leg=ax0.legend(loc=0, fontsize=14)
    leg.get_frame().set_alpha(0.0)

    fig.tight_layout()

    fig.savefig(directory+"xmed_std.pdf")

def plot_sampling(x, directory=""):

    ALPHA = 0.05

    print "Plotting %i Samples..." %len(x)

    fig = plt.figure(figsize=(16,8))
    gs = gridspec.GridSpec(1,2)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax0.set_ylabel("Area Fraction")
    ax0.set_xlabel("Slice #")
    ax1.set_ylabel("Albedo")

    xarea = np.arange(n_slice)
    xalb = np.arange(n_band)

    if n_slice == n_times:
        ax0.set_xlabel("Time [hrs]")
        ax0.set_xlim([np.min(xarea)-0.05, np.max(xarea)+0.05])
    else:
        ax0.set_xlabel("Slice Longitude [deg]")
        xarea = np.array([-180. + (360. / n_slice) * (i + 0.5) for i in range(n_slice)])
        ax0.set_xlim([-185, 185])
        ax0.set_xticks([-180, -90, 0, 90, 180])

    if EPOXI:
        epoxi_bands = np.loadtxt("data/EPOXI_band")
        wl = epoxi_bands[:,1]
        xalb = wl
        ax1.set_xlabel("Wavelength [nm]")
        ax1.set_xlim([300,1000])
    else:
        ax1.set_xlabel("Band")
        ax1.set_xlim([np.min(xalb)-0.05, np.max(xalb)+0.05])

    if EYECOLORS:
        epoxi_bands = np.loadtxt("data/EPOXI_band")
        wl = epoxi_bands[:,1]
        c = [convolve_with_eye(wl, med_alb[i,:]) for i in range(N_TYPE)]
    else:
        c = ["purple", "orange", "green", "lightblue"]

    for s in range(len(x)):
        # Decompose x vector into albedo and area arrays
        alb, area = decomposeX(x[s], n_band, n_slice, N_TYPE)

        for i in range(N_TYPE):
            if s == 0:
                ax0.plot(0, 0, "-", label="Surface %i" %(i+1), color=c[i], alpha=1.0)
            ax0.plot(xarea, area[:,i], "-", color=c[i], alpha=ALPHA)
            ax1.plot(xalb, alb[i,:], "-", color=c[i], alpha=ALPHA)

    ax0.set_ylim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])

    leg=ax0.legend(loc=0, fontsize=16)
    leg.get_frame().set_alpha(0.0)

    fig.tight_layout()

    fig.savefig(directory+"xsamples.pdf")

def convolve_with_eye(wl, spectrum):
    # Construct 2d array for ColorPy
    spec = np.vstack([wl, spectrum]).T
    # Call ColorPy modules to get irgb string
    rgb_eye = colormodels.irgb_string_from_rgb (
        colormodels.rgb_from_xyz (ciexyz.xyz_from_spectrum (spec)))
    return rgb_eye

def plot_reg1(samples):
    reg == 'Tikhonov'
    par = 'sigma', samples[-1]
    fig = plt.figure(figsize=(16,8))
    gs = gridspec.GridSpec(1,2)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax0.set_ylabel("Area Fraction")
    ax0.set_xlabel("Slice #")
    ax1.set_ylabel("Albedo")



#===================================================
if __name__ == "__main__":

    ###### Read command line args ######
    myopts, args = getopt.getopt(sys.argv[1:],"d:b:")
    run = ""
    iburn = DEFAULT_BURN_INDEX
    for o, a in myopts:
        # o == option
        # a == argument passed to the o
        if o == '-d':
            # Get MCMC directory timestamp name
            run=a
        elif o == "-b":
            # Get burn in index
            iburn = int(a)
        else:
            pass
    # Exit if no run directory provided
    if run == "":
        print("Please specify run directory using -d: \n e.g. >python mcmc_physical.py -d 2016-07-13--11-59")
        sys.exit()
    # Check for flag to convolve albedos with eye for plot colors
    if "eyecolors" in str(sys.argv):
        EYECOLORS = True
    else:
        EYECOLORS = False
    # Check for epoxi flag for wavelength labeling
    if "epoxi" in str(sys.argv):
        EPOXI = True
    else:
        EPOXI = False
    ##################################

    print "Burn-in index:", iburn

    MCMC_DIR = DIR + run + "/"

    # Load MCMC samples
    try:
        # Open the file stream
        f = h5py.File(MCMC_DIR+"samurai_out.hdf5", 'r+')
    except IOError:
        print "Run directory does not exist! Check -d argument."
        sys.exit()

    # Extract info from HDF5 file
    samples=f["samples"]
    assert iburn < samples.shape[1]
    N_TYPE = samples.attrs["N_TYPE"]
    n_slice = samples.attrs["N_SLICE"]
    p0 = f["p0"]
    X_names = samples.attrs["X_names"]
    Y_names = samples.attrs["Y_names"]
    nwalkers = samples.shape[0]
    nsteps = samples.shape[1]
    nparam = samples.shape[2]
    # Unpack Data
    Obs_ij = f["Obs_ij"]
    n_times = len(Obs_ij)
    n_band = len(Obs_ij[0])
    N_REGPARAM = samples.attrs["N_REGPARAM"]

    # Compute slice longitude
    #slice_longitude = np.array([-180. + (360. / n_slice) * (i + 0.5) for i in range(n_slice)])

    NAME_YSAM = "samples"
    NAME_XSAM = "physical_samples"

    # If the xsamples are already in the hdf5 file
    if NAME_XSAM in f.keys():
        # load physical samples
        xs = f["physical_samples"]
        print NAME_XSAM + " loaded from file!"
        if (xs.attrs["iburn"] == iburn) and (int(np.sum(xs[0,:])) != 0):
            # This is the exact same file or it has been loaded with 0's
            rerun = False
        else:
            # Must re-run xsamples with new burnin, overwrite
            print "Different burn-in index here. Must reflatten and convert..."
            rerun = True

    # If the xsamples are not in the hdf5 file,
    # or if they need to be re-run
    if NAME_XSAM not in f.keys() or rerun:

        # Determine shape of new dataset
        nxparam = len(transform_Y2X(samples[0,0,:], N_TYPE, n_band, n_slice, flatten=True))
        new_shape = (nwalkers*(nsteps-iburn), nxparam)

        # Construct attrs dictionary
        adic = {"iburn" : iburn}

        # Delete existing dataset if it already exists
        if NAME_XSAM in f.keys():
            del f[NAME_XSAM]

        # Flatten chains
        print "Flattening chains beyond burn-in (slow, especially if low burn-in index)..."
        flat_samples = samples[:,iburn:,:].reshape((-1, nparam))

        # Different approach if there are regularization params
        if (N_REGPARAM > 0):
            print "Filling xsample dataset..."
            sys.stdout.flush()
            # Loop over walkers
            # Exclude regularization parameters from albedo, area samples
            xsam = np.array([transform_Y2X(flat_samples[i,:-1*N_REGPARAM],
                            N_TYPE, n_band, n_slice, flatten=True)
                            for i in range(len(flat_samples))]
                            )
        else:
            # Use all parameters
            xsam = np.array([transform_Y2X(flat_samples[i], N_TYPE, n_band,
                            n_slice, flatten=True)
                            for i in range(len(flat_samples))]
                            )

        # Create new dataset in existing hdf5 file
        xs = f.create_dataset(NAME_XSAM, data=xsam, compression='lzf')
        # Add attributes to dataset
        for key, value in adic.iteritems(): xs.attrs[key] = value

    if "sample" in str(sys.argv):
        N_SAMP = 1000
        rand_sam = xs[np.random.randint(len(xs), size=N_SAMP),:]
        plot_sampling(rand_sam, directory=MCMC_DIR)


    if "median" in str(sys.argv):

        print "Computing Median Parameters..."

        # Find median & standard deviation
        xmed = np.median(xs, axis=0)
        xstd = np.std(xs, axis=0)

        # Decompose into useful 2d arrays
        med_alb, med_area = decomposeX(xmed, n_band, n_slice, N_TYPE)
        std_alb, std_area = decomposeX(xstd, n_band, n_slice, N_TYPE)

        print "Median:", xmed
        print "Std:", xstd

        # Plot median
        plot_median(med_alb, std_alb, med_area, std_area, directory=MCMC_DIR)

        # Save median results
        np.savetxt(MCMC_DIR+"albedo_median.txt", np.vstack([med_alb, std_alb]).T)
        np.savetxt(MCMC_DIR+"area_median.txt", np.vstack([med_area.T, std_area.T]).T)
        print "Saved:", "median_results.txt"

    if "corner" in str(sys.argv):
        print "Making Physical Corner Plot..."

        # Make corner plot
        fig = corner.corner(xs, plot_datapoints=True, plot_contours=False, plot_density=False,
            labels=X_names, show_titles=True)
        fig.savefig(MCMC_DIR+"xcorner.png")


    # Close HDF5 file stream
    f.close()

    sys.exit()
