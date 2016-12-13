# Import standard libraries
import numpy as np
#import matplotlib.pyplot as plt
import sys, imp, os
import datetime
import multiprocessing
from scipy.optimize import minimize
from types import ModuleType, FunctionType, StringType

from pdb import set_trace as stop
# Import dependent modules
import healpy as hp
import emcee
#import corner
import h5py
# Import packages files
import geometry
import prior
import reparameterize
import likelihood
from map_utils import generate_tex_names, save2hdf5, calculate_walkers

__all__ = ["Mapper", "Data"]

# The location to *this* file
RELPATH = os.path.dirname(__file__)

################################################################################

def set_nreg(reg):
    """
    Set number of extra parameters due to regularization
    """
    if reg is not None:
        if reg == 'Tikhonov':
            N = 1
        elif reg == 'GP':
            N = 3
        elif reg == 'GP2':
            N = 2
        else:
            print("%s is not a valid regularization method. Using no regularization." %reg)
            N = 0
    else:
        N = 0
    return N

################################################################################

class Mapper(object):
    """
    """
    def __init__(self, model="map", data=None,
                 ntype=3, nsideseed=4, regularization=None, reg_area=False, reg_albd=False,
                 sigmay=3.0, noiselevel=0.01, Nmcmc=10000, Nmcmc_b=0, mcmc_seedamp=0.5,
                 hdf5_compression='lzf', nslice=9, period=None, lat_s=None, lon_s=None,
                 lat_o=None, lon_o=None, ncpu=None
                 ):
        """
        Samurai mapping object

        Parameters
        ----------

        Returns
        -------
        """
        self.model=model
        self.data=data
        self.ntype=ntype
        self.nsideseed=nsideseed
        self.regularization=regularization
        self.reg_area=reg_area
        self.reg_albd=reg_albd
        self.sigmay=sigmay
        self.noiselevel=noiselevel
        self.Nmcmc=Nmcmc
        self.Nmcmc_b=Nmcmc_b
        self.mcmc_seedamp=mcmc_seedamp
        self.hdf5_compression=hdf5_compression
        if self.ncpu is None:
            self.ncpu = multiprocessing.cpu_count()

        # Params unique to map model
        self.nslice=nslice
        self.lat_s=lat_s
        self.lon_s=lon_s
        self.lat_o=lat_o
        self.lon_o=lon_o
        self._period=period
        if self._period is None:
            self._omega = None
        else:
            self._omega= ( 2. * np.pi / self.period )
        self._regularization=regularization
        self._n_regparam=set_nreg(self.regularization)

    @classmethod
    def from_EPOXI_march(cls):

        lat_s = -0.581  # sub-solar latitude
        lon_s = 262.909  # sub-solar longitude
        lat_o = 1.678  # sub-observer latitude
        lon_o = 205.423 # sub-observer longitude
        infile = "data/raddata_1_norm"
        Time_i = np.arange(25)*1.

        # Return new class instance
        return cls(model="map", lat_s=lat_s, lon_s=lon_s, lat_o=lat_o, lon_o=lon_o,
                   infile=infile)


    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, value):
        self._period = value
        self._omega = ( 2. * np.pi / value )

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        self._omega = value

    #

    @property
    def regularization(self):
        return self._regularization

    @regularization.setter
    def regularization(self, value):
        self._regularization = value
        self._n_regparam = set_nreg(value)

    @property
    def n_regparam(self):
        return self._n_regparam

    @n_regparam.setter
    def n_regparam(self, value):
        self._n_regparam = value

    #
    def run_mcmc(self, verbose=True):
        """
        Run Mapper object with ``emcee`` MCMC code.
        """

        # print start time
        now = datetime.datetime.now()
        if verbose: print(now.strftime("%Y-%m-%d %H:%M:%S"))

        # Create directory for this run
        startstr = now.strftime("%Y-%m-%d--%H-%M")
        run_dir = os.path.join("mcmc_output", startstr)
        os.mkdir(run_dir)
        if verbose: print("Created directory:", run_dir)

        # Maybe pickle object instead?
        """
        # Save THIS file and the param file for reproducibility!
        thisfile = os.path.basename(__file__)
        paramfile = "map_EPOXI_params.py"
        newfile = os.path.join(run_dir, thisfile)
        commandString1 = "cp " + thisfile + " " + newfile
        commandString2 = "cp "+paramfile+" " + os.path.join(run_dir,paramfile)
        os.system(commandString1)
        os.system(commandString2)
        if verbose: print("Saved :", thisfile, " &", paramfile)
        """

        # Unpack class variables
        model = self.model
        Time_i = self.data.Time_i
        nslice = self.nslice
        ntype = self.ntype
        nregparam = self.n_regparam
        regularization = self.regularization
        lat_o = self.lat_o
        lon_o = self.lon_o
        lat_s = self.lat_s
        lon_s = self.lon_s
        omega = self.omega
        ncpu = self.ncpu
        num_mcmc = self.Nmcmc
        seed_amp  = self.mcmc_seedamp
        hdf5_compression = self.hdf5_compression
        waveband_centers = self.data.wlc_i
        waveband_widths = self.data.wlw_i

        # Input data
        Obs_ij = self.data.Obj_ij
        Obsnoise_ij = ( self.noiselevel * self.data.Obs_ij )
        nband = len(Obs_ij[0])

        # Calculate n_side
        nside = 2*2**self.nsideseed

        # Set geometric kernel depending on model
        if model == "map":
            param_geometry = ( lat_o, lon_o, lat_s, lon_s, omega )
            Kernel_il = geometry.kernel( Time_i, nslice, nside, param_geometry )
        elif model == "lightcurve":
            Kernel_il = np.identity( len(Obs_ij) )

        # Initialize the fitting parameters
        X0_albd_kj = 0.3+np.zeros([ntype, nband])
        X0_area_lk = 0.2+np.zeros([nslice, ntype])
        Y0_array = transform_X2Y(X0_albd_kj, X0_area_lk)
        if ( nregparam > 0 ) :
            Y0_array = np.append(Y0_array, np.array([10.]*nregparam) )
        n_dim = len(Y0_array)
        if verbose:
            print('Y0_array', Y0_array)
            print('# of parameters', n_dim)
            print('N_REGPARAM', nregparam)
        if (nregparam > 0):
            X_albd_kj, X_area_lk =  transform_Y2X(Y0_array[:-1*nregparam], ntype, nband, nslice)
        else:
            X_albd_kj, X_area_lk =  transform_Y2X(Y0_array, ntype, nband, nslice)

        # Create list of strings for Y & X parameter names
        Y_names, X_names = generate_tex_names(ntype, nband, nslice)

        ############ run minimization ############

        # minimize
        if verbose: print("finding best-fit values...")
        data = (Obs_ij, Obsnoise_ij, Kernel_il, nregparam, True, False, ntype)
        output = minimize(lnprob, Y0_array, args=data, method="Nelder-Mead")
    #    output = minimize(lnprob, Y0_array, args=data, method="L-BFGS-B" )
        best_fit = output["x"]
        if verbose: print("best-fit", best_fit)

        # more information about the best-fit parameters
        data = (Obs_ij, Obsnoise_ij, Kernel_il, regularization, nregparam, True, False)
        lnprob_bestfit = lnprob( output['x'], *data )

        # compute BIC
        BIC = 2.0 * lnprob_bestfit + len( output['x'] ) * np.log( len(Obs_ij.flatten()) )
        if verbose: print('BIC: ', BIC)

        # best-fit values for physical parameters
        if nregparam > 0:
            X_albd_kj, X_area_lk =  transform_Y2X(output["x"][:-1*nregparam], ntype, nband, nslice)
        else :
            X_albd_kj, X_area_lk =  transform_Y2X(output["x"], ntype, nband, nslice)

        X_albd_kj_T = X_albd_kj.T

        # best-fit values for regularizing parameters
        if regularization is not None:
            if regularization == 'Tikhonov' :
                if verbose: print('sigma', best_fit[-1])
            elif regularization == 'GP' :
                if verbose: print('overall_amp', best_fit[-3])
                if verbose: print('wn_rel_amp', np.exp( best_fit[-2] ) / ( 1. + np.exp( best_fit[-2] ) ))
                if verbose: print('lambda _angular', best_fit[-1] * ( 180. / np.pi ))
            elif regularization == 'GP2' :
                if verbose: print('overall_amp', best_fit[-2])
                if verbose: print('lambda _angular', best_fit[-1]* ( 180. / np.pi ))

        # Flatten best-fitting physical parameters
        bestfit = np.r_[ X_albd_kj.flatten(), X_area_lk.T.flatten() ]

        # Create dictionaries of initial results to convert to hdf5
        # datasets and attributes
        init_dict_datasets = {
            "best_fity" : best_fit,
            "X_area_lk" : X_area_lk,
            "X_albd_kj_T" : X_albd_kj_T,
            "best_fitx" : bestfit
        }
        init_dict_attrs = {
            "best_lnprob" : lnprob_bestfit,
            "best_BIC" : BIC
        }

        ############ run MCMC ############

        # Define MCMC parameters
        n_dim = len(Y0_array)
        n_walkers = calculate_walkers(n_dim)

        # Define data tuple for emcee
        data = (Obs_ij, Obsnoise_ij, Kernel_il, regularization, nregparam, False, False, ntype)

        # Initialize emcee EnsembleSampler
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob, args=data, threads=ncpu)

        # Guess starting position vector
        p0 = seed_amp * np.random.rand(n_dim * n_walkers).reshape((n_walkers, n_dim)) + best_fit

        # Do Burn-in run? No
        if verbose: print("Running MCMC from initial optimization...")

        # Run MCMC
        sampler.run_mcmc( p0, num_mcmc )

        # Extract chain from sampler
        original_samples = sampler.chain

        # Get model evaluations
        blobs = sampler.blobs
        shape = (len(blobs), len(blobs[0]), len(blobs[0][0]), len(blobs[0][0][0]))
        model_ij = np.reshape(blobs, shape)

        ############ Save HDF5 File ############

        # Specify hdf5 save file and group names
        hfile = os.path.join(run_dir, "samurai_out.hdf5")
        grp_init_name = "initial_optimization"
        grp_mcmc_name = "mcmc"
        grp_data_name = "data"
        compression = hdf5_compression

        # print
        if verbose: print("Saving:", hfile)

        # dictionary for global run metadata
        hfile_attrs = {
            "N_TYPE" : ntype,
            "N_SLICE" : nslice,
            "N_REGPARAM" : nregparam,
            "model" : model
        }

        # Create dictionaries for mcmc data and metadata
        mcmc_dict_datasets = {
            "samples" : original_samples,
            "model_ij" : model_ij,
            "p0" : p0
        }
        mcmc_dict_attrs = {
            "Y_names" : Y_names,
            "X_names" : X_names,
        }

        # Create dictionaries for observation data and metadata
        data_dict_datasets = {
            "Obs_ij" : Obs_ij,
            "Obsnoise_ij" : Obsnoise_ij,
            "Kernel_il" : Kernel_il,
            "lam_j" : waveband_centers,
            "dlam_j" : waveband_widths,
            "Time_i" : Time_i
        }
        data_dict_attrs = {
            "LON_S" : lon_s,
            "LAT_S" : lat_s,
            "LON_O" : lon_o,
            "LAT_O" : lat_o
        }

        # Create hdf5 file
        f = h5py.File(hfile, 'w')

        # Add global metadata
        for key, value in hfile_attrs.iteritems(): f.attrs[key] = value

        # Create hdf5 groups (like a directory structure)
        grp_init = f.create_group(grp_init_name)    # f["initial_optimization/"]
        grp_data = f.create_group(grp_data_name)    # f["data/"]
        grp_mcmc = f.create_group(grp_mcmc_name)    # f[mcmc/]

        # Save initial run datasets
        for key, value in init_dict_datasets.iteritems():
            grp_init.create_dataset(key, data=value, compression=compression)
        # Save initial run metadata
        for key, value in init_dict_attrs.iteritems():
            grp_init.attrs[key] = value

        # Save data datasets
        for key, value in data_dict_datasets.iteritems():
            grp_data.create_dataset(key, data=value, compression=compression)
        # Save data metadata
        for key, value in data_dict_attrs.iteritems():
            grp_data.attrs[key] = value

        # Save mcmc run datasets
        for key, value in mcmc_dict_datasets.iteritems():
            grp_mcmc.create_dataset(key, data=value, compression=compression)
        # Save mcmc run metadata
        for key, value in mcmc_dict_attrs.iteritems():
            grp_mcmc.attrs[key] = value

        # Close hdf5 file stream
        f.close()

################################################################################

class Data(object):
    """
    """
    def __init__(self, Time_i=None, Obs_ij=None, Obsnoise_ij=None, wlc_i=None, wlw_i=None):
        self.Time_i=Time_i
        self.Obs_ij=Obs_ij
        self.Obsnoise_ij=Obsnoise_ij
        self.wlc_i=wlc_i
        self.wlw_i=wlw_i

    @classmethod
    def from_EPOXI_march(cls):
        """
        Initialize Data using March 2008 EPOXI observations
        """
        infile = "../data/raddata_1_norm"
        Time_i = np.arange(25)*1.
        Obs_ij = np.loadtxt(os.path.join(RELPATH, infile))
        wlc_i = np.array([350., 450., 550., 650., 750., 850., 950.])
        wlw_i = np.array([100., 100., 100., 100., 100., 100., 100.])
        Obsnoise_ij = None
        # Return new class instance
        return cls(Time_i=Time_i, Obs_ij=Obs_ij, Obsnoise_ij=Obsnoise_ij,
                   wlc_i=wlc_i, wlw_i=wlw_i)

    @classmethod
    def from_EPOXI_june(cls):
        """
        Initialize Data using June 2008 EPOXI observations
        """
        infile = "../data/raddata_2_norm"
        Time_i = np.arange(25)*1.
        Obs_ij = np.loadtxt(os.path.join(RELPATH, infile))
        wlc_i = np.array([350., 450., 550., 650., 750., 850., 950.])
        wlw_i = np.array([100., 100., 100., 100., 100., 100., 100.])
        Obsnoise_ij = None
        # Return new class instance
        return cls(Time_i=Time_i, Obs_ij=Obs_ij, Obsnoise_ij=Obsnoise_ij,
                   wlc_i=wlc_i, wlw_i=wlw_i)

    @classmethod
    def from_test_simpleIGBP(cls):
        """
        Initialize Data using the simple IGBP map
        """
        infile = '../mockdata/simpleIGBP_quadrature_lc'
        Time_i = np.arange(7)/7.*24.
        Obs_ij = np.loadtxt(os.path.join(RELPATH, infile))
        wlc_i = np.array([550., 650., 750., 850.])
        wlw_i = np.array([100., 100., 100., 100.])
        Obsnoise_ij = None
        # Return new class instance
        return cls(Time_i=Time_i, Obs_ij=Obs_ij, Obsnoise_ij=Obsnoise_ij,
                   wlc_i=wlc_i, wlw_i=wlw_i)
