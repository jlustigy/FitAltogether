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
                 sigmay=3.0, noiselevel=0.01, N_mcmc=10000, N_mcmc_b=0, mcmc_seedamp=0.5,
                 hdf5_compression='lzf', period=None, lat_s=None, lon_s=None,
                 lat_o=None, lon_o=None):
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
        self.N_mcmc=N_mcmc
        self.N_mcmc_b=N_mcmc_b
        self.mcmc_seedamp=mcmc_seedamp
        self.hdf5_compression=hdf5_compression

        # Params unique to map model
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
