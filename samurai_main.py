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
from map_utils import generate_tex_names, save2hdf5, calculate_walkers

class map_EPOXI_mcmc(object):
    """
    """
    def __init__(self, path="map_EPOXI_params.py"):

        if not path.endswith('.py'):
            print("Incompatible file.")
            return

        try:
            del sys.modules['input']
        except KeyError:
            pass

        user_input_file = path

        self._input = imp.load_source("input", user_input_file)            # Load inputs into self._input

        inp_dict = self._input.__dict__

        for key, value in inp_dict.items():
            if key.startswith('__') or isinstance(value, ModuleType) or isinstance(value, FunctionType):
                inp_dict.pop(key, None)

        self.__dict__.update(inp_dict)                                        # Make all parameters accessible as self.param

        del self._input


    #--------------------------------------------------------------------
    # Parameters
    #--------------------------------------------------------------------

    #from map_EPOXI_params import N_TYPE, N_SLICE, MONTH, NOISELEVEL, \
    #    NUM_MCMC, NUM_MCMC_BURNIN, SEED_AMP, N_SIDE, OMEGA, REGULARIZATION, \
    #    calculate_walkers, HDF5_COMPRESSION, WAVEBAND_CENTERS, WAVEBAND_WIDTHS

        MONTH = self.MONTH
        REGULARIZATION = self.REGULARIZATION

        self.NCPU = multiprocessing.cpu_count()

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
            LON_S = 280.977
            LAT_S = 22.531
            LON_O = 205.465
            LAT_O = 0.264
        #    LON_O = 165.4663412
        #    LAT_O = -0.3521857
        #    LON_S = 239.1424068
        #    LAT_S = 21.6159766
            INFILE = "data/raddata_2_norm"
            Time_i = np.arange(25)*1.

        elif ( MONTH == 'test' ):
        # from spectroscopic data
        #         Sub-Sun Lon/Lat =      97.091       -0.581 /     W longitude, degrees
        #         Sub-SC  Lon/Lat =     154.577        1.678 /     W longitude, degrees
            LON_S = 280.977
            LAT_S = 22.531
            LON_O = 205.465
            LAT_O = 0.264
        #    INFILE = "mockdata/mock_simple_JuneKernel_scattered0.01_data_with_noise"
            INFILE = "mockdata/mock_simple_3types_JuneKernel_scattered0.01_data_with_noise"
            Time_i = np.arange(25)*1.

        elif ( MONTH == 'simpleIGBP' ):
            LON_S = 90.0
            LAT_S = 0.0
            LON_O = 0.0
            LAT_O = 0.0
            INFILE = 'mockdata/simpleIGBP_quadrature_lc'
            Time_i = np.arange(7)/7.*24.

        else :
            print 'ERROR: Invalid MONTH'
            sys.exit()

        N_REGPARAM = 0
        if REGULARIZATION is not None:
            if REGULARIZATION == 'Tikhonov' :
                N_REGPARAM = 1
            elif REGULARIZATION == 'GP' :
                N_REGPARAM = 3
            elif REGULARIZATION == 'GP2' :
                N_REGPARAM = 2
        else :
            N_REGPARAM = 0

        self.N_REGPARAM = N_REGPARAM
        self.LAT_S = LAT_S
        self.LON_S = LON_S
        self.LAT_O = LAT_O
        self.LON_O = LON_O
        self.INFILE =INFILE
        self.Time_i =Time_i

    #===================================================
    #if __name__ == "__main__":
    def run_mcmc(self):

        # print start time
        now = datetime.datetime.now()
        print now.strftime("%Y-%m-%d %H:%M:%S")

        # Create directory for this run
        startstr = now.strftime("%Y-%m-%d--%H-%M")
        run_dir = os.path.join("mcmc_output", startstr)
        os.mkdir(run_dir)
        print "Created directory:", run_dir

        # Save THIS file and the param file for reproducibility!
        thisfile = os.path.basename(__file__)
        paramfile = "map_EPOXI_params.py"
        newfile = os.path.join(run_dir, thisfile)
        commandString1 = "cp " + thisfile + " " + newfile
        commandString2 = "cp "+paramfile+" " + os.path.join(run_dir,paramfile)
        os.system(commandString1)
        os.system(commandString2)
        print "Saved :", thisfile, " &", paramfile

        # input data
        Obs_ij = np.loadtxt(self.INFILE)
        Obsnoise_ij = ( self.NOISELEVEL * Obs_ij )
        n_band = len(Obs_ij[0])

        # set kernel
        param_geometry = ( self.LAT_O, self.LON_O, self.LAT_S, self.LON_S, self.OMEGA )
        Kernel_il = geometry.kernel( self.Time_i, self.N_SLICE, self.N_SIDE, param_geometry )

        # initialize the fitting parameters
        X0_albd_kj = 0.3+np.zeros([self.N_TYPE, n_band])
        X0_area_lk = 0.2+np.zeros([self.N_SLICE, self.N_TYPE])
        Y0_array = reparameterize.transform_X2Y(X0_albd_kj, X0_area_lk)
        if ( self.N_REGPARAM > 0 ) :
            Y0_array = np.append(Y0_array, np.array([10.]*self.N_REGPARAM) )
        n_dim = len(Y0_array)
        print 'Y0_array', Y0_array
        print '# of parameters', n_dim
        print 'N_REGPARAM', self.N_REGPARAM
        if (self.N_REGPARAM > 0):
            X_albd_kj, X_area_lk =  reparameterize.transform_Y2X(Y0_array[:-1*self.N_REGPARAM], self.N_TYPE, n_band, self.N_SLICE)
        else:
            X_albd_kj, X_area_lk =  reparameterize.transform_Y2X(Y0_array, self.N_TYPE, n_band, self.N_SLICE)

        # Create list of strings for Y & X parameter names
        Y_names, X_names = generate_tex_names(self.N_TYPE, n_band, self.N_SLICE)

        ############ run minimization ############

        # minimize
        print "finding best-fit values..."
        data = (Obs_ij, Obsnoise_ij, Kernel_il, self.N_REGPARAM, True, False)
        output = minimize(self.lnprob, Y0_array, args=data, method="Nelder-Mead")
    #    output = minimize(lnprob, Y0_array, args=data, method="L-BFGS-B" )
        best_fit = output["x"]
        print "best-fit", best_fit

        # more information about the best-fit parameters
        data = (Obs_ij, Obsnoise_ij, Kernel_il, self.N_REGPARAM, True, False)
        lnprob_bestfit = self.lnprob( output['x'], *data )

        # compute BIC
        BIC = 2.0 * lnprob_bestfit + len( output['x'] ) * np.log( len(Obs_ij.flatten()) )
        print 'BIC: ', BIC

        # best-fit values for physical parameters
        if self.N_REGPARAM > 0:
            X_albd_kj, X_area_lk =  reparameterize.transform_Y2X(output["x"][:-1*self.N_REGPARAM], self.N_TYPE, n_band, self.N_SLICE)
        else :
            X_albd_kj, X_area_lk =  reparameterize.transform_Y2X(output["x"], self.N_TYPE, n_band, self.N_SLICE)

        X_albd_kj_T = X_albd_kj.T

        # best-fit values for regularizing parameters
        if self.REGULARIZATION is not None:
            if self.REGULARIZATION == 'Tikhonov' :
                print 'sigma', best_fit[-1]
            elif self.REGULARIZATION == 'GP' :
                print 'overall_amp', best_fit[-3]
                print 'wn_rel_amp', np.exp( best_fit[-2] ) / ( 1. + np.exp( best_fit[-2] ) )
                print 'lambda _angular', best_fit[-1] * ( 180. / np.pi )
            elif REGULARIZATION == 'GP2' :
                print 'overall_amp', best_fit[-2]
                print 'lambda _angular', best_fit[-1]* ( 180. / np.pi )

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

        """
        # Save initialization run as npz
        print "Saving:", run_dir+"initial_minimize.npz"
        np.savez(run_dir+"initial_minimize.npz", data=data, best_fity=best_fit, \
            lnprob_bestfit=lnprob_bestfit, X_area_lk=X_area_lk, X_albd_kj_T=X_albd_kj_T)
        """

        ############ run MCMC ############

        # Define MCMC parameters
        n_dim = len(Y0_array)
        n_walkers = calculate_walkers(n_dim)

        # Define data tuple for emcee
        data = (Obs_ij, Obsnoise_ij, Kernel_il, self.N_REGPARAM, False, False)

        # Initialize emcee EnsembleSampler
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, self.lnprob, args=data, threads=self.NCPU)

        # Guess starting position vector
        p0 = self.SEED_AMP * np.random.rand(n_dim * n_walkers).reshape((n_walkers, n_dim)) + best_fit

        # Do Burn-in run?
        if self.NUM_MCMC_BURNIN > 0:
            print "Running MCMC burn-in..."
            # Run MCMC burn-in
            pos, prob, state = sampler.run_mcmc( p0, self.NUM_MCMC_BURNIN )
            # Save initial positions of chain[n_walkers, steps, n_dim]
            burnin_chain = sampler.chain[:, :, :].reshape((-1, n_dim))
            # Save chain[n_walkers, steps, n_dim] as npz
            now = datetime.datetime.now()
            print "Finished Burn-in MCMC:", now.strftime("%Y-%m-%d %H:%M:%S")
            print "Saving:", run_dir+"mcmc_burnin.npz"
            np.savez(run_dir+"mcmc_burnin.npz", pos=pos, prob=prob, burnin_chain=burnin_chain)
            # Set initial starting position to the current state of chain
            p0 = pos
            # Reset sampler for production run
            sampler.reset()
            print "Running MCMC from burned-in position..."
        else:
            print "Running MCMC from initial optimization..."

        # Run MCMC
        sampler.run_mcmc( p0, self.NUM_MCMC )

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
        compression = self.HDF5_COMPRESSION

        # print
        print "Saving:", hfile

        # dictionary for global run metadata
        hfile_attrs = {
            "N_TYPE" : self.N_TYPE,
            "N_SLICE" : self.N_SLICE,
            "N_REGPARAM" : self.N_REGPARAM
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
            "lam_j" : self.WAVEBAND_CENTERS,
            "dlam_j" : self.WAVEBAND_WIDTHS,
            "Time_i" : self.Time_i
        }
        data_dict_attrs = {
            "datafile" : self.INFILE,
            "LON_S" : self.LON_S,
            "LAT_S" : self.LAT_S,
            "LON_O" : self.LON_O,
            "LAT_O" : self.LAT_O
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

        return
