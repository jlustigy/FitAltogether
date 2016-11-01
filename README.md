# Exoplanet Mapping with `SAMURAI`

Surface Albedo Mapping Using RotAtional Inversion (`SAMURAI`)

# Running the code

## Directly fitting the lightcurve

A1. Set the simulation parameters in: `fitlc_params.py`  

B1. Run the mcmc code:
```bash
python main_fitlc_mcmc_EPOXI_NEW.py
```
This will create a directory for this unique run (labeled by timestamp). The output HDF5 file containing the mcmc chains and data can be found there.  

## Fitting with the map

A2. Set the simulation parameters in: `map_EPOXI_params.py`  

B2. Run the mcmc code:
```bash
python main_map_EPOXI_mcmc.py
```
Essentially the same output as the lightcurve fitting method. 

## Analyzing the output

C. Analyze mcmc output:
```bash
python mcmc_analysis.py -d <run_directory> trace
```
This creates 'trace' plots for the trajectories of all mcmc chains through each parameter's space (with `trace` keyword). It is helpful to inspect the trace plots to determine the 'burn-in' by eye.   

D. Perform physical analysis:
```bash
python mcmc_physical.py -d <run_directory> -b <burn-in index> posterior area-alb model-data
```
This will generate histograms of the posterior distrubution for each parameter (with `posterior` keyword), a plot showing the retrieved area and albedos (with `area-alb` keyword), and a plot showing the model fit to the data (with `model-data` keyword).
