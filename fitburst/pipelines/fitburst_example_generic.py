#! /usr/bin/env python

# configure backend for matplotlib.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fitburst.analysis.model import SpectrumModeler
from fitburst.backend.generic import DataReader
from fitburst.analysis.fitter import LSFitter
from copy import deepcopy
import fitburst.routines.manipulate as manip
import fitburst.utilities as ut
import numpy as np
import argparse
import json
import sys
import os

parser = argparse.ArgumentParser(description=
    "A Python3 script that uses fitburst API to read, preprocess, and fit 'generic'-formatted data " + 
    "against a model of the dynamic spectrum."
)

parser.add_argument(
    "file", 
    action="store", 
    type=str,
    help="A Numpy state file containing data and metadata in fitburst-compliant ('generic') format."
)

parser.add_argument(
    "--fit", 
    action="store", 
    dest="parameters_to_fit", 
    default=[], 
    nargs="+", 
    type=str,
    help="A list of assumed-fixed model parameters to fit during least-squares estimation."
)

parser.add_argument(
    "--fix", 
    action="store", 
    dest="parameters_to_fix", 
    default=[], 
    nargs="+", 
    type=str,
    help="A list of model parameters to hold fixed to initial values."
)

parser.add_argument(
    "--iterations", 
    action="store", 
    dest="num_iterations", 
    default=1, 
    type=int,
    help="Integer number of fit iterations."
)

parser.add_argument(
    "--scattering_timescale", 
    action="store", 
    dest="scattering_timescale",
    default=None, 
    nargs="+", 
    type=float,
    help="Initial guess for scattering timescale."
)

parser.add_argument(
    "--solution", 
    action="store", 
    default=None, 
    dest="solution_file", 
    type=str,
    help="If set, use existing solution in fitburst-compliant JSON format."
)

parser.add_argument(
    "--verbose", 
    action="store_true",
    default=False, 
    dest="verbose",
    help="If set, then print additional information related to fit parameters and fitting."
)

parser.add_argument(
    "--width", 
    action="store", 
    dest="width", 
    default=None, 
    nargs="+", 
    type=float,
    help="Initial guess for burst width, in seconds."
)

parser.add_argument(
    "--window", 
    action="store", 
    dest="window", 
    default=None, 
    type=float,
    help="Half of size of data window, in seconds."
)

### grab CLI inputs from argparse.
args = parser.parse_args()
input_file = args.file
num_iterations = args.num_iterations
parameters_to_fit = args.parameters_to_fit
parameters_to_fix = args.parameters_to_fix
scattering_timescale = args.scattering_timescale
solution_file = args.solution_file
verbose = args.verbose
width = args.width
window = args.window
snr_threshold = 10.

# before proceeding, adjusted fixed-parameter list if necessary.
parameters_to_fix += ["dm_index", "scattering_index", "scattering_timescale"]

for current_fit_parameter in parameters_to_fit:
    if current_fit_parameter in parameters_to_fix:
        parameters_to_fix.remove(current_fit_parameter)

# also, check if JSON solution can be read (if provied.
existing_results = None

if solution_file is not None and os.path.isfile(solution_file):
    try:
        existing_results = json.load(open(solution_file, "r"))

    except:
        print("WARNING: input solution cannot be read; proceeding using basic initial parameters...")

else:
    print("INFO: no solution file found or provided; proceeding with fit...")

### read in input data.
data = DataReader(input_file)

### load data into memory and pre-process.
data.load_data()
data.good_freq = np.sum(data.data_weights, axis=1) // data.num_time

# get parameters and configure initial guesses.
initial_parameters = data.burst_parameters
current_parameters = deepcopy(initial_parameters)

if existing_results is not None:
    current_parameters = existing_results["model_parameters"]

else:
    # assume some basic guesses.
    current_parameters["arrival_time"] = [0.5]
    current_parameters["burst_width"] = [0.05]
    current_parameters["scattering_timescale"] = [0.0]
    current_parameters["spectral_index"] = [-1.0]
    current_parameters["spectral_running"] = [1.0]

# if values are supplied at command line, then overload those here.
if scattering_timescale is not None:
    current_parameters["scattering_timescale"] = scattering_timescale

if width is not None:
    current_parameters["burst_width"] = width

# update DM value to use ("full" or DM offset) for dedispersion if 
# input data are already dedispersed or not.
if data.is_dedispersed:
    print("INFO: input data cube is already dedispersed!")
    print("INFO: setting 'dm' entry to 0, now considered a dm-offset parameter...")
    current_parameters["dm"][0] = 0.0

# print parameter info if desired.
if verbose:
    print(f"INFO: initial guess for {len(current_parameters['dm'])}-component model:")

    for current_parameter_label in current_parameters.keys():
        current_list = current_parameters[current_parameter_label]
        print(f"    * {current_parameter_label}: {current_list}")

# now compute dedisperse matrix for data, given initial DM (or DM offset), 
# and grab windowed data.
print("INFO: computing dedispersion-index matrix")
data.dedisperse(
    initial_parameters["dm"][0],
    current_parameters["arrival_time"][0],
    reference_freq=initial_parameters["ref_freq"][0],
    dm_offset=0.0
)

data_windowed = data.data_full
times_windowed = data.times

if window is not None:
    data_windowed, times_windowed = data.window_data(params["arrival_time"][0], window=window)

### now create initial model.
print("INFO: initializing model")
model = SpectrumModeler()
model.dm0 = initial_parameters["dm"][0]
model.is_dedispersed = data.is_dedispersed
model.is_folded = True
model.set_dimensions(data.num_freq, len(times_windowed))
model.update_parameters(current_parameters)

### now set up fitter and execute least-squares fitting
for current_iteration in range(num_iterations):
    fitter = LSFitter(model)
    fitter.fix_parameter(parameters_to_fix)
    fitter.weighted_fit = True
    fitter.fit(times_windowed, data.freqs, data_windowed)

    # extract best-fit data for next loop.
    if fitter.success:
        model.update_parameters(fitter.fit_statistics["bestfit_parameters"])
 
        # if this is the last iteration, create best-fit model and plot windowed data.
        if current_iteration == (num_iterations - 1):
            bestfit_parameters = fitter.fit_statistics["bestfit_parameters"]
            bestfit_uncertainties = fitter.fit_statistics["bestfit_uncertainties"]
            model.update_parameters(bestfit_parameters)
            bestfit_model = model.compute_model(times_windowed, data.freqs)
            bestfit_residuals = data_windowed - bestfit_model

            if verbose:
                print(f"INFO: best-fit estimate for {len(current_parameters['dm'])}-component model:")

                for current_parameter_label in bestfit_parameters.keys():
                    current_list = bestfit_parameters[current_parameter_label]
                    current_uncertainties = bestfit_uncertainties[current_parameter_label]
                    print(f"    * {current_parameter_label}: {current_list} +/- {current_uncertainties}")        

            # now create plots.
            ut.plotting.plot_summary_triptych(
                times_windowed, data.freqs, data_windowed, fitter.good_freq, model = bestfit_model,
                residuals = bestfit_residuals, output_name = f"summary_plot.png", show = False
            )

            with open(f"results_fitburst.json", "w") as out:
                json.dump(
                    {
                        "initial_dm": initial_parameters["dm"][0],
                        "initial_time": data.times_bin0,
                        "model_parameters": model.get_parameters_dict(),
                        "fit_statistics": fitter.fit_statistics,
                    },
                    out,
                    indent=4
                )
