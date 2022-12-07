#! /user/bin/env python

from fitburst.analysis.peak_finder import FindPeak
from fitburst.analysis.fitter import LSFitter
import fitburst.backend.chimefrb as chimefrb
import fitburst.analysis.model as mod
import chime_frb_constants as const
import fitburst.utilities as ut
import fitburst.routines as rt
import numpy as np
import copy
import json
import sys
import os

### import and configure matplotlib.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

### import and configure logger.
import datetime
import logging
right_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(filename=f"fitburst_run_{right_now}.log", level=logging.DEBUG)
log = logging.getLogger("fitburst")

### import and configure argparse.
import argparse

parser = argparse.ArgumentParser(description=
    "A Python3 script that uses fitburst API to read, preprocess, window, and fit CHIME/FRB data " + 
    "against a model of the dynamic spectrum." 
)

parser.add_argument(
    "eventIDs", action="store", nargs="+", type=int,
    help="One or more CHIME/FRB event IDs."
)

parser.add_argument(
    "--amplitude", action="store", dest="amplitude", default=None, nargs="+", type=float,
    help="Initial guess for burst amplitude, in dex."
)

parser.add_argument(
    "--arrival_time", action="store", dest="arrival_time", default=None, nargs="+", type=float,
    help="Initial guess for arrival time, in seconds."
)

parser.add_argument(
    "--beam", action="store", dest="beam", default=0, type=int,
    help="Index of beam list.."
)

parser.add_argument(
    "--dm", action="store", dest="dm", default=None, nargs="+", type=float,
    help="Initial guess for dispersion measure (DM), in pc/cc."
)

parser.add_argument(
    "--downsample_freq", action="store", dest="factor_freq_downsample", default=64, type=int,
    help="Downsample the raw spectrum along the frequency axis by a specified integer."
)

parser.add_argument(
    "--downsample_time", action="store", dest="factor_time_downsample", default=1, type=int,
    help="Downsample the raw spectrum along the time axis by a specified integer."
)

parser.add_argument(
    "--fit", action="store", dest="parameters_to_fit", default=[], nargs="+", type=str,
    help="A list of model parameters to fit during least-squares estimation."
)

parser.add_argument(
    "--fix", action="store", dest="parameters_to_fix", default=[], nargs="+", type=str,
    help="A list of model parameters to hold fixed to initial values."
)

parser.add_argument(
    "--freqmean", action="store", dest="freq_mean", default=None, nargs="+", type=float,
    help="Initial guess for mean of (Gaussian) spectral energy distribution, in units of MHz."
)

parser.add_argument(
    "--freqwidth", action="store", dest="freq_width", default=None, nargs="+", type=float,
    help="Initial guess for width of (Gaussian) spectral energy distribution, in units of MHz."
)

parser.add_argument(
    "--freqmodel", action="store", dest="freq_model", default="powerlaw", type=str,
    help="Type of model for spectral energy distribution ('gaussian' or 'powerlaw')."
)

parser.add_argument(
    "--iterations", action="store", dest="num_iterations", default=1, type=int,
    help="Integer number of fit iterations."
)

parser.add_argument(
    "--latest", action="store", default=None, dest="latest_solution_location", type=str,
    help="If set, use existing solution if present."
)

parser.add_argument(
    "--normalize_variance", action="store_true", dest="normalize_variance", 
    help="If set, then normalize variance during preprocessing of spectrum."
)

parser.add_argument(
    "--offset_dm", action="store", dest="offset_dm", default=0.0, type=float,
    help="Offset applied to initial dispersion measure, in pc/cc."
)

parser.add_argument(
    "--offset_time", action="store", dest="offset_time", default=0.0, type=float,
    help="Offset applied to initial arrival time, in seconds."
)

parser.add_argument(
    "--pipeline", action="store", dest="pipeline", default="L1", type=str,
    help="Name of CHIME/FRB pipeline whose results will be used as initial guesses."
)

parser.add_argument(
    "--save", action="store_true", dest="save_results",
    help="If set, save best-fit results to a JSON file."
)

parser.add_argument(
    "--scattering_timescale", action="store", dest="scattering_timescale", default=None, nargs="+", type=float,
    help="Initial guess for scattering index."
)

parser.add_argument(
    "--scintillation", action="store_true", dest="scintillation",
    help="If set, then enable per-channel amplitude estimation in cases of scintillation."
)

parser.add_argument(
    "--spectral_index", action="store", dest="spectral_index", default=None, nargs="+", type=float,
    help="Initial guess for spectral index."
)

parser.add_argument(
    "--spectral_running", action="store", dest="spectral_running", default=None, nargs="+", type=float,
    help="Initial guess for spectral running."
)

parser.add_argument(
    "--upsample_freq", action="store", dest="factor_freq_upsample", default=8, type=int,
    help="Upsample the raw spectrum along the frequency axis by a specified integer."
)

parser.add_argument(
    "--upsample_time", action="store", dest="factor_time_upsample", default=4, type=int,
    help="Upsample the raw spectrum along the time axis by a specified integer."
)

parser.add_argument(
    "--variance_range", action="store", dest="variance_range", default=[0.95, 1.05], 
    nargs=2, type=float, help="Range of variance values used to designate channels with RFI."
)

parser.add_argument(
    "--variance_weight", action="store", dest="variance_weight", default=(1. / const.L0_NUM_FRAMES_SAMPLE / 2),
    type=float, help="Scaling value applied to variances in preprocessing step."
)

parser.add_argument(
    "--verbose", action="store_true", dest="verbose",
    help="If set, then print more information during pipeline execution."
)

parser.add_argument(
    "--width", action="store", dest="width", default=None, nargs="+", type=float,
    help="Initial guess for burst width, in seconds."
)

parser.add_argument(
    "--window", action="store", dest="window", default=0.08, type=float,
    help="Half of size of data window, in seconds."
)

# grab CLI inputs from argparse.
args = parser.parse_args()
amplitude = args.amplitude
arrival_time = args.arrival_time
beam = args.beam
dm = args.dm
eventIDs = args.eventIDs
factor_freq_downsample = args.factor_freq_downsample
factor_time_downsample = args.factor_time_downsample
factor_freq_upsample = args.factor_freq_upsample
factor_time_upsample = args.factor_time_upsample
latest_solution_location = args.latest_solution_location
normalize_variance = args.normalize_variance
num_iterations = args.num_iterations
offset_dm = args.offset_dm
offset_time = args.offset_time
parameters_to_fit = args.parameters_to_fit
parameters_to_fix = args.parameters_to_fix
pipeline = args.pipeline
save_results = args.save_results
scattering_timescale = args.scattering_timescale
scintillation = args.scintillation
spectral_index = args.spectral_index
spectral_running = args.spectral_running
variance_range = args.variance_range
variance_weight = args.variance_weight
verbose = args.verbose
width = args.width
window_orig = args.window

# before looping over events, suss out model parameters to fit and/or hold fixed.
parameters_to_fix += ["dm_index", "scattering_index", "scattering_timescale"]

for current_fit_parameter in parameters_to_fit:
    if current_fit_parameter in parameters_to_fix:
        parameters_to_fix.remove(current_fit_parameter)

# loop over all CHIME/FRB events supplied at command line.
for current_event_id in eventIDs:

    # grab initial parameters to check if pipeline-specific parameters exist.
    data = chimefrb.DataReader(current_event_id, beam_id=beam)
    print(data.burst_parameters)
    initial_parameters = data.get_parameters(pipeline=pipeline)
        
    # if returned parameter dictionary is empty, move on to next event.
    if bool(initial_parameters):
        log.info(f"successfully grabbed parameter data from CHIME/FRB {pipeline} pipeline")
        
    else:
        log.error(f"couldn't grab CHIME/FRB {pipeline} pipeline data for event {current_event_id}")
        continue

    window = window_orig

    # load data into memory and pre-process.
    try:
        data.load_data(data.files)
        log.info(f"successfully read raw msgpack data for event {current_event_id}")

    except Exception as exc:
        log.error(f"couldn't read raw msgpack data for event {current_event_id}")
        continue

    # now that frame0-nano value is available after loading of data, grab parameters 
    # to obtain timestamp info.
    initial_parameters = data.get_parameters(pipeline=pipeline)
    # if a JSON file containing results already exists, then read that in.
    latest_solution_file = f"{latest_solution_location}/results_fitburst_{current_event_id}.json"

    if (
        latest_solution_location is not None and os.path.isfile(latest_solution_file)
    ):
        log.info(f"loading data from results file for event {current_event_id}")
        results = json.load(open(latest_solution_file, "r"))
        initial_parameters = results["model_parameters"]

        try:
            window = results["fit_logistics"]["spectrum_window"]

        except:
            log.warning(f"window size not found in file '{latest_solution_file}'")
            

        log.info("window size adjusted to +/- {0:.1f} ms".format(window * 1e3))

    elif (
        latest_solution_location is not None and os.path.isfile(latest_solution_file)
    ):
        log.info(f"results already obtained and saved for {current_event_id}; ignoring fit and moving on...")
        continue

    else: 
        pass
        #initial_parameters["burst_width"] = [window / 10.]

    # if scattering timescale is a fit parameter, initially set to width.
    if (
        initial_parameters["scattering_timescale"][0] == 0. and 
        "scattering_timescale" not in parameters_to_fix
    ):
        initial_parameters["scattering_timescale"] = copy.deepcopy(
            (np.array(initial_parameters["burst_width"]) * 3.).tolist()
        )
        initial_parameters["burst_width"] = (np.array(initial_parameters["burst_width"]) / 3.).tolist()

    # if guesses are provided through CLI, overload them into the initial-guess dictionary.
    initial_parameters["dm"][0] += offset_dm
    initial_parameters["arrival_time"][0] += offset_time

    if amplitude is not None:
        initial_parameters["amplitude"] = amplitude
   
    if arrival_time is not None:
        initial_parameters["arrival_time"] = arrival_time
   
    if dm is not None:
        initial_parameters["dm"] = dm

    if scattering_timescale is not None:
        initial_parameters["scattering_timescale"] = scattering_timescale

    if spectral_index is not None:
        initial_parameters["spectral_index"] = spectral_index

    if spectral_running is not None:
        initial_parameters["spectral_running"] = spectral_running

    if width is not None:
        initial_parameters["burst_width"] = width
   
    # now, clean and normalize data.
    data.preprocess_data(
        normalize_variance=normalize_variance,
        variance_range=variance_range,
        variance_weight=variance_weight
    )

    # if the number of RFI-flagged channels is "too large", skip this event altogether.
    num_bad_freq = data.num_freq - np.sum(data.good_freq)

    if (num_bad_freq / data.num_freq) > 0.7:
        log.error(
            f" {num_bad_freq} out of {data.num_freq} frequencies masked for event {current_event_id}"
        )
        continue

    # now compute dedisperse matrix for data, given initial DM, and grab windowed data.
    print("INFO: computing dedispersion-index matrix")
    print("INFO: dedispersing over freq range ({0:.3f}, {1:.3f}) MHz".format(
            np.min(data.freqs), np.max(data.freqs)
        )
    )
    params = initial_parameters.copy()#data.burst_parameters["fitburst"]["round_2"]
    data.dedisperse(
        params["dm"][0],
        params["arrival_time"][0],
        ref_freq=params["ref_freq"][0]
    )

    # before doing anything, check if window size doesn't extend beyond data set.
    # if it does, adjust down by an appropriate amount.
    window_max = data.times[-1] - initial_parameters["arrival_time"][0]

    if window > window_max:
        window = window_max - 0.001
        print("INFO: window size adjusted to +/- {0:.1f} ms".format(window * 1e3))

    data_windowed, times_windowed = data.window_data(params["arrival_time"][0], window=window)
    #plt.plot(data_windowed.sum(axis=0))
    #plt.savefig("test.png")
    #sys.exit()

    # before defining model, adjust model parameters with peak-finding algorithm.
    #peaks = FindPeak(data_windowed, times_windowed, data.freqs, rms=105.)
    #peaks.find_peak() 
    #initial_parameters = peaks.get_parameters_dict(initial_parameters)

    #print(initial_parameters)
    #sys.exit()

    # before proceeding further, ensure that SED parameters match desired model.
    if freq_model == "gaussian":

        # if power-law parameters reside in dictionary, remove them.
        if "spectral_index" in initial_parameters:
            initial_parameters.pop("spectral_index", None)

        if "spectral_running" in initial_parameters:
            initial_parameters.pop("spectral_running", None)

        # now check that Gaussian-model parameters are initialized.
        if "freq_mean" not in initial_parameters or "freq_width" not in initial_parameters:
            sys.exit("ERROR: missing SED parameters for Gaussian model in parameter dictionary.")

        elif freq_model == "powerlaw":
            pass

    elif freq_model == "powerlaw":
        pass

    else:
        sys.exit(f"ERROR: cannot recognize SED model of type '{freq_model}.'")

    # now create initial model.
    # since CHIME/FRB data are in msgpack format, define a few things 
    # so that this version of fitburst works similar to the original version on site.
    print("INFO: initializing model")
    num_components = len(initial_parameters["amplitude"])
    initial_parameters["dm"] = [0.] * num_components

    model = mod.SpectrumModeler(
        data.freqs,
        times_windowed,
        dm_incoherent=params["dm"][0],
        factor_freq_upsample=factor_freq_upsample,
        factor_time_upsample=factor_time_upsample,
        num_components=num_components,
        is_dedispersed=True,
        is_folded=False,
        scintillation=scintillation,
        verbose=verbose,
    )

    model.update_parameters(initial_parameters)

    ### now set up fitter and execute least-squares fitting.
    for current_iteration in range(num_iterations):
        print(f"INFO: fitting model, loop #{current_iteration + 1}")
        fitter = LSFitter(model, good_freq=data.good_freq, weighted_fit=True)
        fitter.fix_parameter(parameters_to_fix)
        fitter.fit(data_windowed)
    
        # before executing the fitting loop, overload model class with best-fit parameters.
        if fitter.success:
            model.update_parameters(fitter.fit_statistics["bestfit_parameters"])

    ### now compute best-fit model of spectrum and plot.
    if fitter.success:
        bestfit_model = model.compute_model(data=data_windowed) * data.good_freq[:, None]
        bestfit_params = model.get_parameters_dict()
        bestfit_params["dm"] = [params["dm"][0] + x for x in bestfit_params["dm"]]
        bestfit_residuals = data_windowed - bestfit_model

        # create summary plot.
        data_grouped = ut.plotting.compute_downsampled_data(
            times_windowed, data.freqs, data_windowed, data.good_freq,
            spectrum_model = bestfit_model, factor_freq = factor_freq_downsample,
            factor_time = factor_time_downsample
        )

        ut.plotting.plot_summary_triptych(
            data_grouped, output_name = f"summary.{current_event_id}.png", show=False
        )

        # create JSON file contain burst parameters and statistics.
        with open(f"results_fitburst_{current_event_id}.json", "w") as out:
            json.dump(
                {
                    "model_parameters": bestfit_params,
                    "fit_statistics": fitter.fit_statistics,
                    "fit_logistics" : {
                        "dm_incoherent" : params["dm"],
                        "factor_freq_upsample" : factor_freq_upsample,
                        "factor_time_upsample" : factor_time_upsample,
                        "is_repeater": True,
                        "normalize_variance" : normalize_variance,
                        "spectrum_window": window,
                        "variance_range" : variance_range,
                        "variance_weight" : variance_weight
                    }
                },
                out, 
                indent=4
            )

        # finally, if desired, save spectrum and burst-parameter/metadata dictionaries.
        if save_results:
            np.savez(
                f"test_data_CHIMEFRB_{current_event_id}.npz",
                burst_parameters = bestfit_params,
                data_full = data_windowed,
                metadata = {
                    "bad_chans" : [],
                    "freqs_bin0" : data.freqs[0],
                    "is_dedispersed" : True,
                    "num_freq" : data.num_freq,
                    "num_time" : len(times_windowed),
                    "times_bin0" : times_windowed[0],
                    "res_freq" : data.res_freq,
                    "res_time" : data.res_time,
                }
                
            )
