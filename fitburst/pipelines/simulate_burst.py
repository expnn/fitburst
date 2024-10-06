#! /bin/env/python

import matplotlib
from copy import deepcopy
import matplotlib.pyplot as plt
import fitburst as fb
import numpy as np
from datetime import datetime


matplotlib.rcParams["font.family"] = "times"
matplotlib.rcParams["font.size"] = 15
matplotlib.rcParams["xtick.labelsize"] = 12
matplotlib.rcParams["ytick.labelsize"] = 12

# define dimensions of the data.
is_dedispersed = False
num_freq = 2 ** 10
num_time = 1600
freq_lo = 1200.
freq_hi = 1600.
time_lo = 0.
time_hi = 0.08

freqs = np.linspace(freq_lo, freq_hi, num=num_freq)
times = np.linspace(time_lo, time_hi, num=num_time)

# define physical parameters for a dispersed burst to simulate.
params = {
    "amplitude": [0., -0.4, 0.],
    "arrival_time": [0.03, 0.04, 0.05],
    "burst_width": [0.001, 0.002, 0.0005],
    "dm": [349.5, 349.5, 349.5],
    "dm_index": [-2., -2., -2.],
    "ref_freq": [1500., 1400., 1300.],
    "scattering_index": [-4., -4., -4.],
    "scattering_timescale": [0., 0., 0.],
    "spectral_index": [0., 0., 0.],
    "spectral_running": [-300., -300., -300.],
}

num_components = len(params["dm"])

# define and/or extract parameters.
new_params = deepcopy(params)

if is_dedispersed:
    new_params["dm"] = [0.] * num_components

# define model object for CHIME/FRB data and load in parameter values.
model_obj = fb.analysis.model.SpectrumModeler(
    freqs,
    times,
    is_dedispersed=is_dedispersed,
    num_components=num_components,
    scintillation={
        "force_enable": False,
        "enable_prob": 0.5,
        # "avg_disappear_lifetime": {
        #     "type": "uniform",
        #     "args": {
        #         "low": 0.002,
        #         "high": 0.008
        #     },
        # },
        "avg_disappear_lifetime": {
            "type": "scipy.stats.beta",
            "args": {
                "a": 1.0,
                "b": 5.0,
                "loc": 0.002,
                "scale": 0.008,
            },
        },
        "avg_survival_lifetime": {
            "type": "uniform",
            "args": {
                "low": 0.002,
                "high": 0.008
            },
        },
        "delta_amplification": {
            "type": "normal",
            "args": {
                "loc": {
                    "type": "uniform",
                    "args": {
                        "low": 0.2,
                        "high": 0.8,
                    }
                },
                "scale": {
                    "type": "uniform",
                    "args": {
                        "low": 0.1,
                        "high": 0.2,
                    }
                }
            }
        }
    },
    verbose=False,
)

model_obj.update_parameters(new_params)

# now compute model and add noise.
model = model_obj.compute_model()
model += np.random.normal(0., 0.2, size=model.shape)
print(model.shape)

# plot.
fig = plt.figure()
ax = fig.gca()
ax.pcolormesh(times, freqs, model)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Observing Frequency (MHz)")
# plt.savefig(f"simulated_data-{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}.png", dpi=300)
fig.show()
fig = plt.figure()
ax = fig.gca()
ax.pcolormesh(times, freqs, np.sum(model_obj.clean_spec_per_component, axis=0))
ax.set_xlabel("Time (s)")
ax.set_ylabel("Observing Frequency (MHz)")
fig.show()

input("Press enter to continue...")
#
# # finally, save data into fitburst-generic format.
# metadata = {
#     "bad_chans": [],
#     "freqs_bin0": freqs[0],
#     "is_dedispersed": is_dedispersed,
#     "num_freq": num_freq,
#     "num_time": num_time,
#     "times_bin0": 0.,
#     "res_freq": freqs[1] - freqs[0],
#     "res_time": times[1] - times[0]
# }
#
# np.savez(
#     "simulated_data.npz",
#     data_full=model,
#     metadata=metadata,
#     burst_parameters=params,
# )
