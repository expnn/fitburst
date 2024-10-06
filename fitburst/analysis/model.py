"""
Object for Computing and Updating Models of Dynamic Spectra

The SpectrumModeler() object is designed to compute models of dynamic
spectra based on parameter values, and handle the updating of one or more
model parameters. The updating/retrieval methods are used in the fitburst
fitter object, and are written to handle user-specified fixing of parameters.
"""
import scipy
import numpy as np
from numpy.typing import ArrayLike
from itertools import chain
from typing import Sequence, Optional
from fitburst.backend import general
import fitburst.routines as rt
from fitburst.utilities import import_class


class SpectrumModeler:
    """
    A Python structure that contains all information regarding parameters that
    describe and are used to compute models of dynamic spectra.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, freqs: np.ndarray, times: np.ndarray, dm_incoherent: float = 0.,
                 factor_freq_upsample: int = 1, factor_time_upsample: int = 1, num_components: int = 1,
                 is_dedispersed: bool = False, is_folded: bool = False, scintillation: dict | bool = None,
                 verbose: bool = False) -> None:
        """
        Instantiates the model object and sets relevant parameters, depending on
        desired model for spectral energy distribution.

        Parameters
        ----------

        freqs: np.ndarray
            The frequency channels in the spectrum (including masked ones)

        times: np.ndarray
            The time samples in the spectrum

        dm_incoherent : float, optional
            The DM used to incoherently dedisperse input data; this is only used if the
            'is_dedispersed' argument is set to True

        factor_freq_upsample : int, optional
            The factor to upsample each frequency label into an array of subbands

        factor_time_upsample : int, optional
            The factor to upsample the array of timestamps

        is_dedispersed : bool, optional
            If true, then assume that the dispersion measure is an 'offset' parameter
            and computes the relative dispersion for non-zero offset values

        is_folded : bool, optional
            If true, then the temporal profile is computed over two realizations and then
            averaged down to one (in order to allow for wrapping of a folded pulse shape)

        num_components : int, optional
            The number of distinct burst components in the model

        scintillation : dict | bool, optional
            if true, the compute per-channel amplitudes using input data.

        verbose : bool, optional
            If true, then print parameter values during each function call
            (This is mainly useful to gauge least-squares fitting algorithms.)

        """

        # pylint: disable=too-many-arguments,too-many-locals

        # first define model-configuration parameters that are not fittable.
        self.dm_incoherent = dm_incoherent
        self.factor_freq_upsample = factor_freq_upsample
        self.factor_time_upsample = factor_time_upsample
        self.freqs = freqs
        self.is_dedispersed = is_dedispersed
        self.is_folded = is_folded
        self.num_components = num_components
        self.scintillation = bool(scintillation)
        self.scintillation_config = scintillation if isinstance(scintillation, dict) else None
        self.times = times
        self.verbose = verbose

        # derive some additional data needed for downstream fitting.
        self.num_freq = len(self.freqs)
        self.num_time = len(self.times)
        self.res_freq = np.fabs(self.freqs[1] - self.freqs[0])
        self.res_time = np.fabs(self.times[1] - self.times[0])

        # define all *fittable* model parameters first.
        # NOTE: 'ref_freq' is not listed here as it's a parameter that is always held fixed.
        self.parameters = [
            "amplitude",
            "arrival_time",
            "burst_width",
            "dm",
            "dm_index",
            "scattering_timescale",
            "scattering_index",
            "spectral_index",
            "spectral_running"
        ]

        # now instantiate parameter attributes and set initially to NoneType.
        self.ref_freq: Optional[list] = None
        self.amplitude: Optional[list] = None
        self.arrival_time: Optional[list] = None
        self.burst_width: Optional[list] = None
        self.dm: Optional[list] = None
        self.dm_index: Optional[list] = None
        self.scattering_timescale: Optional[list] = None
        self.scattering_index: Optional[list] = None
        self.spectral_index: Optional[list] = None
        self.spectral_running: Optional[list] = None

        # now instantiate the structures for per-component models, time differences, and temporal profiles.
        # (the following are used for computing derivatives and/or per-channel amplitudes.)
        self.amplitude_per_component = np.zeros(
            (self.num_components, self.num_freq, self.num_time), dtype=float
        )

        self.spectrum_per_component = np.zeros(
            (self.num_components, self.num_freq, self.num_time), dtype=float
        )

        self.timediff_per_component = np.zeros(
            (self.num_components, self.num_freq, self.num_time), dtype=float
        )

        self.timeprof_per_component = np.zeros(
            (self.num_components, self.num_freq, self.num_time), dtype=float
        )

        self.clean_spec_per_component = self.spectrum_per_component  # alias to spectrum_per_component

    def compute_model(self, data: np.ndarray = None) -> np.ndarray:
        """
        Computes the model dynamic spectrum based on model parameters (set as class
        attributes) and input values of times and frequencies.

        Parameters
        ----------
        data : float, optional
            an array of time_averaged data values to be used for per-channel normalization
        """

        # pylint: disable=no-member,too-many-locals

        if self.ref_freq is None and any(getattr(self, a) is None for a in self.parameters):
            unset_parameters = []
            for a in chain(self.parameters, ("ref_freq",)):
                if getattr(self, a) is None:
                    unset_parameters.append(a)
            raise ValueError(f"ERROR: model parameters are not set: {unset_parameters}")

        # if scintillation modeling is desired but data aren't provide, then exit.
        # if self.scintillation and data is None:
        #     ValueError("ERROR: scintillation modelling is desired by data are missing!")

        # now loop over bandpass.
        for current_freq in range(self.num_freq):
            # loop over all components.
            for current_component in range(self.num_components):
                # extract parameter values for current component.
                current_amplitude = self.amplitude[current_component]
                current_arrival_time = self.arrival_time[current_component]
                current_dm = self.dm[0]
                current_dm_index = self.dm_index[0]
                current_ref_freq = self.ref_freq[current_component]
                current_sc_idx = self.scattering_index[0]
                current_sc_time = self.scattering_timescale[0]
                current_sp_idx = self.spectral_index[current_component]
                current_sp_run = self.spectral_running[current_component]
                current_width = self.burst_width[current_component]

                if self.verbose and current_freq == 0:
                    if self.scintillation:
                        print(f"{current_dm:.5f} {current_arrival_time:.5f} ",
                              f"{current_sc_idx:.5f}  {current_sc_time:.5f}  {current_width:.5f}", end=" ")
                    else:
                        print(
                            f"{current_dm:.5f}  {current_amplitude:.5f}  {current_arrival_time:.5f}  ",
                            f"{current_sc_idx:.5f}  {current_sc_time:.5f}  {current_width:.5f}", end=" ")

                # create an upsampled version of the current frequency label.
                # even if no upsampling is desired, this will return an array
                # of length 1.
                current_freq_arr = rt.manipulate.upsample_1d(
                    [self.freqs[current_freq]],
                    self.res_freq,
                    self.factor_freq_upsample
                )

                # first, compute "full" delays for all upsampled frequency labels.
                relative_delay = rt.ism.compute_time_dm_delay(
                    self.dm_incoherent + current_dm,
                    general["constants"]["dispersion"],
                    current_dm_index,
                    current_freq_arr,
                    freq2=current_ref_freq,
                )

                # then compute "relative" delays with respect to central frequency.
                relative_delay -= rt.ism.compute_time_dm_delay(
                    self.dm_incoherent,
                    general["constants"]["dispersion"],
                    current_dm_index,
                    self.freqs[current_freq],  # noqa
                    freq2=current_ref_freq,
                )

                # now compute current-times array corrected for relative delay.
                current_times = rt.manipulate.upsample_1d(
                    self.times.copy() - current_arrival_time,
                    self.res_time,
                    self.factor_time_upsample
                )

                current_times_arr, _ = np.meshgrid(current_times, current_freq_arr)
                current_times_arr -= relative_delay[:, None]

                # before proceeding, compute and save the per-component time difference map.
                self.timediff_per_component[current_component, current_freq, :] = \
                    rt.manipulate.downsample_1d(
                        current_times_arr.mean(axis=0),
                        self.factor_time_upsample
                    )

                # first, adjust scattering timescale to current frequency label(s).
                current_sc_time_scaled = rt.ism.compute_time_scattering(
                    current_freq_arr,
                    current_ref_freq,
                    current_sc_time,
                    current_sc_idx
                )

                # 由于 scattering_timescale 通常被设置为零, 此时 current_sc_time_scaled == current_sc_time = 0, 两者 没有区别
                # print(f"current_sc_time = {current_sc_time} and current_sc_time_scaled = {current_sc_time_scaled}. "
                #       f"diff = {current_sc_time_scaled - current_sc_time}")

                # second, compute raw temporal profile.
                current_profile = self.compute_profile(
                    current_times_arr,
                    0.0,  # since 'current_times' is already corrected for DM.
                    current_sc_time,  # TODO: current_sc_time_scaled?
                    current_sc_idx,
                    current_width,
                    current_freq_arr[:, None],
                    current_ref_freq,
                    is_folded=self.is_folded,
                )

                self.timeprof_per_component[current_component, current_freq, :] = rt.manipulate.downsample_1d(
                    current_profile.mean(axis=0),
                    self.factor_time_upsample
                )

                # third, compute and scale profile by spectral energy distribution.
                current_profile *= rt.spectrum.compute_spectrum_rpl(
                    current_freq_arr,
                    current_ref_freq,
                    current_sp_idx,
                    current_sp_run,
                )[:, None]

                # before writing, downsize upsampled array to original size.
                current_profile = rt.manipulate.downsample_1d(
                    current_profile.mean(axis=0),
                    self.factor_time_upsample
                )

                # before exiting the loop, save different snapshots of the model.
                self.amplitude_per_component[current_component, current_freq, :] = rt.spectrum.compute_spectrum_rpl(
                    self.freqs[current_freq],
                    current_ref_freq,
                    current_sp_idx,
                    current_sp_run
                ) * (10 ** current_amplitude)
                self.spectrum_per_component[current_component, current_freq, :] = (
                        10 ** current_amplitude * current_profile)  # spectral index/running for current component.
                if current_freq == 0:
                    if self.verbose and not self.scintillation:
                        print(f"{current_sp_idx:.5f}  {current_sp_run:.5f}")

        # if desired, then compute per-channel amplitudes in cases where scintillation is significant.
        if self.scintillation:
            if data is not None:
                for freq in range(self.num_freq):
                    current_amplitudes = rt.ism.compute_amplitude_per_channel(
                        data[freq], self.timeprof_per_component[:, freq, :]
                    )
                    # now compute model with per-channel amplitudes determined.
                    for component in range(self.num_components):
                        current_profile = self.timeprof_per_component[component, freq, :]
                        self.amplitude_per_component[component, freq] = current_amplitudes[component]
                        self.spectrum_per_component[component, freq] = current_amplitudes[component] * current_profile
            else:  # this branch is only for simulation
                times = rt.ism.compute_time_dm_delay(
                    self.dm[0],
                    general["constants"]["dispersion"],
                    self.dm_index[0],
                    self.freqs.ravel(),  # noqa
                    self.ref_freq[0],
                )[::-1] + self.arrival_time[0]
                scint_cfg = self.scintillation_config
                force_enable = scint_cfg.get("force_enable", False)
                enable_prob = scint_cfg.get('enable_prob', 0.0)

                spectrum_per_component = np.copy(self.spectrum_per_component)
                for comp in range(self.num_components):
                    # 如果没有激活闪烁, 则保持该 component 不变. 没有激活必须满足两个条件, 第一是没有强制激活. 第二是随机概率大于阈值.
                    if (not force_enable) and np.random.uniform(0.0, 1.0) > enable_prob:
                        continue

                    if self.dm[0] > 0:  # 目前算法只能根据时间计算闪烁，无色散时，无法将频率转换为时间，因而无法计算闪烁。
                        assert np.all(np.greater(np.diff(times), 0.0))
                        w1 = self.compute_scintillation(times)  # (#freqs, )
                        self.amplitude_per_component[comp] *= np.reshape(w1, (-1, 1))

                    w2 = self.compute_scintillation(self.times)    # (#times, )
                    self.amplitude_per_component[comp] *= np.reshape(w2, (1, -1))
                    spectrum_per_component[comp] = (self.amplitude_per_component[comp] *
                                                    self.timeprof_per_component[comp])
                # spectrum_per_component is point to the new weighted array.
                # self.clean_spec_per_component is not changed.
                self.spectrum_per_component = spectrum_per_component
        return np.sum(self.spectrum_per_component, axis=0)

    # @staticmethod
    # def compute_scintillation(freq: np.ndarray, freq_ref: ArrayLike) -> np.ndarray:
    #     """ Include spectral scintillation across
    #         the band. Approximate effect as a sinusoid,
    #         with a random phase and a random decorrelation
    #         bandwidth.
    #     """
    #     freq_ref = np.array(freq_ref).ravel()
    #     num_components = len(freq_ref)
    #     freq_ref = freq_ref.reshape(num_components, 1)  # (#components, 1)
    #     freq = freq.ravel().reshape(1, -1)  # (1, #freqs)
    #
    #     # Make location of peaks / troughs random
    #     scint_phi = np.random.rand(num_components).reshape(num_components, 1)
    #
    #     # Make number of scintils between 0 and 10 (ish)
    #     max_nscint = np.random.uniform(45.0, 50.0)
    #     nscint = np.exp(np.random.uniform(np.log(40.0), np.log(max_nscint), size=num_components))
    #     nscint = nscint.reshape(num_components, 1)
    #
    #     print(f"nscint = {nscint}, scint_phi = {scint_phi}")
    #     envelope = (np.cos(2 * np.pi * nscint * (freq ** -2 / freq_ref ** -2) + scint_phi) + 1.0) / 2.0
    #     return envelope

    def compute_scintillation(self, t: np.ndarray):
        assert t.ndim == 1
        scint_cfg = self.scintillation_config
        time_step = t[1] - t[0]

        delta_amplification_dist = self.generate_multi_level_rv_dist_from_cfg(scint_cfg['delta_amplification'])
        lifetime_cfgs = [scint_cfg['avg_disappear_lifetime'], scint_cfg['avg_survival_lifetime']]
        amplification_multiplier = [-1., 1.]

        def gen_peak_point(t1, t2, size=1):
            delta_t = t2 - t1
            sigma = 0.25 * delta_t
            return scipy.stats.truncnorm.rvs(-1.2, 1.2, loc=(t1 + t2) / 2, scale=sigma, random_state=None, size=size)

        def gen_peak(t1, t2, m):
            peak_t_ = gen_peak_point(t1, t2)
            peak_val_ = 1.0 + m * np.abs(delta_amplification_dist())
            return peak_t_.item(), peak_val_.item()

        start_idx = 0
        interpolate_points = []
        loop_idx = np.random.choice(2)
        while start_idx < len(t) - 32:
            select_idx = loop_idx % 2
            steps = 0
            for _ in range(100):
                duration = self.generate_lifetime(lifetime_cfgs[select_idx])
                steps = int(np.floor(duration / time_step))
                if steps >= 32:
                    break
            steps = max(2, steps)  # 生成低分辨率动态谱时, 上面 for 循环 100 次可能也不能获取到足够大的 steps, 导致后续代码报错.
            end_idx = min(start_idx + steps, len(t))
            peak = gen_peak(t[start_idx], t[end_idx - 1], amplification_multiplier[select_idx])
            # try:
            #     peak = gen_peak(t[start_idx], t[end_idx - 1], amplification_multiplier[select_idx])
            # except:
            #     print(f"gen_peak failed: t[start_idx] = {t[start_idx]}, t[end_idx - 1] = {t[end_idx - 1]}, "
            #           f"start_idx = {start_idx}, end_idx = {end_idx}, steps = {steps}, len(t) = {len(t)}")
            #     raise
            interpolate_points.append((t[start_idx].item(), 1.0))
            interpolate_points.append(peak)

            start_idx = end_idx
            loop_idx += 1
        if interpolate_points:
            interpolate_points.append((t[-1].item(), 1.0))

        if len(interpolate_points) > 3:
            x, y = zip(*interpolate_points)
            x = np.array(x)
            y = np.array(y)
            f = scipy.interpolate.interp1d(x, y, kind='quadratic', assume_sorted=True)
            return np.clip(f(t), 0.0, 1.5).astype(np.float32)
        return np.ones_like(t, dtype=np.float32)

    @staticmethod
    def generate_lifetime(lifetime_cfg):
        lifetime_gen_func = import_class(lifetime_cfg.get('type', 'uniform'), 'numpy.random')
        if hasattr(lifetime_gen_func, 'rvs'):
            avg_lifetime = lifetime_gen_func.rvs(**lifetime_cfg['args'])
        else:
            avg_lifetime = lifetime_gen_func(**lifetime_cfg['args'])
        return np.random.exponential(avg_lifetime)

    @classmethod
    def generate_multi_level_rv_dist_from_cfg(cls, cfg):
        if not isinstance(cfg, dict):
            return lambda: cfg
        dist = cfg.get('type', 'constant')
        if dist == 'constant':
            if 'value' not in cfg:
                return lambda: cfg
            return lambda: cls.generate_multi_level_rv_from_cfg(cfg.get('value'))
        rv_gen_func = import_class(dist, 'numpy.random')
        kwargs = {}
        for k, v in cfg['args'].items():
            kwargs[k] = cls.generate_multi_level_rv_from_cfg(v)
        return lambda: rv_gen_func(**kwargs)

    @classmethod
    def generate_multi_level_rv_from_cfg(cls, cfg):
        rng = cls.generate_multi_level_rv_dist_from_cfg(cfg)
        return rng()

    @staticmethod
    def compute_profile(times: np.ndarray, arrival_time: float, sc_time_ref: float, sc_index: float,
                        width: float, freqs: float, ref_freq: float, is_folded: bool = False) -> np.ndarray:
        """
        Returns the temporal profile, depending on input values of width
        and scattering timescale.

        Parameters
        ----------
        times : float
            One or more values corresponding to time

        arrival_time : float
            The arrival time of the burst

        sc_time_ref : float
            The scattering timescale of the burst (which depends on frequency label)

        sc_index : float
            The index of frequency dependence on the scattering timescale

        width : float
            The intrinsic temporal width of the burst

        freqs : float
            The index of frequency dependence on the scattering timescale

        ref_freq : float
            The index of frequency dependence on the scattering timescale

        is_folded : bool, optional
            If true, then the temporal profile is computed over two realizations and then
            averaged down to one (in order to allow for wrapping of a folded pulse shape)

        Returns
        -------
        profile : float
            One or more values of the temporal profile, evaluated at the input timestamps
        """

        # pylint: disable=too-many-arguments,no-self-use

        # if data are "folded" (i.e., data from pulsar timing observations),
        # model at twice the timespan and wrap/average the two realizations.
        # this step accounts for potential wrapping of pulse shape.
        times_copy = times.copy()

        if is_folded:
            res_time = np.diff(times_copy, axis=1)[:, 0]
            start = times[:, -1] + res_time
            stop = times[:, -1] + (res_time * times.shape[1])
            times_extended = np.linspace(start=start, stop=stop, num=times.shape[1], axis=1)
            times_copy = np.append(times, times_extended, axis=1)

        # compute either Gaussian or pulse-broadening function, depending on inputs.
        sc_time = sc_time_ref * (freqs / ref_freq) ** sc_index

        if np.any(sc_time < np.fabs(0.15 * width)):
            profile = rt.profile.compute_profile_gaussian(times_copy, arrival_time, width)
        else:
            # the following times array manipulates the times array so that we avoid a
            # floating-point overlow in the exp((-times - toa) / sc_time) term in the
            # PBF call. TODO: use a better, more transparent method for avoiding this.
            times_copy[times_copy < -5 * width] = -5 * width
            profile = rt.profile.compute_profile_pbf(
                times_copy, arrival_time, width, freqs, ref_freq, sc_time_ref, sc_index=sc_index
            )

        # if data are folded and time/profile data contain two realizations, then
        # average along the appropriate axis to obtain a single realization.
        if is_folded:
            profile = profile.reshape(times.shape[0], 2, times.shape[1]).mean(1)

        return profile

    def get_parameters_dict(self) -> dict:
        """
        Returns model parameters as a dictionary, with keys set to the parameter names
        and values set to the Python list containing parameter values.

        Parameters
        ----------
        None : NoneType
            this method uses existing class attributes

        Returns
        -------
        parameter_dict : dict
            A dictionary containing parameter names as keys, and lists of per-component
            values as the dictionary values.
        """

        parameter_dict = {}

        # loop over all fittable parameters and grab their values.
        for current_parameter in self.parameters:
            parameter_dict[current_parameter] = getattr(self, current_parameter)

        # before exiting, grab the values of the reference frequency, which
        # isn't fittable and is therefore not in the 'parameters' list.
        parameter_dict["ref_freq"] = getattr(self, "ref_freq")

        return parameter_dict

    def update_parameters(
            self, model_parameters: dict, global_parameters: Sequence = ("dm", "scattering_timescale")) -> None:
        """
        Overloads parameter values stored in object with those supplied by the user.

        Parameters
        ----------
        model_parameters : dict
            a Python dictionary with parameter names listed as keys, parameter values
            supplied as lists tied to keys.
        global_parameters : list | tuple
            parameters that are shared among components.
        Returns
        -------
        None : None
            this method overloads class attributes.
        """

        # first, overload attributes with values for supplied parameters.
        for current_parameter in model_parameters.keys():
            setattr(self, current_parameter, model_parameters[current_parameter])

            # if number of components is 2 or greater, update num_components attribute.
            if len(model_parameters[current_parameter]) > 1:
                num_components = len(model_parameters[current_parameter])
                setattr(self, "num_components", num_components)

                if current_parameter in global_parameters:
                    setattr(self, current_parameter, [model_parameters[current_parameter][0]] * num_components)
