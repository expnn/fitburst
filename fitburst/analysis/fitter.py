"""
Object for Fitting Models against Data of Dynamic Spectra

The LSFitter() object is designed to least-squares fitting of dynamic
spectra based on the model defined and instantiated by the SpectrumModeler()
object. The LSFitter defines several methods to handle the fixing and fitting
of one or more parameters.
"""

from scipy.optimize import least_squares
import numpy as np

class LSFitter:
    """
    A Python object that defines methods and configurations for
    least-squares fitting of radio dynamic spectra.
    """

    def __init__(self, model_class: object, good_freq: bool, weighted_fit: bool = True):
        """
        Initializes object with methods and attributes defined in
        the model.SpectrumModeler() class.

        Parameters
        ----------
        model_class : object
            An instantiation of the SpectrumModeler() object

        good_freq : bool
            An array of boolean values that indicate if a frequency channel is good (True)
            or flagged as radio frequency interference (False)

        weighted_fit : bool, optional
            If set to true, then each channel will be weighted by its standard deviation (or,
            equivalently, the goodness of fit statistic will be a weighted chi-squared value.)

        Returns
        -------
        None : NoneType
            Several atributes are instantiated here
        """

        # load in model into fitter class.
        self.fit_parameters = []
        self.fit_statistics = {}
        self.model = model_class

        # initialize fit-parameter list.
        if self.model.scintillation:
            all_parameters = self.model.parameters.copy()

            for current_parameter in self.model.parameters:
                if current_parameter not in ["amplitude", "spectral_index", "spectral_running"]:
                    self.fit_parameters += [current_parameter]

        else:
            self.fit_parameters = self.model.parameters.copy()

        # set parameters for fitter configuration.
        self.good_freq = good_freq
        self.weighted_fit = weighted_fit
        self.weights = None
        self.success = None

    def compute_residuals(self, parameter_list: list, times: float,
        freqs: float, spectrum_observed: float) -> float:
        """
        Computes the chi-squared statistic used for least-squares fitting.

        Parameters
        ----------
        parameter_list : list
            A list of names for fit parameters

        times: float
            An array of values corresponding to observing times

        freqs : float
            An array of observing frequencies at which to evaluate spectrum

        spectrum_observed : float
            A matrix of spectrum data, with dimenions that match those of the times
            and freqs arrays

        Returns
        -------
        resid : np.ndarray
            An array of residuals (i.e., the difference between the observed and model spectra)
        """

        # define base model with given parameters.
        parameter_dict = self.load_fit_parameters_list(parameter_list)
        self.model.update_parameters(parameter_dict)
        model = self.model.compute_model(data=spectrum_observed)

        # now compute resids and return.
        resid = spectrum_observed - model
        resid *= self.weights[:, None]
        resid = resid.flat[:]

        return resid

    def fit(self, spectrum_observed: float) -> None:
        """
        Executes least-squares fitting of the model spectrum to data, and stores
        results to child class.

        Parameters
        ----------
        times: np.ndarray
            An array of values corresponding to observing times

        freqs : np.ndarray
            An array of observing frequencies at which to evaluate spectrum

        spectrum_observed : np.ndarray
            A matrix of spectrum data, with dimenions that match those of the times
            and freqs arrays

        Returns
        -------
        None
            A dictionary attribute is defined that contains the best-fit results from
            the scipy.optimize.least_aquares solver, as well as estimates of the
            covariance matrix and parameter uncertainties.
        """

        # pylint: disable=broad-except

        # convert loaded parameter dictionary/entries into a list for scipy object.
        parameter_list = self.get_fit_parameters_list()

        # before running fit, determine per-channel weights.
        self._set_weights(spectrum_observed)

        # do fit!
        try:
            results = least_squares(
                self.compute_residuals,
                parameter_list,
                args = (self.model.times, self.model.freqs, spectrum_observed)
            )

            self.success = results.success

            if self.success:
                print("INFO: fit successful!")

            else:
                print("INFO: fit didn't work!")

            # now try computing uncertainties and fit statistics.
            self._compute_fit_statistics(spectrum_observed, results)

            if self.success:
                print("INFO: derived uncertainties and fit statistics")

        except Exception as exc:
            print("ERROR: solver encountered a failure! Debug!")
            print(exc)

    def fix_parameter(self, parameter_list: list) -> None:
        """
        Updates 'fit_parameters' attributes to remove parameters that will
        be held fixed during least-squares fitting.

        Parameters
        ----------
        parameter_list : list
            A list of parameter names that will be fixed to input values during execution
            of the fitting routine

        Returns
        -------
        None : NoneType
            The 'fit_parameters' attribute is updated with supplied parameters removed

        Notes
        -----
        Names of parameters must match those defined in the model.SpectrumModeler class
        """

        print("INFO: removing the following parameters:", ", ".join(parameter_list))

        # removed desired parameters from fit_parameters list.
        for current_parameter in parameter_list:
            self.fit_parameters.remove(current_parameter)

        print("INFO: new list of fit parameters:", ", ".join(self.fit_parameters))

    def get_fit_parameters_list(self, global_parameters: list = ["dm", "scattering_timescale"]
        ) -> list:
        """
        Determines a list of values corresponding to fit parameters.

        Parameters
        ----------
        global_parameters : list, optional
            One or more fit parameters that are tied to all modeled burst components

        Returns
        -------
        parameter_list : list
            A list of floating-point values for fit parameters, which is used by SciPy solver
            as an argument to the residual-computation function
        """

        # pylint: disable=dangerous-default-value

        parameter_list = []

        # loop over all parameters, only extract values for fit parameters.
        for current_parameter in self.model.parameters:
            if current_parameter in self.fit_parameters:
                current_sublist = getattr(self.model, current_parameter)

                if current_parameter in global_parameters:
                    parameter_list += [current_sublist[0]]

                else:
                    parameter_list += current_sublist

        return parameter_list

    def load_fit_parameters_list(self, parameter_list: list,
        global_parameters: list = ["dm", "scattering_timescale"]) -> dict:
        """
        Determines a dictionary where keys are fit-parameter names and values
        are lists (with length self.model.num_components) contain the per-burst
        values of the given parameter/key.

        Parameters
        ----------
        parameter_list : list
            A list of floating-point values for fit parameters, which is used by SciPy solver
            as an argument to the residual-computation function

        global_parameters : list, optional
            One or more fit parameters that are tied to all modeled burst components

        Returns
        -------
        parameter_dict : dict
            A dictionary containing fit parameters as keys and their values as dictionary values
        """

        # pylint: disable=dangerous-default-value

        parameter_dict = {}

        # loop over all parameters, only preserve values for fit parameters.
        current_idx = 0

        for current_parameter in self.model.parameters:
            if current_parameter in self.fit_parameters:

                # if global parameter, load list of length == 1 into dictionary.
                if current_parameter in global_parameters:
                    parameter_dict[current_parameter] = [parameter_list[current_idx]]
                    current_idx += 1

                else:
                    parameter_dict[current_parameter] = parameter_list[
                        current_idx : (current_idx + self.model.num_components)
                    ]

                    current_idx += self.model.num_components


        return parameter_dict

    def _compute_fit_statistics(self, spectrum_observed: float, fit_result: object) -> None:
        """
        Computes and stores a variety of statistics and best-fit results.

        Parameters
        ----------
        spectrum_observed : np.ndarray
            A matrix of spectrum data, with dimenions that match those of the times
            and freqs arrays

        fit_result : scipy.optimize.OptimizeResult
            The output object from scipy.optimize.least_squares()

        Returns
        -------
        None : None
            The 'fit_statistics' attribute is defined as a Python dicitonary.
        """

        # pylint: disable=broad-except

        # compute various statistics of input data used for fit.
        num_freq, num_time = spectrum_observed.shape
        num_freq_good = int(np.sum(self.good_freq))
        num_fit_parameters = len(fit_result.x)

        self.fit_statistics["num_freq"] = num_freq
        self.fit_statistics["num_freq_good"] = num_freq_good
        self.fit_statistics["num_fit_parameters"] = num_fit_parameters
        self.fit_statistics["num_observations"] = num_freq_good * int(num_time) - num_fit_parameters
        self.fit_statistics["num_time"] = num_time

        # compute chisq values and the fitburst S/N.
        chisq_initial = float(np.sum((spectrum_observed * self.weights[:, None])**2))
        chisq_final = float(np.sum(fit_result.fun**2))
        chisq_final_reduced = chisq_final / self.fit_statistics["num_observations"]

        self.fit_statistics["chisq_initial"] = chisq_initial
        self.fit_statistics["chisq_final"] = chisq_final
        self.fit_statistics["chisq_final_reduced"] = chisq_final_reduced
        self.fit_statistics["snr"] = float(np.sqrt(chisq_initial - chisq_final))

        # now compute covarance matrix and parameter uncertainties.
        self.fit_statistics["bestfit_parameters"] = self.load_fit_parameters_list(
            fit_result.x.tolist())
        self.fit_statistics["bestfit_uncertainties"] = None
        self.fit_statistics["bestfit_covariance"] = None

        try:
            hessian = fit_result.jac.T.dot(fit_result.jac)
            covariance = np.linalg.inv(hessian) * chisq_final_reduced
            uncertainties = [float(x) for x in np.sqrt(np.diag(covariance)).tolist()]

            self.fit_statistics["bestfit_uncertainties"] = self.load_fit_parameters_list(
                uncertainties)
            self.fit_statistics["bestfit_covariance"] = None # return the full matrix at some point?

        except Exception as exc:
            print(f"ERROR: {exc}; designating fit as unsuccessful...")
            self.success = False

    def _set_weights(self, spectrum_observed: float) -> None:
        """
        Sets an attribute containing weights to be applied during least-squares fitting.

        Parameters
        ----------
        spectrum_observed : np.ndarray
            A matrix containing the dynamic spectrum to be analyzed for model fitting.

        Returns
        -------
        None : NoneType
            Two object attributes are defined and used for masking and weighting data during fit.
        """

        # compute RMS deviation for each channel.
        variance = np.mean(spectrum_observed**2, axis=1)
        std = np.sqrt(variance)
        bad_freq = np.logical_not(self.good_freq)

        # now compute statistical weights for "good" channels.
        self.weights = np.empty_like(std)

        if self.weighted_fit:
            self.weights[self.good_freq] = 1. / std[self.good_freq]

        else:
            self.weights[self.good_freq] = 1.

        # zero-weight bad channels.
        self.weights[bad_freq] = 0.
