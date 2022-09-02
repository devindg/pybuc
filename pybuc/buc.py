import numpy as np
from numpy import dot
from numpy.linalg import solve
from numba import njit
import matplotlib.pyplot as plt
from multiprocessing import Pool
import warnings
from typing import Union, NamedTuple
import pandas as pd

from .statespace.kalman_filter import kalman_filter as kf
from .statespace.durbin_koopman_smoother import dk_smoother as dks
from .utils import array_operations as ao
from .vectorized import distributions as dist


class Posterior(NamedTuple):
    num_samp: int
    smoothed_state: np.ndarray
    smoothed_errors: np.ndarray
    smoothed_prediction: np.ndarray
    filtered_state: np.ndarray
    filtered_prediction: np.ndarray
    response_variance: np.ndarray
    state_covariance: np.ndarray
    response_error_variance: np.ndarray
    state_error_covariance: np.ndarray
    regression_coefficients: np.ndarray = np.array([[[]]])


class ModelSetup(NamedTuple):
    components: dict
    response_var_scale_prior: float
    response_var_shape_post: np.ndarray
    state_var_scale_prior: np.ndarray
    state_var_shape_post: np.ndarray
    gibbs_iter0_init_state: np.ndarray
    gibbs_iter0_error_variances: np.ndarray
    init_state_plus_values: np.ndarray
    init_state_covariance: np.ndarray
    reg_coeff_mean_prior: np.ndarray = np.array([[]])
    reg_coeff_precision_prior: np.ndarray = np.array([[]])
    reg_coeff_precision_inv_prior: np.ndarray = np.array([[]])
    reg_coeff_precision_inv_post: np.ndarray = np.array([[]])
    reg_coeff_precision_post: np.ndarray = np.array([[]])
    gibbs_iter0_reg_coeff: np.ndarray = np.array([[]])


@njit
def set_seed(value):
    np.random.seed(value)


@njit(cache=True)
def _simulate_posterior_predictive_response(posterior: Posterior,
                                            burn: int = 0,
                                            num_first_obs_ignore: int = 0,
                                            random_sample_size_prop: float = 1.) -> np.ndarray:
    """

    :param posterior:
    :param burn:
    :param num_first_obs_ignore:
    :param random_sample_size_prop:
    :return:
    """

    response_mean = posterior.filtered_prediction[burn:, num_first_obs_ignore:, 0]
    response_variance = posterior.response_variance[burn:, num_first_obs_ignore:, 0]
    num_posterior_samp = response_mean.shape[0]
    n = response_mean.shape[1]

    if int(random_sample_size_prop * num_posterior_samp) > posterior.num_samp:
        raise ValueError('random_sample_size_prop must be between 0 and 1.')

    if int(random_sample_size_prop * num_posterior_samp) < 1:
        raise ValueError('random_sample_size_prop implies a sample with less than 1 observation. '
                         'Provide a random_sample_size_prop such that the number of samples '
                         'is at least 1 but no larger than the number of posterior samples.')

    if random_sample_size_prop == 1.:
        num_samp = num_posterior_samp
        S = list(np.arange(num_posterior_samp))
    else:
        num_samp = int(random_sample_size_prop * num_posterior_samp)
        S = list(np.random.choice(num_posterior_samp, num_samp, replace=False))

    y_post = np.empty((num_samp, n), dtype=np.float64)
    i = 0
    for s in S:
        y_post[i] = dist.vec_norm(response_mean[s],
                                  np.sqrt(response_variance[s][:, 0]))
        i += 1

    return y_post


def _simulate_posterior_predictive_state_worker(state_mean: np.ndarray,
                                                state_covariance: np.ndarray) -> np.ndarray:
    state_post = (np.random
                  .multivariate_normal(mean=state_mean,
                                       cov=state_covariance,
                                       tol=1e-6).reshape(-1, 1))

    return state_post


def _simulate_posterior_predictive_state(posterior: Posterior,
                                         burn: int = 0,
                                         num_first_obs_ignore: int = 0,
                                         random_sample_size_prop: float = 1.,
                                         has_predictors: bool = False) -> np.ndarray:
    """

    :param posterior:
    :param burn:
    :param num_first_obs_ignore:
    :param random_sample_size_prop:
    :param has_predictors:
    :return:
    """

    if has_predictors:
        mean = posterior.filtered_state[burn:, num_first_obs_ignore:-1, :-1, 0]
        cov = posterior.state_covariance[burn:, num_first_obs_ignore:-1, :-1, :-1]
    else:
        mean = posterior.filtered_state[burn:, num_first_obs_ignore:-1, :, 0]
        cov = posterior.state_covariance[burn:, num_first_obs_ignore:-1]

    num_posterior_samp = mean.shape[0]
    n = mean.shape[1]
    m = mean.shape[2]

    if int(random_sample_size_prop * num_posterior_samp) > posterior.num_samp:
        raise ValueError('random_sample_size_prop must be between 0 and 1.')

    if int(random_sample_size_prop * num_posterior_samp) < 1:
        raise ValueError('random_sample_size_prop implies a sample with less than 1 observation. '
                         'Provide a random_sample_size_prop such that the number of samples '
                         'is at least 1 but no larger than the number of posterior samples.')

    if random_sample_size_prop == 1.:
        num_samp = num_posterior_samp
        S = list(np.arange(num_posterior_samp))
    else:
        num_samp = int(random_sample_size_prop * num_posterior_samp)
        S = list(np.random.choice(num_posterior_samp, num_samp, replace=False))

    state_mean_args = [mean[i, j] for i in S for j in range(n)]
    state_covariance_args = [cov[i, j] for i in S for j in range(n)]
    with Pool() as pool:
        state_post = np.array(pool.starmap(_simulate_posterior_predictive_state_worker,
                                           zip(state_mean_args, state_covariance_args)))

    state_post = state_post.reshape((num_samp, n, m, 1))

    return state_post


@njit(cache=True)
def _forecast(posterior: Posterior,
              num_periods: int,
              state_observation_matrix: np.ndarray,
              state_transition_matrix: np.ndarray,
              state_error_transformation_matrix: np.ndarray,
              future_predictors: np.ndarray = np.array([[]]),
              burn: int = 0) -> np.ndarray:
    """

    :param posterior:
    :param num_periods:
    :param state_observation_matrix:
    :param state_transition_matrix:
    :param state_error_transformation_matrix:
    :param future_predictors:
    :param burn:
    :return:
    """

    Z = state_observation_matrix
    T = state_transition_matrix
    R = state_error_transformation_matrix
    X_fut = future_predictors
    m = R.shape[0]
    q = R.shape[1]

    response_error_variance = posterior.response_error_variance[burn:]
    if q > 0:
        state_error_covariance = posterior.state_error_covariance[burn:]

    smoothed_state = posterior.smoothed_state[burn:]
    num_samp = posterior.num_samp - burn

    if X_fut.size > 0:
        reg_coeff = posterior.regression_coefficients[burn:]

    y_forecast = np.empty((num_samp, num_periods, 1), dtype=np.float64)
    num_periods_zeros = np.zeros((num_periods, 1))
    num_periods_u_zeros = np.zeros((num_periods, q, 1))
    num_periods_ones = np.ones((num_periods, 1))
    num_periods_u_ones = np.ones((num_periods, q, 1))

    for s in range(num_samp):
        obs_error = dist.vec_norm(num_periods_zeros,
                                  num_periods_ones * np.sqrt(response_error_variance[s][0, 0]))
        if q > 0:
            state_error = dist.vec_norm(num_periods_u_zeros,
                                        num_periods_u_ones * np.sqrt(ao.diag_2d(state_error_covariance[s])))

        if X_fut.size > 0:
            Z[:, :, -1] = X_fut.dot(reg_coeff[s])

        alpha = np.empty((num_periods + 1, m, 1), dtype=np.float64)
        alpha[0] = smoothed_state[s, -1]
        for t in range(num_periods):
            y_forecast[s, t] = Z[t].dot(alpha[t]) + obs_error[t]
            if q > 0:
                alpha[t + 1] = T.dot(alpha[t]) + R.dot(state_error[t])
            else:
                alpha[t + 1] = T.dot(alpha[t])

    return y_forecast


class BayesianUnobservedComponents:
    def __init__(self,
                 response: Union[np.ndarray, pd.Series, pd.DataFrame],
                 predictors: Union[np.ndarray, pd.Series, pd.DataFrame] = np.array([[]]),
                 level: bool = False,
                 stochastic_level: bool = True,
                 slope: bool = False,
                 stochastic_slope: bool = True,
                 dummy_seasonal: tuple = (),
                 stochastic_dummy_seasonal: tuple = (),
                 trig_seasonal: tuple = (),
                 stochastic_trig_seasonal: tuple = (),
                 standardize: bool = False):

        """

        :param response: Numpy array, Pandas Series, or Pandas DataFrame, float64. Array that represents
        the response variable to modeled.

        :param predictors: Numpy array, Pandas Series, or Pandas DataFrame, float64. Array that represents
        the predictors, if any, to be used for predicting the response variable. Default is an empty array.

        :param level: bool. If true, a level component is added to the model. Default is false.

        :param stochastic_level: bool. If true, the level component evolves stochastically. Default is true.

        :param slope: bool. If true, a slope component is added to the model. Note, that a slope
        is applicable only when a level component is specified. Default is false.

        :param stochastic_slope: bool. If true, the slope component evolves stochastically. Default is true.

        :param dummy_seasonal: tuple of integers. Each integer in the tuple represents a distinct
        form of dummy seasonality/periodicity in the data. Default is an empty tuple, i.e., no
        dummy seasonality.

        :param stochastic_dummy_seasonal: tuple of bools. Each boolean in the tuple specifies whether the
        corresponding periodicities in dummy_seasonal evolve stochastically. Default is an empty tuple,
        which will be converted to true bools if dummy_seasonal is non-empty.

        :param trig_seasonal: tuple of 2-tuples that takes the form ((periodicity_1, num_harmonics_1),
        (periodicity_2, num_harmonics_2), ...). Each (periodicity, num_harmonics) pair in trig_seasonal
        specifies a periodicity and number of harmonics associated with the periodicity. For example,
        (12, 6) would specify a periodicity of 12 with 6 harmonics. The number of harmonics must be an
        integer in [1, periodicity / 2] if periodicity is even, or in [1, (periodicity - 1) / 2]
        if periodicity is odd. Each period specified must be distinct.

        :param stochastic_trig_seasonal: tuple of bools. Each boolean in the tuple specifies whether the
        corresponding (periodicity, num_harmonics) in trig_seasonal evolve stochastically. Default is an
        empty tuple, which will be converted to true bools if trig_seasonal is non-empty.

        :param standardize: bool. If true, the response is centered and scaled by its mean and standard
        deviation, respectively.
        """

        self.model_setup = None
        self.response_name = None
        self.predictors_names = None
        self.historical_time_index = None
        self.future_time_index = None
        self.num_first_obs_ignore = None
        self.posterior = None

        # TODO: Add functionality for autoregressive slope.

        # CHECK AND PREPARE RESPONSE DATA
        # -- data types, name, and index
        if not isinstance(response, (pd.Series, pd.DataFrame, np.ndarray)):
            raise ValueError("The response array must be a Numpy array, Pandas Series, or Pandas DataFrame.")
        else:
            response = response.copy()
            if isinstance(response, (pd.Series, pd.DataFrame)):
                if not isinstance(response.index, pd.DatetimeIndex):
                    raise ValueError("Pandas' DatetimeIndex is currently the only supported "
                                     "index type for Pandas objects.")
                else:
                    if response.index.freq is None:
                        warnings.warn('Frequency of DatetimeIndex is None. Frequency will be inferred for response.')
                        response.index.freq = pd.infer_freq(response.index)

                if isinstance(response, pd.Series):
                    self.response_name = [response.name]
                else:
                    self.response_name = response.columns.values.tolist()

                response = response.sort_index()
                self.historical_time_index = response.index
                response = response.to_numpy()

        # -- dimensions
        if response.dtype != 'float64':
            raise ValueError('All values in the response array must be of type float.')
        if response.ndim == 0:
            raise ValueError('The response array must have dimension 1 or 2.')
        if response.ndim == 1:
            response = response.reshape(-1, 1)
        if response.ndim > 2:
            raise ValueError('The response array must have dimension 1 or 2.')
        if response.ndim == 2:
            if all(i > 1 for i in response.shape):
                raise ValueError('The response array must have shape (1, n) or (n, 1), '
                                 'where n is the number of observations. Both the row and column '
                                 'count exceed 1.')
            else:
                response = response.reshape(-1, 1)

        # CHECK AND PREPARE PREDICTORS DATA, IF APPLICABLE
        # -- check if correct data type
        if not isinstance(predictors, (pd.Series, pd.DataFrame, np.ndarray)):
            raise ValueError("The predictors array must be a Numpy array, Pandas Series, or Pandas DataFrame.")
        else:
            if predictors.size > 0:
                predictors = predictors.copy()
                # -- check if response and predictors are same date type.
                if not isinstance(predictors, type(response)):
                    raise ValueError('Object types for response and predictors arrays must match.')

                # -- get predictor names if a Pandas object and sort index
                if isinstance(predictors, (pd.Series, pd.DataFrame)):
                    if not isinstance(predictors.index, pd.DatetimeIndex):
                        raise ValueError("Pandas' DatetimeIndex is currently the only supported "
                                         "index type for Pandas objects.")
                    else:
                        if not (predictors.index == self.historical_time_index).all():
                            raise ValueError('The response and predictors indexes must match.')

                    if isinstance(predictors, pd.Series):
                        self.predictors_names = [predictors.name]
                    else:
                        self.predictors_names = predictors.columns.values.tolist()

                    predictors = predictors.sort_index().to_numpy()

                # -- dimension and null/inf checks
                if predictors.dtype != 'float64':
                    raise ValueError('All values in the predictors array must be of type float.')
                if predictors.ndim == 0:
                    raise ValueError('The predictors array must have dimension 1 or 2. Dimension is 0.')
                if predictors.ndim == 1:
                    predictors = predictors.reshape(-1, 1)
                if predictors.ndim > 2:
                    raise ValueError('The predictors array must have dimension 1 or 2. Dimension exceeds 2.')
                if predictors.ndim == 2:
                    if 1 in predictors.shape:
                        predictors = predictors.reshape(-1, 1)
                if np.isnan(predictors).any():
                    raise ValueError('The predictors array cannot have null values.')
                if np.isinf(predictors).any():
                    raise ValueError('The predictors array cannot have Inf and/or -Inf values.')

                # -- conformable number of observations
                if predictors.shape[0] != response.shape[0]:
                    raise ValueError('The number of observations in the predictors array must match '
                                     'the number of observations in the response array.')

                # -- warn about model stability if the number of predictors exceeds number of observations
                if predictors.shape[1] > predictors.shape[0]:
                    warnings.warn('The number of predictors exceeds the number of observations. '
                                  'Results will be sensitive to choice of priors.')

        # CHECK AND PREPARE DUMMY SEASONAL
        if not isinstance(dummy_seasonal, tuple):
            raise ValueError('dummy_seasonal must be a tuple.')

        if not isinstance(stochastic_dummy_seasonal, tuple):
            raise ValueError('stochastic_dummy_seasonal must be a tuple.')

        if len(dummy_seasonal) > 0:
            if not all(isinstance(v, int) for v in dummy_seasonal):
                raise ValueError('The period for a dummy seasonal component must be an integer.')

            if len(dummy_seasonal) != len(set(dummy_seasonal)):
                raise ValueError('Each specified period in dummy_seasonal must be distinct.')

            if any(v < 2 for v in dummy_seasonal):
                raise ValueError('The period for a dummy seasonal component must be an integer greater than 1.')

            if len(stochastic_dummy_seasonal) > 0:
                if not all(isinstance(v, bool) for v in stochastic_dummy_seasonal):
                    raise ValueError('If an non-empty tuple is passed for the stochastic specification '
                                     'of the dummy seasonal components, all elements must be of boolean type.')

            if len(dummy_seasonal) > len(stochastic_dummy_seasonal):
                if len(stochastic_dummy_seasonal) > 0:
                    raise ValueError('Some of the dummy seasonal components were given a stochastic '
                                     'specification, but not all. Partial specification of the stochastic '
                                     'profile is not allowed. Either leave the stochastic specification blank '
                                     'by passing an empty tuple (), which will default to True '
                                     'for all components, or pass a stochastic specification '
                                     'for each seasonal component.')

            if len(dummy_seasonal) < len(stochastic_dummy_seasonal):
                raise ValueError('The tuple which specifies the number of stochastic dummy seasonal components '
                                 'has greater length than the tuple that specifies the number of dummy seasonal '
                                 'components. Either pass a blank tuple () for the stochastic profile, or a '
                                 'boolean tuple of same length as the tuple that specifies the number of '
                                 'dummy seasonal components.')

        else:
            if len(stochastic_dummy_seasonal) > 0:
                warnings.warn('No dummy seasonal components were specified, but a non-empty '
                              'stochastic profile was passed for dummy seasonality. If dummy '
                              'seasonal components are desired, specify the period for each '
                              'component via a tuple passed to the dummy_seasonal argument. '
                              'Otherwise, the stochastic profile for dummy seasonality will '
                              'be treated as inadvertent and ignored.')

        # CHECK AND PREPARE TRIGONOMETRIC SEASONAL
        if not isinstance(trig_seasonal, tuple):
            raise ValueError('trig_seasonal must be a tuple.')

        if not isinstance(stochastic_trig_seasonal, tuple):
            raise ValueError('stochastic_trig_seasonal must be a tuple.')

        if len(trig_seasonal) > 0:
            if not all(isinstance(v, tuple) for v in trig_seasonal):
                raise ValueError('Each element in trig_seasonal must be a tuple.')

            if not all(len(v) == 2 for v in trig_seasonal):
                raise ValueError('A (period, num_harmonics) tuple must be provided for each specified trigonometric '
                                 'seasonal component.')

            if not all(isinstance(v[0], int) for v in trig_seasonal):
                raise ValueError('The period for a specified trigonometric seasonal component must be an integer.')

            if not all(isinstance(v[1], int) for v in trig_seasonal):
                raise ValueError('The number of harmonics for a specified trigonometric seasonal component must '
                                 'be an integer.')

            if any(v[0] < 2 for v in trig_seasonal):
                raise ValueError('The period for a trigonometric seasonal component must be an integer greater than 1.')

            if any(v[1] < 1 and v[1] != 0 for v in trig_seasonal):
                raise ValueError('The number of harmonics for a trigonometric seasonal component can take 0 or '
                                 'integers at least as large as 1 as valid options. A value of 0 will enforce '
                                 'the highest possible number of harmonics for the given period, which is period / 2 '
                                 'if period is even, or (period - 1) / 2 if period is odd.')

            periodicities = []
            for v in trig_seasonal:
                period, num_harmonics = v
                periodicities.append(period)
                if ao.is_odd(period):
                    if num_harmonics > int(period - 1) / 2:
                        raise ValueError('The number of harmonics for a trigonometric seasonal component cannot '
                                         'exceed (period - 1) / 2 when period is odd.')
                else:
                    if num_harmonics > int(period / 2):
                        raise ValueError('The number of harmonics for a trigonometric seasonal component cannot '
                                         'exceed period / 2 when period is even.')

            if len(trig_seasonal) != len(set(periodicities)):
                raise ValueError('Each specified period in trig_seasonal must be distinct.')

            if len(stochastic_trig_seasonal) > 0:
                if not all(isinstance(v, bool) for v in stochastic_trig_seasonal):
                    raise ValueError('If an non-empty tuple is passed for the stochastic specification '
                                     'of the trigonometric seasonal components, all elements must be of boolean type.')

            if len(trig_seasonal) > len(stochastic_trig_seasonal):
                if len(stochastic_trig_seasonal) > 0:
                    raise ValueError('Some of the trigonometric seasonal components '
                                     'were given a stochastic specification, but not all. '
                                     'Partial specification of the stochastic profile is not '
                                     'allowed. Either leave the stochastic specification blank '
                                     'by passing an empty tuple (), which will default to True '
                                     'for all components, or pass a stochastic specification '
                                     'for each seasonal component.')

            if len(trig_seasonal) < len(stochastic_trig_seasonal):
                raise ValueError('The tuple which specifies the number of stochastic trigonometric '
                                 'seasonal components has greater length than the tuple that specifies '
                                 'the number of trigonometric seasonal components. Either pass a blank '
                                 'tuple () for the stochastic profile, or a boolean tuple of same length '
                                 'as the tuple that specifies the number of trigonometric seasonal components.')

        else:
            if len(stochastic_trig_seasonal) > 0:
                warnings.warn('No trigonometric seasonal components were specified, but a non-empty '
                              'stochastic profile was passed for trigonometric seasonality. If trigonometric '
                              'seasonal components are desired, specify the period for each '
                              'component via a tuple passed to the trig_seasonal argument. '
                              'Otherwise, the stochastic profile for trigonometric seasonality will '
                              'be treated as inadvertent and ignored.')

        # FINAL VALIDITY CHECKS
        if not isinstance(level, bool) or not isinstance(stochastic_level, bool):
            raise ValueError('level and stochastic_level must be of boolean type.')

        if not isinstance(slope, bool) or not isinstance(stochastic_slope, bool):
            raise ValueError('slope and stochastic_slope must be of boolean type.')

        if len(dummy_seasonal) == 0 and len(trig_seasonal) == 0 and not level:
            raise ValueError('At least a level or seasonal component must be specified.')

        if slope and not level:
            raise ValueError('Slope cannot be specified without a level component.')

        # ASSIGN CLASS ATTRIBUTES IF ALL VALIDITY CHECKS ARE PASSED
        self.response = response
        self.predictors = predictors
        self.level = level
        self.stochastic_level = stochastic_level
        self.slope = slope
        self.stochastic_slope = stochastic_slope
        self.standardize = standardize
        self.dummy_seasonal = dummy_seasonal
        self.stochastic_dummy_seasonal = stochastic_dummy_seasonal
        self.trig_seasonal = trig_seasonal
        self.stochastic_trig_seasonal = stochastic_trig_seasonal

        if len(dummy_seasonal) > 0 and len(stochastic_dummy_seasonal) == 0:
            self.stochastic_dummy_seasonal = (True,) * len(dummy_seasonal)

        if len(trig_seasonal) > 0:
            ts = ()
            for c, v in enumerate(trig_seasonal):
                period, num_harmonics = v
                if num_harmonics == 0:
                    if ao.is_odd(num_harmonics):
                        h = int((period - 1) / 2)
                    else:
                        h = int(period / 2)
                else:
                    h = num_harmonics

                v_updated = (period, h)
                ts = ts + (v_updated,)

            self.trig_seasonal = ts

            if len(stochastic_trig_seasonal) == 0:
                self.stochastic_trig_seasonal = (True,) * len(trig_seasonal)

        if self.historical_time_index is None:
            self.historical_time_index = np.arange(response.shape[0])

        if self.has_predictors:
            if self.predictors_names is None:
                self.predictors_names = [f"x{i + 1}" for i in range(self.num_predictors)]

    @property
    def num_dummy_seasonal_state_eqs(self) -> int:
        if len(self.dummy_seasonal) == 0:
            return 0
        else:
            num_eqs = 0
            for v in self.dummy_seasonal:
                num_eqs += v - 1

        return num_eqs

    @property
    def num_stochastic_dummy_seasonal_state_eqs(self) -> int:
        num_stochastic = 0
        for c, v in enumerate(self.dummy_seasonal):
            num_stochastic += 1 * self.stochastic_dummy_seasonal[c]

        return num_stochastic

    @property
    def num_trig_seasonal_state_eqs(self) -> int:
        if len(self.trig_seasonal) == 0:
            return 0
        else:
            num_eqs = 0
            for v in self.trig_seasonal:
                _, freq = v
                num_eqs += 2 * freq

        return num_eqs

    @property
    def num_stochastic_trig_seasonal_state_eqs(self) -> int:
        num_stochastic = 0
        for c, v in enumerate(self.trig_seasonal):
            _, freq = v
            num_stochastic += 2 * freq * self.stochastic_trig_seasonal[c]

        return num_stochastic

    @property
    def num_seasonal_state_eqs(self) -> int:
        return self.num_dummy_seasonal_state_eqs + self.num_trig_seasonal_state_eqs

    @property
    def num_stochastic_seasonal_state_eqs(self) -> int:
        return self.num_stochastic_dummy_seasonal_state_eqs + self.num_stochastic_trig_seasonal_state_eqs

    @property
    def has_predictors(self) -> bool:
        if self.predictors.size > 0:
            return True
        else:
            return False

    @property
    def num_predictors(self) -> int:
        if self.predictors.size == 0:
            return 0
        else:
            return self.predictors.shape[1]

    @property
    def num_state_eqs(self) -> int:
        return ((self.level + self.slope) * 1
                + self.num_seasonal_state_eqs
                + self.has_predictors * 1)

    @property
    def num_stochastic_states(self) -> int:
        return ((self.level * self.stochastic_level
                 + self.slope * self.stochastic_slope) * 1
                + self.num_stochastic_seasonal_state_eqs)

    @property
    def num_obs(self) -> int:
        return self.response.shape[0]

    @property
    def mean_response(self) -> np.ndarray:
        return np.nanmean(self.response)

    @property
    def sd_response(self) -> np.ndarray:
        return np.nanstd(self.response)

    @property
    def z_response(self) -> np.ndarray:
        return (self.response - self.mean_response) / self.sd_response

    @property
    def y(self) -> np.ndarray:
        if self.standardize:
            return self.z_response
        else:
            return self.response

    @staticmethod
    def trig_transition_matrix(freq: int) -> np.ndarray:
        real_part = np.array([[np.cos(freq), np.sin(freq)]])
        imaginary_part = np.array([[-np.sin(freq), np.cos(freq)]])
        return np.concatenate((real_part, imaginary_part), axis=0)

    def observation_matrix(self,
                           num_rows: int = 0) -> np.ndarray:
        if num_rows == 0:
            num_rows = self.num_obs
        m = self.num_state_eqs
        Z = np.zeros((num_rows, 1, m), dtype=np.float64)

        # Note that if a regression component is specified, the observation matrix
        # will vary with time (assuming the predictors vary with time).
        # At this stage of the Kalman Filter setup, however, the regression
        # component in the observation matrix will be set to 0 for each observation.
        # The 0's will be reassigned based on the prior and posterior means for
        # the regression coefficients.

        j = 0
        if self.level:
            Z[:, :, j] = 1.
            j += 1
        if self.slope:
            j += 1
        if len(self.dummy_seasonal) > 0:
            i = j
            for v in self.dummy_seasonal:
                Z[:, :, i] = 1.
                i += v - 1
            j += self.num_dummy_seasonal_state_eqs
        if len(self.trig_seasonal) > 0:
            Z[:, :, j::2] = 1.
            j += self.num_trig_seasonal_state_eqs
        if self.has_predictors:
            Z[:, :, j] = 0.

        return Z

    @property
    def state_transition_matrix(self) -> np.ndarray:
        m = self.num_state_eqs
        T = np.zeros((m, m), dtype=np.float64)

        i, j = 0, 0
        if self.level:
            T[i, j] = 1.
            i += 1
            j += 1
        if self.slope:
            T[i - 1, j] = 1.
            T[i, j] = 1.
            i += 1
            j += 1
        if len(self.dummy_seasonal) > 0:
            for v in self.dummy_seasonal:
                T[i, j:j + v - 1] = -1.
                for k in range(1, v - 1):
                    T[i + k, j + k - 1] = 1.
                i += v - 1
                j += v - 1
        if len(self.trig_seasonal) > 0:
            for v in self.trig_seasonal:
                period, freq = v
                for k in range(1, freq + 1):
                    T[i:i + 2, j:j + 2] = self.trig_transition_matrix(2. * np.pi * k / period)
                    i += 2
                    j += 2
        if self.has_predictors:
            T[i, j] = 1.

        return T

    @property
    def state_error_transformation_matrix(self) -> np.ndarray:
        m = self.num_state_eqs
        q = self.num_stochastic_states
        R = np.zeros((m, q), dtype=np.float64)

        if q == 0:
            pass
        else:
            i, j = 0, 0
            if self.level:
                if self.stochastic_level:
                    R[i, j] = 1.
                    j += 1
                i += 1
            if self.slope:
                if self.stochastic_slope:
                    R[i, j] = 1.
                    j += 1
                i += 1
            if len(self.dummy_seasonal) > 0:
                for c, v in enumerate(self.dummy_seasonal):
                    if self.stochastic_dummy_seasonal[c]:
                        R[i, j] = 1.
                        j += 1
                    i += v - 1
            if len(self.trig_seasonal) > 0:
                for c, v in enumerate(self.trig_seasonal):
                    _, freq = v
                    num_terms = 2 * freq
                    if self.stochastic_trig_seasonal[c]:
                        for k in range(num_terms):
                            R[i + k, j + k] = 1.
                        j += num_terms
                    i += num_terms

        return R

    @property
    def posterior_state_error_covariance_transformation_matrix(self) -> np.ndarray:
        q = self.num_stochastic_states
        A = np.zeros((q, q), dtype=np.float64)

        if q == 0:
            pass
        else:
            if self.num_stochastic_trig_seasonal_state_eqs == 0:
                np.fill_diagonal(A, 1.)
            else:
                i = 0
                if self.level:
                    if self.stochastic_level:
                        A[i, i] = 1.
                        i += 1
                if self.slope:
                    if self.stochastic_slope:
                        A[i, i] = 1.
                        i += 1
                if len(self.dummy_seasonal) > 0:
                    for c, v in enumerate(self.dummy_seasonal):
                        if self.stochastic_dummy_seasonal[c]:
                            A[i, i] = 1.
                            i += 1
                if len(self.trig_seasonal) > 0:
                    for c, v in enumerate(self.trig_seasonal):
                        _, freq = v
                        num_terms = 2 * freq
                        if self.stochastic_trig_seasonal[c]:
                            for k in range(num_terms):
                                A[i + k, i:i + num_terms] = 1. / (2 * freq)
                            i += 2 * freq

        return A

    @staticmethod
    def _first_value(x: np.ndarray):
        index = np.nonzero(~np.isnan(x))
        value = x[index][0]
        return value, index

    @staticmethod
    def _last_value(x: np.ndarray):
        x_flip = np.flip(x)
        index = np.nonzero(~np.isnan(x_flip))
        value = x_flip[index][0]
        return value, index

    def _gibbs_iter0_init_level(self):
        if self.level:
            first_y = self._first_value(self.y)[0]
            return dist.vec_norm(first_y, np.nanstd(self.y))
        else:
            return

    def _gibbs_iter0_init_slope(self):
        if self.slope:
            first_y = self._first_value(self.y)
            last_y = self._last_value(self.y)
            num_steps = (self.y.size - last_y[1][0][0] - 1) - first_y[1][0][0]
            if num_steps == 0:
                slope = 0.
            else:
                slope = (last_y[0] - first_y[0]) / num_steps
            return dist.vec_norm(slope, np.nanstd(self.y))
        else:
            return

    def _gibbs_iter0_init_dummy_seasonal(self):
        if len(self.dummy_seasonal) > 0:
            num_eqs = self.num_dummy_seasonal_state_eqs
            return dist.vec_norm(np.zeros(num_eqs), np.ones(num_eqs) * np.nanstd(self.y))
        else:
            return

    def _gibbs_iter0_init_trig_seasonal(self):
        if len(self.trig_seasonal) > 0:
            num_eqs = self.num_trig_seasonal_state_eqs
            return dist.vec_norm(np.zeros(num_eqs), np.ones(num_eqs) * np.nanstd(self.y))
        else:
            return

    def _model_setup(self,
                     response_var_shape_prior: float, response_var_scale_prior: float,
                     level_var_shape_prior: float, level_var_scale_prior: float,
                     slope_var_shape_prior: float, slope_var_scale_prior: float,
                     dum_season_var_shape_prior: tuple, dum_season_var_scale_prior: tuple,
                     trig_season_var_shape_prior: tuple, trig_season_var_scale_prior: tuple,
                     reg_coeff_mean_prior: np.ndarray, reg_coeff_precision_prior: np.ndarray) -> ModelSetup:

        n = self.num_obs

        # Create list that will capture what components are specified.
        # This will be used for plotting. The irregular component
        # will always be a part of the model (for now).
        components = dict()
        components['Irregular'] = dict()

        # Get priors for specified components
        if np.isnan(response_var_shape_prior):
            response_var_shape_prior = 0.001
        if np.isnan(response_var_scale_prior):
            response_var_scale_prior = 0.001

        response_var_shape_post = np.array([[response_var_shape_prior + 0.5 * n]])
        gibbs_iter0_response_error_var = [0.01 * self.sd_response ** 2]

        state_var_scale_prior = []
        state_var_shape_post = []
        init_state_variances = []
        init_state_plus_values = []
        gibbs_iter0_state_error_var = []
        gibbs_iter0_init_state = []
        seasonal_period = (0,)
        j = 0
        if self.level:
            if self.stochastic_level:
                if np.isnan(level_var_shape_prior):
                    level_var_shape_prior = 0.001

                if np.isnan(level_var_scale_prior):
                    level_var_scale_prior = 0.001

                state_var_shape_post.append(level_var_shape_prior + 0.5 * n)
                state_var_scale_prior.append(level_var_scale_prior)
                gibbs_iter0_init_state.append(self._gibbs_iter0_init_level())
                gibbs_iter0_state_error_var.append(0.01 * self.sd_response ** 2)

            # The level state equation represents a random walk.
            # Diffuse initialization is required in this case.
            # The fake states used in Durbin-Koopman for simulating
            # the smoothed states can take any arbitrary value
            # under diffuse initialization. Zero is chosen.
            # For the initial state covariance matrix, an
            # approximate diffuse initialization method is adopted.
            # That is, a diagonal matrix with large values along
            # the diagonal. Large in this setting is defined as 1e6.
            init_state_plus_values.append(0.)
            init_state_variances.append(1e6)
            components['Level'] = dict(start_obs_mat_col_index=j,
                                       end_obs_mat_col_index=j + 1,
                                       state_error_cov_diag_index=j,
                                       stochastic=self.stochastic_level)
            j += 1

        if self.slope:
            if self.stochastic_slope:
                if np.isnan(slope_var_shape_prior):
                    slope_var_shape_prior = 0.001

                if np.isnan(slope_var_scale_prior):
                    slope_var_scale_prior = 0.001

                state_var_shape_post.append(slope_var_shape_prior + 0.5 * n)
                state_var_scale_prior.append(slope_var_scale_prior)
                gibbs_iter0_init_state.append(self._gibbs_iter0_init_slope())
                gibbs_iter0_state_error_var.append(0.01 * self.sd_response ** 2)

            components['Trend'] = dict(state_error_cov_diag_index=j,
                                       stochastic=self.stochastic_slope)

            # The slope state equation represents a random walk.
            # Diffuse initialization is required in this case.
            # The fake states used in Durbin-Koopman for simulating
            # the smoothed states can take any arbitrary value
            # under diffuse initialization. Zero is chosen.
            # For the initial state covariance matrix, an
            # approximate diffuse initialization method is adopted.
            # That is, a diagonal matrix with large values along
            # the diagonal. Large in this setting is defined as 1e6.
            init_state_plus_values.append(0.)
            init_state_variances.append(1e6)
            j += 1

        if len(self.dummy_seasonal) > 0:
            if True in self.stochastic_dummy_seasonal:
                if len(dum_season_var_shape_prior) == 0:
                    dum_season_var_shape_prior = (0.001,) * len(self.dummy_seasonal)

                if len(dum_season_var_scale_prior) == 0:
                    dum_season_var_scale_prior = (0.001,) * len(self.dummy_seasonal)

            i = j
            for c, v in enumerate(self.dummy_seasonal):
                seasonal_period = seasonal_period + (v,)
                if self.stochastic_dummy_seasonal[c]:
                    state_var_shape_post.append(dum_season_var_shape_prior[c] + 0.5 * n)
                    state_var_scale_prior.append(dum_season_var_scale_prior[c])
                    gibbs_iter0_state_error_var.append(0.01 * self.sd_response ** 2)

                # The dummy seasonal state equations represent random walks.
                # Diffuse initialization is required in this case.
                # The fake states used in Durbin-Koopman for simulating
                # the smoothed states can take any arbitrary value
                # under diffuse initialization. Zero is chosen.
                # For the initial state covariance matrix, an
                # approximate diffuse initialization method is adopted.
                # That is, a diagonal matrix with large values along
                # the diagonal. Large in this setting is defined as 1e6.
                for k in range(v - 1):
                    init_state_plus_values.append(0.)
                    init_state_variances.append(1e6)

                components[f'Dummy-Seasonal.{v}'] = dict(start_obs_mat_col_index=i,
                                                         end_obs_mat_col_index=i + (v - 1) + 1,
                                                         state_error_cov_diag_index=i,
                                                         stochastic=self.stochastic_dummy_seasonal[c])
                i += v - 1

            for k in self._gibbs_iter0_init_dummy_seasonal():
                gibbs_iter0_init_state.append(k)

            j += self.num_dummy_seasonal_state_eqs

        if len(self.trig_seasonal) > 0:
            if True in self.stochastic_trig_seasonal:
                if len(trig_season_var_shape_prior) == 0:
                    trig_season_var_shape_prior = (0.001,) * len(self.trig_seasonal)

                if len(trig_season_var_scale_prior) == 0:
                    trig_season_var_scale_prior = (0.001,) * len(self.trig_seasonal)

            i = j
            for c, v in enumerate(self.trig_seasonal):
                f, h = v
                num_terms = 2 * h
                seasonal_period = seasonal_period + (f,)
                if self.stochastic_trig_seasonal[c]:
                    for k in range(num_terms):
                        state_var_shape_post.append(trig_season_var_shape_prior[c] + 0.5 * n)
                        state_var_scale_prior.append(trig_season_var_scale_prior[c])
                        gibbs_iter0_state_error_var.append(0.01 * self.sd_response ** 2)

                # The trigonometric seasonal state equations represent random walks.
                # Diffuse initialization is required in this case.
                # The fake states used in Durbin-Koopman for simulating
                # the smoothed states can take any arbitrary value
                # under diffuse initialization. Zero is chosen.
                # For the initial state covariance matrix, an
                # approximate diffuse initialization method is adopted.
                # That is, a diagonal matrix with large values along
                # the diagonal. Large in this setting is defined as 1e6.
                for k in range(num_terms):
                    init_state_plus_values.append(0.)
                    init_state_variances.append(1e6)

                components[f'Trigonometric-Seasonal.{f}.{h}'] = dict(start_obs_mat_col_index=i,
                                                                     end_obs_mat_col_index=i + num_terms + 1,
                                                                     state_error_cov_diag_index=i,
                                                                     stochastic=self.stochastic_trig_seasonal[c])
                i += 2 * h

            for k in self._gibbs_iter0_init_trig_seasonal():
                gibbs_iter0_init_state.append(k)

            j += self.num_trig_seasonal_state_eqs

        if self.has_predictors:
            components['Regression'] = dict()
            X = self.predictors
            init_state_plus_values.append(0.)
            init_state_variances.append(0.)
            gibbs_iter0_init_state.append(1.)

            if reg_coeff_mean_prior.size == 0:
                print('No mean prior was provided for the regression coefficient vector. '
                      'A 0-mean prior will be enforced.')
                reg_coeff_mean_prior = np.zeros((self.num_predictors, 1))

            if reg_coeff_precision_prior.size == 0:
                kappa = 1.
                g = kappa / n
                print('No precision prior was provided for the regression coefficient vector. '
                      'A g=1/n Zellner g-prior will be enforced.')
                reg_coeff_precision_prior = g * (0.5 * dot(X.T, X) + 0.5 * np.diag(np.diag(dot(X.T, X))))
                reg_coeff_precision_inv_prior = solve(reg_coeff_precision_prior, np.eye(self.num_predictors))
            else:
                reg_coeff_precision_inv_prior = solve(reg_coeff_precision_prior, np.eye(self.num_predictors))

            reg_coeff_precision_post = dot(X.T, X) + reg_coeff_precision_inv_prior
            reg_coeff_precision_inv_post = solve(reg_coeff_precision_post, np.eye(self.num_predictors))
            gibbs_iter0_reg_coeff = dot(reg_coeff_precision_inv_post,
                                        (dot(X.T, self.y) + dot(reg_coeff_precision_inv_prior, reg_coeff_mean_prior)))

        state_var_shape_post = np.vstack(state_var_shape_post)
        state_var_scale_prior = np.vstack(state_var_scale_prior)
        gibbs_iter0_init_state = np.vstack(gibbs_iter0_init_state)
        gibbs_iter0_error_variances = np.concatenate((gibbs_iter0_response_error_var, gibbs_iter0_state_error_var))
        init_state_plus_values = np.vstack(init_state_plus_values)
        init_state_covariance = np.diag(init_state_variances)
        self.num_first_obs_ignore = max(1 + max(seasonal_period), self.num_state_eqs)

        if self.has_predictors:
            self.model_setup = ModelSetup(components,
                                          response_var_scale_prior,
                                          response_var_shape_post,
                                          state_var_scale_prior,
                                          state_var_shape_post,
                                          gibbs_iter0_init_state,
                                          gibbs_iter0_error_variances,
                                          init_state_plus_values,
                                          init_state_covariance,
                                          reg_coeff_mean_prior,
                                          reg_coeff_precision_prior,
                                          reg_coeff_precision_inv_prior,
                                          reg_coeff_precision_inv_post,
                                          reg_coeff_precision_post,
                                          gibbs_iter0_reg_coeff)
        else:
            self.model_setup = ModelSetup(components,
                                          response_var_scale_prior,
                                          response_var_shape_post,
                                          state_var_scale_prior,
                                          state_var_shape_post,
                                          gibbs_iter0_init_state,
                                          gibbs_iter0_error_variances,
                                          init_state_plus_values,
                                          init_state_covariance)

        return self.model_setup

    def sample(self,
               num_samp: int,
               response_var_shape_prior: float = np.nan, response_var_scale_prior: float = np.nan,
               level_var_shape_prior: float = np.nan, level_var_scale_prior: float = np.nan,
               slope_var_shape_prior: float = np.nan, slope_var_scale_prior: float = np.nan,
               dum_season_var_shape_prior: tuple = (), dum_season_var_scale_prior: tuple = (),
               trig_season_var_shape_prior: tuple = (), trig_season_var_scale_prior: tuple = (),
               reg_coeff_mean_prior: np.ndarray = np.array([[]]), reg_coeff_precision_prior: np.ndarray = np.array([[]])
               ) -> Posterior:

        """

        :param num_samp: integer > 0. Specifies the number of posterior samples to draw.

        :param response_var_shape_prior: float > 0. Specifies the inverse-Gamma shape prior for the
        response error variance. Default is 0.001.

        :param response_var_scale_prior: float > 0. Specifies the inverse-Gamma scale prior for the
        response error variance. Default is 0.001.

        :param level_var_shape_prior: float > 0. Specifies the inverse-Gamma shape prior for the
        level state equation error variance. Default is 0.001.

        :param level_var_scale_prior: float > 0. Specifies the inverse-Gamma scale prior for the
        level state equation error variance. Default is 0.001.

        :param slope_var_shape_prior: float > 0. Specifies the inverse-Gamma shape prior for the
        slope state equation error variance. Default is 0.001.

        :param slope_var_scale_prior: float > 0. Specifies the inverse-Gamma scale prior for the
        slope state equation error variance. Default is 0.001.

        :param dum_season_var_shape_prior: tuple of floats > 0. Specifies the inverse-Gamma shape priors
        for each periodicity in dummy_seasonal. Default is 0.001 for each periodicity.

        :param dum_season_var_scale_prior: tuple of floats > 0. Specifies the inverse-Gamma scale priors
        for each periodicity in dummy_seasonal. Default is 0.001 for each periodicity.

        :param trig_season_var_shape_prior: tuple of floats > 0. Specifies the inverse-Gamma shape priors
        for each (periodicity, num_harmonics) pair in trig_seasonal. In total, there are
        2 * SUM[num_harmonics[p]] shape priors for p=1, 2, ..., P periodicities. For example, if
        trig_seasonal = ((12, 3), (10, 2)) and stochastic_trig_seasonal = (True, True), then there are
        2 * 3 + 2 * 2 = 10 shape priors required. Default is 0.001 for each (periodicity, num_harmonics) pair.

        :param trig_season_var_scale_prior: tuple of floats > 0. Specifies the inverse-Gamma scale priors
        for each (periodicity, num_harmonics) pair in trig_seasonal. In total, there are
        2 * SUM[num_harmonics[p]] scale priors for p=1, 2, ..., P periodicities. For example, if
        trig_seasonal = ((12, 3), (10, 2)) and stochastic_trig_seasonal = (True, True), then there are
        2 * 3 + 2 * 2 = 10 scale priors required. Default is 0.001 for each (periodicity, num_harmonics) pair.

        :param reg_coeff_mean_prior: Numpy array of dimension (k, 1), where k is the number of predictors.
        Data type must be float64. If predictors are specified without a mean prior, a k-dimensional zero
        vector will be assumed.

        :param reg_coeff_precision_prior: Numpy array of dimension (k, k), where k is the number of predictors.
        Data type must be float64. If predictors are specified without a precision prior, Zellner's g-prior
        will be assumed. That is, the covariance matrix will take the form
        g * (0.5 * dot(X.T, X) + 0.5 * np.diag(dot(X.T, X))), where g = 1 / n and X is the predictor matrix.

        :return: NamedTuple with the following:
        
                    num_samp: Number of posterior samples drawn
                    smoothed_state: Posterior simulated smoothed state from Durbin-Koopman smoother
                    smoothed_errors: Posterior simulated smoothed errors from Durbin-Koopman smoother
                    smoothed_prediction: Posterior prediction for response based on smoothed state
                    filtered_state: Posterior filtered state from Kalman filter
                    filtered_prediction: Posterior prediction for response based on filtered state
                    response_variance: Posterior variance of the response from the Kalman filter
                    state_covariance: Posterior variance-covariance matrix for the state vector
                    response_error_variance: Posterior variance of the response equation's error term
                    state_error_covariance: Posterior variance-covariance matrix for the state error vector
                    regression_coefficients: Posterior regression coefficients
        """

        if not isinstance(num_samp, int):
            raise ValueError('num_samp must be of type float.')
        else:
            if not num_samp > 0:
                raise ValueError('num_samp must a strictly positive integer.')

        # Response prior check
        if not isinstance(response_var_shape_prior, float):
            raise ValueError('response_var_shape_prior must be of type float.')
        else:
            if not np.isnan(response_var_shape_prior):
                if not response_var_shape_prior > 0:
                    raise ValueError('response_var_shape_prior must be a strictly positive float.')

        if not isinstance(response_var_scale_prior, float):
            raise ValueError('response_var_scale_prior must be of type float.')
        else:
            if not np.isnan(response_var_scale_prior):
                if not response_var_scale_prior > 0:
                    raise ValueError('response_var_scale_prior must be a strictly positive float.')

        # Level prior check
        if not isinstance(level_var_shape_prior, float):
            raise ValueError('level_var_shape_prior must be of type float.')
        else:
            if not np.isnan(level_var_shape_prior):
                if not level_var_shape_prior > 0:
                    raise ValueError('level_var_shape_prior must be a strictly positive float.')

        if not isinstance(level_var_scale_prior, float):
            raise ValueError('level_var_scale_prior must be of type float.')
        else:
            if not np.isnan(level_var_scale_prior):
                if not level_var_scale_prior > 0:
                    raise ValueError('level_var_scale_prior must be a strictly positive float.')

        # Slope variance check
        if not isinstance(slope_var_shape_prior, float):
            raise ValueError('slope_var_shape_prior must be of type float.')
        else:
            if not np.isnan(slope_var_shape_prior):
                if not slope_var_shape_prior > 0:
                    raise ValueError('slope_var_shape_prior must be a strictly positive float.')

        if not isinstance(slope_var_scale_prior, float):
            raise ValueError('slope_var_scale_prior must be of type float.')
        else:
            if not np.isnan(slope_var_scale_prior):
                if not slope_var_scale_prior > 0:
                    raise ValueError('slope_var_scale_prior must be a strictly positive float.')

        # Dummy seasonal prior check
        if not isinstance(dum_season_var_shape_prior, tuple):
            raise ValueError('dum_seasonal_var_shape_prior must be a tuple of floats'
                             'to accommodate potentially multiple seasonality.')
        else:
            if len(dum_season_var_shape_prior) > 0:
                if len(dum_season_var_shape_prior) != len(self.dummy_seasonal):
                    raise ValueError('A non-empty tuple for dum_season_var_shape_prior must have the same '
                                     'length as dummy_seasonal. That is, for each periodicity in dummy_seasonal, '
                                     'there must be a corresponding shape prior.')
                if not all(isinstance(i, float) for i in dum_season_var_shape_prior):
                    raise ValueError('All values in dum_season_var_shape_prior must be of type float.')
                if not all(i > 0 for i in dum_season_var_shape_prior):
                    raise ValueError('All values in dum_season_var_shape_prior must be strictly positive floats.')

        if not isinstance(dum_season_var_scale_prior, tuple):
            raise ValueError('dum_seasonal_var_scale_prior must be of type tuple '
                             'to account for multiple seasonality.')
        else:
            if len(dum_season_var_scale_prior) > 0:
                if len(dum_season_var_scale_prior) != len(self.dummy_seasonal):
                    raise ValueError('A non-empty tuple for dum_season_var_scale_prior must have the same '
                                     'length as dummy_seasonal. That is, for each periodicity in dummy_seasonal, '
                                     'there must be a corresponding scale prior.')
                if not all(isinstance(i, float) for i in dum_season_var_scale_prior):
                    raise ValueError('All values in dum_season_var_scale_prior must be of type float.')
                if not all(i > 0 for i in dum_season_var_scale_prior):
                    raise ValueError('All values in dum_season_var_scale_prior must be strictly positive floats.')

        # Trigonometric seasonal prior check
        if not isinstance(trig_season_var_shape_prior, tuple):
            raise ValueError('trig_seasonal_var_shape_prior must be a tuple of floats'
                             'to accommodate potentially multiple seasonality.')
        else:
            if len(trig_season_var_shape_prior) > 0:
                if len(trig_season_var_shape_prior) != len(self.trig_seasonal):
                    raise ValueError('A non-empty tuple for trig_season_var_shape_prior must have the same '
                                     'length as trig_seasonal. That is, for each periodicity in trig_seasonal, '
                                     'there must be a corresponding shape prior.')
                if not all(isinstance(i, float) for i in trig_season_var_shape_prior):
                    raise ValueError('All values in trig_season_var_shape_prior must be of type float.')
                if not all(i > 0 for i in trig_season_var_shape_prior):
                    raise ValueError('All values in trig_season_var_shape_prior must be strictly positive floats.')

        if not isinstance(trig_season_var_scale_prior, tuple):
            raise ValueError('trig_seasonal_var_scale_prior must be of type tuple '
                             'to account for multiple seasonality.')
        else:
            if len(trig_season_var_scale_prior) > 0:
                if len(trig_season_var_scale_prior) != len(self.trig_seasonal):
                    raise ValueError('A non-empty tuple for trig_season_var_scale_prior must have the same '
                                     'length as trig_seasonal. That is, for each periodicity in trig_seasonal, '
                                     'there must be a corresponding scale prior.')
                if not all(isinstance(i, float) for i in trig_season_var_scale_prior):
                    raise ValueError('All values in trig_season_var_scale_prior must be of type float.')
                if not all(i > 0 for i in trig_season_var_scale_prior):
                    raise ValueError('All values in trig_season_var_scale_prior must be strictly positive floats.')

        # Predictors prior check
        if not isinstance(reg_coeff_mean_prior, np.ndarray):
            raise ValueError('reg_coeff_mean_prior must be of type Numpy ndarray.')
        else:
            if reg_coeff_mean_prior.size > 0:
                if reg_coeff_mean_prior.dtype != 'float64':
                    raise ValueError('All values in reg_coeff_mean_prior must be of type float.')
                if reg_coeff_mean_prior.ndim == 0:
                    raise ValueError('reg_coeff_mean_prior must have dimension 1 or 2. Dimension is 0.')
                if reg_coeff_mean_prior.ndim == 1:
                    reg_coeff_mean_prior = reg_coeff_mean_prior.reshape(-1, 1)
                if reg_coeff_mean_prior.ndim > 2:
                    raise ValueError('reg_coeff_mean_prior must have dimension 1 or 2. Dimension exceeds 2.')
                if reg_coeff_mean_prior.ndim == 2:
                    if 1 in reg_coeff_mean_prior.shape:
                        reg_coeff_mean_prior = reg_coeff_mean_prior.reshape(-1, 1)
                if reg_coeff_mean_prior.shape[0] != self.num_predictors:
                    raise ValueError('The number of elements in reg_coeff_mean_prior must match the number '
                                     'of predictors.')
                if np.isnan(reg_coeff_mean_prior).any():
                    raise ValueError('reg_coeff_mean_prior cannot have null values.')
                if np.isinf(reg_coeff_mean_prior).any():
                    raise ValueError('reg_coeff_mean_prior cannot have Inf and/or -Inf values.')

        if not isinstance(reg_coeff_precision_prior, np.ndarray):
            raise ValueError('reg_coeff_precision_prior must be of type Numpy ndarray.')
        else:
            if reg_coeff_precision_prior.size > 0:
                if reg_coeff_precision_prior.dtype != 'float64':
                    raise ValueError('All values in reg_coeff_precision_prior must be of type float.')
                if reg_coeff_precision_prior.ndim != 2:
                    raise ValueError('reg_coeff_precision_prior must have dimension 2.')
                if np.isnan(reg_coeff_precision_prior).any():
                    raise ValueError('reg_coeff_precision_prior cannot have null values.')
                if np.isinf(reg_coeff_precision_prior).any():
                    raise ValueError('reg_coeff_precision_prior cannot have Inf and/or -Inf values.')
                if reg_coeff_precision_prior.shape[0] != reg_coeff_precision_prior.shape[1]:
                    raise ValueError('reg_coeff_precision_prior must be square, i.e., the number '
                                     'of rows must match the number of columns.')
                if not ao.is_positive_definite(reg_coeff_precision_prior):
                    raise ValueError('reg_coeff_precision_prior must be a positive definite matrix.')
                if not ao.is_symmetric(reg_coeff_precision_prior):
                    raise ValueError('reg_coeff_precision_prior must be a symmetric matrix.')
                if reg_coeff_mean_prior.size > 0:
                    if reg_coeff_precision_prior.shape[0] != self.num_predictors:
                        raise ValueError('The number of rows and number of columns in '
                                         'reg_coeff_precision_prior must match the number of rows '
                                         'in reg_coeff_mean_prior (i.e., the number of predictors).')

        # Define variables
        y = self.y
        n = self.num_obs
        q = self.num_stochastic_states
        m = self.num_state_eqs
        Z = self.observation_matrix()
        T = self.state_transition_matrix
        R = self.state_error_transformation_matrix
        A = self.posterior_state_error_covariance_transformation_matrix
        X = self.predictors

        # Bring in the model configuration from _model_setup()
        model = self._model_setup(response_var_shape_prior, response_var_scale_prior,
                                  level_var_shape_prior, level_var_scale_prior,
                                  slope_var_shape_prior, slope_var_scale_prior,
                                  dum_season_var_shape_prior, dum_season_var_scale_prior,
                                  trig_season_var_shape_prior, trig_season_var_scale_prior,
                                  reg_coeff_mean_prior, reg_coeff_precision_prior)

        response_var_scale_prior = model.response_var_scale_prior
        response_var_shape_post = model.response_var_shape_post
        state_var_scale_prior = model.state_var_scale_prior
        state_var_shape_post = model.state_var_shape_post
        gibbs_iter0_init_state = model.gibbs_iter0_init_state
        gibbs_iter0_error_variances = model.gibbs_iter0_error_variances
        init_state_plus_values = model.init_state_plus_values
        init_state_covariance = model.init_state_covariance

        # Initialize output arrays
        if q > 0:
            state_error_covariance = np.empty((num_samp, q, q), dtype=np.float64)
        else:
            state_error_covariance = np.empty((num_samp, 0, 0))

        response_error_variance = np.empty((num_samp, 1, 1), dtype=np.float64)
        smoothed_errors = np.empty((num_samp, n, 1 + q, 1), dtype=np.float64)
        smoothed_state = np.empty((num_samp, n + 1, m, 1), dtype=np.float64)
        smoothed_prediction = np.empty((num_samp, n, 1), dtype=np.float64)
        filtered_state = np.empty((num_samp, n + 1, m, 1), dtype=np.float64)
        filtered_prediction = np.empty((num_samp, n, 1), dtype=np.float64)
        state_covariance = np.empty((num_samp, n + 1, m, m), dtype=np.float64)
        response_variance = np.empty((num_samp, n, 1, 1), dtype=np.float64)

        # Set initial values for response and state error variances
        gibbs_iter0_response_error_variance = np.array([[gibbs_iter0_error_variances[0]]])
        if q > 0:
            gibbs_iter0_state_error_covariance = np.diag(gibbs_iter0_error_variances[1:])

        # Helper matrices
        q_eye = np.eye(q)
        n_ones = np.ones((n, 1))

        if self.has_predictors:
            y_nan_indicator = np.isnan(y) * 1.
            y_no_nan = ao.replace_nan(y)
            reg_coeff_mean_prior = model.reg_coeff_mean_prior
            reg_coeff_precision_inv_prior = model.reg_coeff_precision_inv_prior
            reg_coeff_precision_inv_post = model.reg_coeff_precision_inv_post
            gibbs_iter0_reg_coeff = model.gibbs_iter0_reg_coeff
            regression_coefficients = np.empty((num_samp, self.num_predictors, 1), dtype=np.float64)

        # Run Gibbs sampler
        for s in range(num_samp):
            if s < 1:
                init_state_values = gibbs_iter0_init_state
                response_err_var = gibbs_iter0_response_error_variance
                state_err_cov = gibbs_iter0_state_error_covariance
            else:
                init_state_values = smoothed_state[s - 1, 0]
                response_err_var = response_error_variance[s - 1]
                state_err_cov = state_error_covariance[s - 1]

            if self.has_predictors:
                if s < 1:
                    reg_coeff = gibbs_iter0_reg_coeff
                else:
                    reg_coeff = regression_coefficients[s - 1]

                Z[:, :, -1] = X.dot(reg_coeff)

            # Filtered state
            y_kf = kf(y,
                      Z,
                      T,
                      R,
                      response_err_var,
                      state_err_cov,
                      init_state=init_state_values,
                      init_state_covariance=init_state_covariance)

            filtered_state[s] = y_kf.filtered_state
            state_covariance[s] = y_kf.state_covariance
            filtered_prediction[s] = y - y_kf.one_step_ahead_prediction_residual[:, :, 0]
            response_variance[s] = y_kf.response_variance

            # Get smoothed state from DK smoother
            dk = dks(y,
                     Z,
                     T,
                     R,
                     response_err_var,
                     state_err_cov,
                     init_state_plus_values=init_state_plus_values,
                     init_state_values=init_state_values,
                     init_state_covariance=init_state_covariance,
                     has_predictors=self.has_predictors)

            # Smoothed disturbances and state
            smoothed_errors[s] = dk.simulated_smoothed_errors
            smoothed_state[s] = dk.simulated_smoothed_state
            smoothed_prediction[s] = dk.simulated_smoothed_prediction

            # Get new draws for state variances
            if q > 0:
                state_residual = smoothed_errors[s][:, 1:, 0]
                state_sse = dot(state_residual.T ** 2, n_ones)
                state_var_scale_post = state_var_scale_prior + 0.5 * state_sse
                state_var_post = dot(A, dist.vec_ig(state_var_shape_post, state_var_scale_post))
                state_error_covariance[s] = q_eye * state_var_post

            # Get new draw for observation variance
            smooth_one_step_ahead_prediction_residual = smoothed_errors[s, :, 0]
            response_var_scale_post = (response_var_scale_prior
                                       + 0.5 * dot(smooth_one_step_ahead_prediction_residual.T,
                                                   smooth_one_step_ahead_prediction_residual))
            response_error_variance[s] = dist.vec_ig(response_var_shape_post, response_var_scale_post)

            if self.has_predictors:
                # Get new draw for regression coefficients
                y_adj = y_no_nan + y_nan_indicator * smoothed_prediction[s]
                smooth_time_prediction = smoothed_prediction[s] - Z[:, :, -1]
                y_tilde = y_adj - smooth_time_prediction  # y with smooth time prediction subtracted out
                reg_coeff_mean_post = dot(reg_coeff_precision_inv_post,
                                          (dot(X.T, y_tilde) + dot(reg_coeff_precision_inv_prior,
                                                                   reg_coeff_mean_prior)))

                cov_post = response_error_variance[s][0, 0] * reg_coeff_precision_inv_post
                regression_coefficients[s] = (np
                                              .random
                                              .multivariate_normal(mean=reg_coeff_mean_post.flatten(),
                                                                   cov=cov_post).reshape(-1, 1))

        if self.has_predictors:
            self.posterior = Posterior(num_samp, smoothed_state, smoothed_errors, smoothed_prediction,
                                       filtered_state, filtered_prediction, response_variance, state_covariance,
                                       response_error_variance, state_error_covariance, regression_coefficients)
        else:
            self.posterior = Posterior(num_samp, smoothed_state, smoothed_errors, smoothed_prediction,
                                       filtered_state, filtered_prediction, response_variance, state_covariance,
                                       response_error_variance, state_error_covariance)

        return self.posterior

    def forecast(self,
                 num_periods: int,
                 burn: int = 0,
                 future_predictors: Union[np.ndarray, pd.Series, pd.DataFrame] = np.array([[]])):

        """

        :param num_periods:
        :param burn:
        :param future_predictors:
        :return:
        """

        Z = self.observation_matrix(num_rows=num_periods)
        T = self.state_transition_matrix
        R = self.state_error_transformation_matrix

        if not isinstance(num_periods, int):
            raise ValueError('num_periods must be of type integer.')
        else:
            if not num_periods > 0:
                raise ValueError('num_periods must be a strictly positive integer.')

        if not isinstance(burn, int):
            raise ValueError('burn must be of type integer.')
        else:
            if burn < 0:
                raise ValueError('burn must be a non-negative integer.')

        if isinstance(self.historical_time_index, pd.DatetimeIndex):
            freq = self.historical_time_index.freq
            last_historical_date = self.historical_time_index[-1]
            first_future_date = last_historical_date + 1 * freq
            last_future_date = last_historical_date + num_periods * freq
            self.future_time_index = pd.date_range(first_future_date, last_future_date, freq=freq)
        else:
            self.future_time_index = np.arange(self.num_obs, self.num_obs + num_periods)

        if self.has_predictors:
            future_predictors = future_predictors.copy()
            # Check and prepare future predictor data
            # -- data types match across predictors and future_predictors
            if not isinstance(future_predictors, type(self.predictors)):
                raise ValueError('Object types for predictors and future_predictors must match.')

            # -- check if object type is valid
            if not isinstance(future_predictors, (pd.Series, pd.DataFrame, np.ndarray)):
                raise ValueError("The future_predictors array must be a Numpy array, Pandas Series, "
                                 "or Pandas DataFrame.")
            else:
                # -- if Pandas type, grab index and column names
                if isinstance(future_predictors, (pd.Series, pd.DataFrame)):
                    if not isinstance(future_predictors.index, type(self.future_time_index)):
                        raise ValueError('The future_predictors index and predictors index must be of the same type.')

                    if not (future_predictors.index == self.future_time_index).all():
                        raise ValueError('The future_predictors index must match the future time index '
                                         'implied by the last observed date for the response and the '
                                         'number of desired forecast periods. Check the class attribute '
                                         'future_time_index to verify that it is correct.')

                    if isinstance(future_predictors, pd.Series):
                        future_predictors_names = [future_predictors.name]
                    else:
                        future_predictors_names = future_predictors.columns.values.tolist()

                    if len(future_predictors_names) != self.num_predictors:
                        raise ValueError(
                            f'The number of predictors used for historical estimation {self.num_predictors} '
                            f'does not match the number of predictors specified for forecasting '
                            f'{len(future_predictors_names)}. The same set of predictors must be used.')
                    else:
                        if not all(self.predictors_names[i] == future_predictors_names[i]
                                   for i in range(self.num_predictors)):
                            raise ValueError('The order and names of the columns in predictors must match '
                                             'the order and names in future_predictors.')

                    future_predictors = future_predictors.sort_index().to_numpy()

            # -- dimensions
            if future_predictors.ndim == 0:
                raise ValueError('The future_predictors array must have dimension 1 or 2. Dimension is 0.')
            if future_predictors.ndim == 1:
                future_predictors = future_predictors.reshape(-1, 1)
            if future_predictors.ndim > 2:
                raise ValueError('The future_predictors array must have dimension 1 or 2. Dimension exceeds 2.')
            if future_predictors.ndim == 2:
                if 1 in future_predictors.shape:
                    future_predictors = future_predictors.reshape(-1, 1)
            if np.isnan(future_predictors).any():
                raise ValueError('The future_predictors array cannot have null values.')
            if np.isinf(future_predictors).any():
                raise ValueError('The future_predictors array cannot have Inf and/or -Inf values.')

            # Final sanity checks
            if self.num_predictors != future_predictors.shape[1]:
                raise ValueError(f'The number of predictors used for historical estimation {self.num_predictors} '
                                 f'does not match the number of predictors specified for forecasting '
                                 f'{future_predictors.shape[1]}. The same set of predictors must be used.')

            if num_periods > future_predictors.shape[0]:
                raise ValueError(f'The number of requested forecast periods {num_periods} exceeds the '
                                 f'number of observations provided in future_predictors {future_predictors.shape[0]}. '
                                 f'The former must be no larger than the latter.')
            else:
                if num_periods < future_predictors.shape[0]:
                    warnings.warn(f'The number of requested forecast periods {num_periods} is less than the '
                                  f'number of observations provided in future_predictors {future_predictors.shape[0]}. '
                                  f'Only the first {num_periods} observations will be used '
                                  f'in future_predictors.')

        y_forecast = _forecast(self.posterior,
                               num_periods,
                               Z,
                               T,
                               R,
                               future_predictors,
                               burn)

        return y_forecast

    def plot_components(self,
                        burn: int = 0,
                        cred_int_level: float = 0.05,
                        random_sample_size_prop: float = 1.,
                        smoothed: bool = True):

        """

        :param burn:
        :param cred_int_level:
        :param random_sample_size_prop:
        :param smoothed:
        :return:
        """

        if not isinstance(burn, int):
            raise ValueError('burn must be of type integer.')
        else:
            if burn < 0:
                raise ValueError('burn must be a non-negative integer.')

        if not isinstance(cred_int_level, float):
            raise ValueError('cred_int_level must be of type float.')
        else:
            if cred_int_level <= 0 or cred_int_level >= 1:
                raise ValueError('cred_int_level must be a value in the '
                                 'interval (0, 1).')

        if not isinstance(random_sample_size_prop, float):
            raise ValueError('random_sample_size_prop must be of type float.')
        else:
            if random_sample_size_prop <= 0 or random_sample_size_prop > 1:
                raise ValueError('random_sample_size_prop must be a value in the '
                                 'interval (0, 1].')

        if not isinstance(smoothed, bool):
            raise ValueError('smoothed must be of type bool.')

        num_first_obs_ignore = self.num_first_obs_ignore

        if self.has_predictors:
            X = self.predictors[num_first_obs_ignore:, :]
            reg_coeff = self.posterior.regression_coefficients[burn:, :, 0].T

        y = self.y[num_first_obs_ignore:, 0]
        n = self.num_obs
        Z = self.observation_matrix(num_rows=n - num_first_obs_ignore)
        historical_time_index = self.historical_time_index[num_first_obs_ignore:]
        model = self.model_setup
        components = model.components
        cred_int_lb = 0.5 * cred_int_level
        cred_int_ub = 1. - 0.5 * cred_int_level

        filtered_prediction = _simulate_posterior_predictive_response(self.posterior,
                                                                      burn,
                                                                      num_first_obs_ignore,
                                                                      random_sample_size_prop)
        smoothed_prediction = self.posterior.smoothed_prediction[burn:, num_first_obs_ignore:, 0]

        if smoothed:
            prediction = smoothed_prediction
            state = self.posterior.smoothed_state[burn:, num_first_obs_ignore:n, :, :]
        else:
            prediction = filtered_prediction
            state = _simulate_posterior_predictive_state(self.posterior,
                                                         burn,
                                                         num_first_obs_ignore,
                                                         random_sample_size_prop,
                                                         self.has_predictors)

        fig, ax = plt.subplots(1 + len(components))
        fig.set_size_inches(12, 10)
        ax[0].plot(historical_time_index, y)
        ax[0].plot(historical_time_index, np.mean(filtered_prediction, axis=0))
        lb = np.quantile(filtered_prediction, cred_int_lb, axis=0)
        ub = np.quantile(filtered_prediction, cred_int_ub, axis=0)
        ax[0].fill_between(historical_time_index, lb, ub, alpha=0.2)
        ax[0].title.set_text('Predicted vs. observed response')
        ax[0].legend(('Observed', 'One-step-ahead prediction', f'{100 * (1 - cred_int_level)}% prediction interval'),
                     loc='upper left')

        for i, c in enumerate(components):
            if c == 'Irregular':
                resid = y[np.newaxis] - prediction
                ax[i + 1].plot(historical_time_index, np.mean(resid, axis=0))
                lb = np.quantile(resid, cred_int_lb, axis=0)
                ub = np.quantile(resid, cred_int_ub, axis=0)
                ax[i + 1].fill_between(historical_time_index, lb, ub, alpha=0.2)
                ax[i + 1].title.set_text(c)

            if c == 'Trend':
                time_component = state[:, :, 1, 0]
                ax[i + 1].plot(historical_time_index, np.mean(time_component, axis=0))
                lb = np.quantile(time_component, cred_int_lb, axis=0)
                ub = np.quantile(time_component, cred_int_ub, axis=0)
                ax[i + 1].fill_between(historical_time_index, lb, ub, alpha=0.2)
                ax[i + 1].title.set_text(c)

            if c not in ('Irregular', 'Regression', 'Trend'):
                v = components[c]
                start_index, end_index = v['start_obs_mat_col_index'], v['end_obs_mat_col_index']
                A = Z[:, 0, start_index:end_index]
                B = state[:, :, start_index:end_index, 0]
                time_component = (A[np.newaxis] * B).sum(axis=2)

                ax[i + 1].plot(historical_time_index, np.mean(time_component, axis=0))
                lb = np.quantile(time_component, cred_int_lb, axis=0)
                ub = np.quantile(time_component, cred_int_ub, axis=0)
                ax[i + 1].fill_between(historical_time_index, lb, ub, alpha=0.2)
                ax[i + 1].title.set_text(c)

            if c == 'Regression':
                reg_component = X.dot(reg_coeff)
                ax[i + 1].plot(historical_time_index, np.mean(reg_component, axis=1))
                lb = np.quantile(reg_component, cred_int_lb, axis=1)
                ub = np.quantile(reg_component, cred_int_ub, axis=1)
                ax[i + 1].fill_between(historical_time_index, lb, ub, alpha=0.2)
                ax[i + 1].title.set_text(c)

        fig.tight_layout()

        return

    def summary(self,
                burn: int = 0,
                cred_int_level: float = 0.05) -> dict:

        """

        :param burn:
        :param cred_int_level:
        :return:
        """

        if not isinstance(burn, int):
            raise ValueError('burn must be of type integer.')
        else:
            if burn < 0:
                raise ValueError('burn must be a non-negative integer.')

        if not isinstance(cred_int_level, float):
            raise ValueError('cred_int_level must be of type float.')
        else:
            if cred_int_level <= 0 or cred_int_level >= 1:
                raise ValueError('cred_int_level must be a value in the '
                                 'interval (0, 1).')

        components = self.model_setup.components
        resp_err_var = self.posterior.response_error_variance[burn:]
        state_err_cov = self.posterior.state_error_covariance[burn:]
        summary_d = dict()
        summary_d['Number of posterior samples (after burn)'] = self.posterior.num_samp - burn

        for c in components:
            if c == 'Irregular':
                summary_d[f"Posterior.Mean[{c}]"] = np.mean(resp_err_var)
                summary_d[f"Posterior.StdDev[{c}]"] = np.std(resp_err_var)
                summary_d[f"Posterior.CredInt.LB[{c}]"] = np.quantile(resp_err_var,
                                                                      0.5 * cred_int_level)
                summary_d[f"Posterior.CredInt.UB[{c}]"] = np.quantile(resp_err_var,
                                                                      1. - 0.5 * cred_int_level)

            if c not in ('Irregular', 'Regression'):
                v = components[c]
                stochastic = v['stochastic']
                cov_idx = v['state_error_cov_diag_index']

                if stochastic:
                    summary_d[f"Posterior.Mean[{c}]"] = np.mean(state_err_cov[:, cov_idx, cov_idx])
                    summary_d[f"Posterior.StdDev[{c}]"] = np.std(state_err_cov[:, cov_idx, cov_idx])
                    summary_d[f"Posterior.CredInt.LB[{c}]"] = np.quantile(state_err_cov[:, cov_idx, cov_idx],
                                                                          0.5 * cred_int_level)
                    summary_d[f"Posterior.CredInt.UB[{c}]"] = np.quantile(state_err_cov[:, cov_idx, cov_idx],
                                                                          1. - 0.5 * cred_int_level)

            if c == 'Regression':
                reg_coeff = self.posterior.regression_coefficients[burn:, :, 0]
                mean_reg_coeff = np.mean(reg_coeff, axis=0)
                std_reg_coeff = np.std(reg_coeff, axis=0)
                ci_lb_reg_coeff = np.quantile(reg_coeff, 0.5 * cred_int_level, axis=0)
                ci_ub_reg_coeff = np.quantile(reg_coeff, 1. - 0.5 * cred_int_level, axis=0)
                for k in range(self.num_predictors):
                    summary_d[f"Posterior.Mean[beta.{self.predictors_names[k]}]"] = mean_reg_coeff[k]
                    summary_d[f"Posterior.StdDev[beta.{self.predictors_names[k]}]"] = std_reg_coeff[k]
                    summary_d[f"Posterior.CredInt.LB[beta.{self.predictors_names[k]}]"] = ci_lb_reg_coeff[k]
                    summary_d[f"Posterior.CredInt.UB[beta.{self.predictors_names[k]}]"] = ci_ub_reg_coeff[k]

        return summary_d
