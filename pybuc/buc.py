import numpy as np
from numpy import dot
from numpy.linalg import solve
from numba import njit
from collections import namedtuple
import matplotlib.pyplot as plt
from multiprocessing import Pool
import warnings
from typing import Union
import pandas as pd

from .statespace.kalman_filter import kalman_filter as kf
from .statespace.durbin_koopman_smoother import dk_smoother as dks
from .utils import array_operations as ao
from .vectorized import distributions as dist

post = namedtuple('post',
                  ['num_samp',
                   'smoothed_state',
                   'smoothed_errors',
                   'smoothed_prediction',
                   'filtered_state',
                   'filtered_prediction',
                   'response_variance',
                   'state_covariance',
                   'response_error_variance',
                   'state_error_variance',
                   'regression_coefficients'])

model_setup = namedtuple('model_setup',
                         ['components',
                          'response_var_scale_prior',
                          'response_var_shape_post',
                          'state_var_scale_prior',
                          'state_var_shape_post',
                          'reg_coeff_mean_prior',
                          'reg_coeff_var_prior',
                          'init_error_variances',
                          'init_state_covariance'])


def _is_odd(x: int) -> bool:
    return np.mod(x, 2) != 0


@njit
def set_seed(value):
    np.random.seed(value)


@njit(cache=True)
def _simulate_posterior_predictive_response(posterior, burn=0, num_fit_ignore=0,
                                            random_sample_size_prop=1.):
    response_mean = posterior.filtered_prediction[burn:, num_fit_ignore:, 0]
    response_variance = posterior.response_variance[burn:, num_fit_ignore:, 0]
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


def _simulate_posterior_predictive_state_worker(state_mean, state_covariance):
    state_post = (np.random
                  .multivariate_normal(mean=state_mean,
                                       cov=state_covariance).reshape(-1, 1))

    return state_post


def _simulate_posterior_predictive_state(posterior, burn=0, num_fit_ignore=0, random_sample_size_prop=1.,
                                         has_predictors=False, static_regression=False):
    if has_predictors and static_regression:
        mean = posterior.filtered_state[burn:, num_fit_ignore:-1, :-1, 0]
        cov = posterior.state_covariance[burn:, num_fit_ignore:-1, :-1, :-1]
    else:
        mean = posterior.filtered_state[burn:, num_fit_ignore:-1, :, 0]
        cov = posterior.state_covariance[burn:, num_fit_ignore:-1]

    num_posterior_samp = mean.shape[0]
    n = mean.shape[1]
    m = mean.shape[2]

    if int(random_sample_size_prop * num_posterior_samp) > posterior.num_samp:
        raise ValueError('random_sample_size_prop must be between 0 and 1.')

    if int(random_sample_size_prop * num_posterior_samp) < 1:
        raise ValueError('random_sample_size_prop implies a sample with less than 1 observation. '
                         'Provide a random_sample_size_prop such that the number of samples '
                         'such that the number of samples is at least 1 but no larger than the '
                         'number of posterior samples.')

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
def _forecast(posterior, num_periods, burn, state_observation_matrix, state_transition_matrix,
              state_error_transformation_matrix, future_predictors=np.array([[]])):
    Z = state_observation_matrix
    T = state_transition_matrix
    R = state_error_transformation_matrix
    X_fut = future_predictors
    m = R.shape[0]
    q = R.shape[1]

    response_error_variance = posterior.response_error_variance[burn:]
    if q > 0:
        state_error_variance = posterior.state_error_variance[burn:]
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
                                        num_periods_u_ones * np.sqrt(ao.diag_2d(state_error_variance[s])))

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
                 seasonal: int = 0,
                 stochastic_seasonal: bool = True,
                 trig_seasonal: tuple[tuple] = (),
                 stochastic_trig_seasonal: tuple = (),
                 standardize: bool = False,
                 static_regression: bool = True):

        self.level = level
        self.stochastic_level = stochastic_level
        self.slope = slope
        self.stochastic_slope = stochastic_slope
        self.seasonal = seasonal
        self.stochastic_seasonal = stochastic_seasonal
        self._trig_seasonal = trig_seasonal
        self._stochastic_trig_seasonal = stochastic_trig_seasonal
        self.standardize = standardize
        self.predictors = predictors
        self.static_regression = static_regression

        self.model_setup = None
        self.num_fit_ignore = None
        self.response_name = None
        self.response_index = None
        self.predictors_names = None
        self.predictors_index = None

        # TODO: Add functionality for multiple dummy seasonality.
        # TODO: Add functionality for autoregressive slope.

        # Check and prepare response data
        # -- data types, name, and index
        if not isinstance(response, (pd.Series, pd.DataFrame, np.ndarray)):
            raise ValueError("The response array must be a Numpy array, Pandas Series, or Pandas DataFrame.")
        else:
            if isinstance(response, (pd.Series, pd.DataFrame)):
                self.response = response.to_numpy()
                self.response_index = response.index
                if isinstance(response, pd.Series):
                    self.response_name = [response.name]
                else:
                    self.response_name = response.columns.values.tolist()
            else:
                self.response = response

        # -- dimensions
        if self.response.ndim == 0:
            raise ValueError('The response array must have dimension 1 or 2.')
        elif self.response.ndim == 1:
            self.response = self.response.reshape(-1, 1)
        elif self.response.ndim > 2:
            raise ValueError('The response array must have dimension 1 or 2.')
        else:
            if all(i > 1 for i in self.response.shape):
                raise ValueError('The response array must have shape (1, n) or (n, 1), '
                                 'where n is the number of observations. Both the row and column '
                                 'count exceed 1.')
            else:
                self.response = self.response.reshape(-1, 1)

        # -- name and index if response is a Numpy array
        if self.response_name is None:
            self.response_name = ['y']

        # Check and prepare predictors data
        # -- data types. names, and index
        if self.has_predictors:
            if not isinstance(predictors, (pd.Series, pd.DataFrame, np.ndarray)):
                raise ValueError("The predictors array must be a Numpy array, Pandas Series, or Pandas DataFrame.")
            else:
                if isinstance(predictors, (pd.Series, pd.DataFrame)):
                    self.predictors = predictors.to_numpy()
                    self.predictors_index = predictors.index
                    if isinstance(predictors, pd.Series):
                        self.predictors_names = [predictors.name]
                    else:
                        self.predictors_names = predictors.columns.values.tolist()
                else:
                    self.predictors = predictors

            # -- dimensions
            if self.predictors.ndim == 0:
                raise ValueError('The predictors array must have dimension 1 or 2.')
            elif self.predictors.ndim == 1:
                self.predictors = predictors.reshape(-1, 1)
            elif self.predictors.ndim > 2:
                raise ValueError('The predictors array must have dimension 1 or 2.')
            else:
                if np.isnan(self.predictors).any():
                    raise ValueError('The predictors array cannot have null values.')
                if np.isinf(self.predictors).any():
                    raise ValueError('The predictors array cannot have Inf and/or -Inf values.')
                if 1 in self.predictors.shape:
                    self.predictors = predictors.reshape(-1, 1)

            # -- conformable number of observations
            if self.predictors.shape[0] != self.response.shape[0]:
                raise ValueError('The number of observations in the predictors array must match '
                                 'the number of observations in the response array.')

            # -- warn about model stability if the number of predictors exceeds number of observations
            if self.predictors.shape[1] > self.predictors.shape[0]:
                warnings.warn('The number of predictors exceeds the number of observations. '
                              'Results will be sensitive to choice of priors.')

            # -- names and index if predictors is a Numpy array
            if self.predictors_names is None:
                self.predictors_names = [f"x_{i}" for i in range(self.num_predictors)]

            # -- response index matches predictors index
            if not (self.response_index == self.predictors_index).all():
                raise ValueError('The response index must match the predictors index.')

        # Check time components
        if seasonal == 1:
            raise ValueError('The seasonal argument takes 0 and integers greater than 1 as valid inputs. '
                             'A value of 1 is not valid.')

        if self.seasonal == 0 and not self.level:
            raise ValueError('At least a level or seasonal component must be specified.')

        if self.slope and not self.level:
            raise ValueError('Slope cannot be specified without a level component.')

        self._check_trig_seasonal()

    def _check_trig_seasonal(self):
        if len(self.trig_seasonal) > 0:
            if len(self.trig_seasonal) > len(self._stochastic_trig_seasonal):
                if len(self._stochastic_trig_seasonal) > 0:
                    raise ValueError('Some of the trigonometric seasonal components '
                                     'were given a stochastic specification, but not all. '
                                     'Partial specification of the stochastic profile is not '
                                     'allowed. Either leave the stochastic specification blank '
                                     'by passing an empty tuple (), which will default to True '
                                     'for all components, or pass a stochastic specification '
                                     'for each seasonal component.')
            elif len(self.trig_seasonal) < len(self.stochastic_trig_seasonal):
                raise ValueError('The tuple which specifies the number of stochastic trigonometric '
                                 'seasonal components has greater length than the tuple that specifies '
                                 'the number of trigonometric seasonal components. Either pass a blank '
                                 'tuple () for the stochastic profile, or a boolean tuple of same length '
                                 'as the tuple that specifies the number of trigonometric seasonal components.')
            else:
                pass

            for v in self.trig_seasonal:
                if len(v) != 2:
                    raise ValueError(f'A (period, frequency) tuple must be provided '
                                     f'for each specified trigonometric seasonal component '
                                     f'{v} was passed for one of the seasonal components.')
                period, freq = v
                if not isinstance(period, int) or not isinstance(freq, int):
                    raise ValueError('Both the period and frequency for a specified '
                                     'trigonometric seasonal component must be of integer type.')
                if period <= 1:
                    raise ValueError('The period for a trigonometric seasonal component must '
                                     'be an integer greater than 1.')
                if freq < 1 and freq != 0:
                    raise ValueError(f'The frequency for a trigonometric seasonal component can take 0 or integers '
                                     f'at least as large as 1 as valid options. A value of 0 will enforce '
                                     f'the highest possible frequency for the given period, which is period / 2 '
                                     f'if period is even, or (period - 1) / 2 if period is odd. The frequency '
                                     f'passed, {freq}, is not a valid option.')
                if _is_odd(period):
                    if freq > int(period - 1) / 2:
                        raise ValueError('The frequency value for a trigonometric seasonal component cannot '
                                         'exceed (period - 1) / 2 when period is odd.')
                else:
                    if freq > int(period / 2):
                        raise ValueError('The frequency value for a trigonometric seasonal component cannot '
                                         'exceed period / 2 when period is even.')

            if len(self.stochastic_trig_seasonal) > 0:
                for v in self.stochastic_trig_seasonal:
                    if not isinstance(v, bool):
                        raise ValueError('If an empty tuple is not passed for the stochastic specification '
                                         'of the trigonometric seasonal components, all elements must be of '
                                         'boolean type.')

        else:
            if len(self.stochastic_trig_seasonal) > 0:
                raise ValueError('No trigonometric seasonal components were specified, but a non-empty '
                                 'stochastic profile was passed for trigonometric seasonality. If no '
                                 'trigonometric seasonal components are desired, then an empty stochastic '
                                 'profile, i.e., (), should be passed to stochastic_trig_seasonal. An '
                                 'empty stochastic profile is passed by default.')

        return

    @property
    def trig_seasonal(self):
        f = ()
        for c, v in enumerate(self._trig_seasonal):
            period, freq = v
            if freq == 0:
                if _is_odd(freq):
                    h = int((period - 1) / 2)
                else:
                    h = int(period / 2)
            else:
                h = freq

            g = (period, h)
            f = f + (g,)

        return f

    @property
    def stochastic_trig_seasonal(self):
        if len(self.trig_seasonal) > len(self._stochastic_trig_seasonal):
            if len(self._stochastic_trig_seasonal) == 0:
                return (True,) * len(self.trig_seasonal)
        else:
            return self._stochastic_trig_seasonal

    @property
    def num_trig_seasonal_state_eqs(self):
        if len(self.trig_seasonal) == 0:
            return 0
        else:
            num_eqs = 0
            for v in self.trig_seasonal:
                _, freq = v
                num_eqs += 2 * freq
        return num_eqs

    @property
    def num_stochastic_trig_seasonal_states(self):
        num_stochastic = 0
        for c, v in enumerate(self.trig_seasonal):
            _, freq = v
            num_stochastic += 2 * freq * self.stochastic_trig_seasonal[c]

        return num_stochastic

    @property
    def num_indicator_seasonal_state_eqs(self):
        return (self.seasonal - 1) * (self.seasonal > 1) * 1

    @property
    def has_predictors(self):
        if self.predictors.size != 0:
            return True
        else:
            return False

    @property
    def num_obs(self):
        return self.response.shape[0]

    @property
    def num_seasonal_state_eqs(self):
        return self.num_indicator_seasonal_state_eqs + self.num_trig_seasonal_state_eqs

    @property
    def num_predictors(self):
        if self.predictors.size == 0:
            return 0
        else:
            return self.predictors.shape[1]

    @property
    def num_state_eqs(self):
        return ((self.level + self.slope) * 1
                + self.num_seasonal_state_eqs
                + (self.num_predictors > 0) * 1)

    @property
    def num_stochastic_states(self):
        return (self.level * self.stochastic_level
                + self.slope * self.stochastic_slope
                + (self.seasonal > 1) * self.stochastic_seasonal
                + self.num_stochastic_trig_seasonal_states) * 1

    @property
    def mean_response(self):
        return np.nanmean(self.response)

    @property
    def sd_response(self):
        return np.nanstd(self.response)

    @property
    def z_response(self):
        return (self.response - self.mean_response) / self.sd_response

    @property
    def y(self):
        if self.standardize:
            return self.z_response
        else:
            return self.response

    @staticmethod
    def trig_transition_matrix(freq):
        real_part = np.array([[np.cos(freq), np.sin(freq)]])
        imaginary_part = np.array([[-np.sin(freq), np.cos(freq)]])
        return np.concatenate((real_part, imaginary_part), axis=0)

    def observation_matrix(self, num_rows=0):
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
        if self.seasonal > 1:
            Z[:, :, j] = 1.
            j += self.num_indicator_seasonal_state_eqs
        if len(self.trig_seasonal) > 0:
            Z[:, :, j::2] = 1.
            j += self.num_trig_seasonal_state_eqs
        if self.num_predictors > 0:
            Z[:, :, j] = 0.

        return Z

    def state_transition_matrix(self):
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
        if self.seasonal > 1:
            T[i, j:j + self.num_indicator_seasonal_state_eqs] = -1.
            for k in range(self.num_indicator_seasonal_state_eqs):
                T[i + k, j + k] = 1.
            i += self.num_indicator_seasonal_state_eqs
            j += self.num_indicator_seasonal_state_eqs
        if len(self.trig_seasonal) > 0:
            for c, w in enumerate(self.trig_seasonal):
                period, freq = w
                for k in range(1, freq + 1):
                    T[i:i + 2, j:j + 2] = self.trig_transition_matrix(2. * np.pi * k / period)
                    i += 2
                    j += 2
        if self.num_predictors > 0:
            T[i, j] = 1.

        return T

    def state_error_transformation_matrix(self):
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
            if self.seasonal > 1:
                if self.stochastic_seasonal:
                    R[i, j] = 1.
                    j += 1
                i += self.num_indicator_seasonal_state_eqs
            if len(self.trig_seasonal) > 0:
                for c, w in enumerate(self.trig_seasonal):
                    _, freq = w
                    num_terms = 2 * freq
                    if self.stochastic_trig_seasonal[c]:
                        for k in range(num_terms):
                            R[i + k, j + k] = 1.
                        j += num_terms
                    i += num_terms

        return R

    def posterior_state_error_variance_transformation_matrix(self):
        q = self.num_stochastic_states
        A = np.zeros((q, q), dtype=np.float64)

        if q == 0:
            pass
        else:
            if self.num_stochastic_trig_seasonal_states == 0:
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
                if self.seasonal > 1:
                    if self.stochastic_seasonal:
                        A[i, i] = 1.
                        i += 1
                if len(self.trig_seasonal) > 0:
                    for c, w in enumerate(self.trig_seasonal):
                        _, freq = w
                        num_terms = 2 * freq
                        if self.stochastic_trig_seasonal[c]:
                            for k in range(num_terms):
                                A[i + k, i:i + num_terms] = 1. / (2 * freq)
                            i += 2 * freq

        return A

    def _model_setup(self, response_var_shape_prior, response_var_scale_prior,
                     level_var_shape_prior, level_var_scale_prior,
                     slope_var_shape_prior, slope_var_scale_prior,
                     season_var_shape_prior, season_var_scale_prior,
                     trig_season_var_shape_prior, trig_season_var_scale_prior,
                     reg_coeff_mean_prior, reg_coeff_var_prior):

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
        init_response_error_var = [0.01 * self.sd_response ** 2]

        state_var_scale_prior = []
        state_var_shape_post = []
        init_state_error_var = []
        init_state_variances = []
        num_fit_ignore = []
        j = 0
        if self.level:
            num_fit_ignore.append(1)
            if self.stochastic_level:
                if np.isnan(level_var_shape_prior):
                    level_var_shape_prior = 0.001

                if np.isnan(level_var_scale_prior):
                    level_var_scale_prior = 0.001

                state_var_shape_post.append(level_var_shape_prior + 0.5 * n)
                state_var_scale_prior.append(level_var_scale_prior)
                init_state_error_var.append(0.01 * self.sd_response ** 2)

            init_state_variances.append(1e6)
            components['Level'] = dict(start_index=j, end_index=j + 1)
            j += 1

        if self.slope:
            num_fit_ignore.append(1)
            if self.stochastic_slope:
                if np.isnan(slope_var_shape_prior):
                    slope_var_shape_prior = 0.001

                if np.isnan(slope_var_scale_prior):
                    slope_var_scale_prior = 0.001

                state_var_shape_post.append(slope_var_shape_prior + 0.5 * n)
                state_var_scale_prior.append(slope_var_scale_prior)
                init_state_error_var.append(0.01 * self.sd_response ** 2)

            components['Trend'] = dict()
            init_state_variances.append(1e6)
            j += 1

        if self.seasonal > 1:
            num_fit_ignore.append(self.num_indicator_seasonal_state_eqs)
            if self.stochastic_seasonal:
                if np.isnan(season_var_shape_prior):
                    season_var_shape_prior = 0.001

                if np.isnan(season_var_scale_prior):
                    season_var_scale_prior = 0.001

                state_var_shape_post.append(season_var_shape_prior + 0.5 * n)
                state_var_scale_prior.append(season_var_scale_prior)
                init_state_error_var.append(0.01 * self.sd_response ** 2)

            for k in range(self.seasonal - 1):
                init_state_variances.append(1e6)

            components[f'Seasonal.{self.seasonal}'] = dict(start_index=j,
                                                           end_index=j + (self.seasonal - 1) + 1)
            j += self.seasonal - 1

        if len(self.trig_seasonal) > 0:
            num_fit_ignore.append(self.num_trig_seasonal_state_eqs)
            if True in self.stochastic_trig_seasonal:
                if np.isnan(trig_season_var_shape_prior):
                    trig_season_var_shape_prior = 0.001

                if np.isnan(trig_season_var_scale_prior):
                    trig_season_var_scale_prior = 0.001

            i = j
            for c, v in enumerate(self.trig_seasonal):
                f, h = v
                num_terms = 2 * h
                if self.stochastic_trig_seasonal[c]:
                    for k in range(num_terms):
                        state_var_shape_post.append(trig_season_var_shape_prior + 0.5 * n)
                        state_var_scale_prior.append(trig_season_var_scale_prior)
                        init_state_error_var.append(0.01 * self.sd_response ** 2)

                for k in range(num_terms):
                    init_state_variances.append(1e6)
                components[f'Trigonometric-Seasonal.{f}.{h}'] = dict(start_index=i,
                                                                     end_index=i + num_terms + 1)
                i += 2 * h
            j += self.num_trig_seasonal_state_eqs

        if self.num_predictors > 0:
            components['Regression'] = dict()
            X = self.predictors
            init_state_variances.append(0.)

            if reg_coeff_mean_prior.size == 0:
                print('No mean prior was provided for the regression coefficient vector. '
                      'A 0-mean prior will be enforced.')
                reg_coeff_mean_prior = np.zeros((self.num_predictors, 1))

            if reg_coeff_var_prior.size == 0:
                kappa = 1.
                g = kappa / n
                print('No variance prior was provided for the regression coefficient vector. '
                      'A g=1/n Zellner g-prior will be enforced.')
                reg_coeff_var_prior = g * (0.5 * dot(X.T, X) + 0.5 * np.diag(dot(X.T, X)))

        self.num_fit_ignore = sum(num_fit_ignore)
        state_var_shape_post = np.array(state_var_shape_post).reshape(-1, 1)
        state_var_scale_prior = np.array(state_var_scale_prior).reshape(-1, 1)
        init_error_variances = np.concatenate((init_response_error_var, init_state_error_var))
        init_state_covariance = np.diag(init_state_variances)

        self.model_setup = model_setup(components,
                                       response_var_scale_prior,
                                       response_var_shape_post,
                                       state_var_scale_prior,
                                       state_var_shape_post,
                                       reg_coeff_mean_prior,
                                       reg_coeff_var_prior,
                                       init_error_variances,
                                       init_state_covariance)

        return self.model_setup

    def sample(self, num_samp=1000,
               response_var_shape_prior=np.nan, response_var_scale_prior=np.nan,
               level_var_shape_prior=np.nan, level_var_scale_prior=np.nan,
               slope_var_shape_prior=np.nan, slope_var_scale_prior=np.nan,
               season_var_shape_prior=np.nan, season_var_scale_prior=np.nan,
               trig_season_var_shape_prior=np.nan, trig_season_var_scale_prior=np.nan,
               reg_coeff_mean_prior=np.array([[]]), reg_coeff_var_prior=np.array([[]])):

        # Define variables
        y = self.y
        n = self.num_obs
        q = self.num_stochastic_states
        m = self.num_state_eqs
        Z = self.observation_matrix()
        T = self.state_transition_matrix()
        R = self.state_error_transformation_matrix()
        A = self.posterior_state_error_variance_transformation_matrix()
        X = self.predictors
        k = self.num_predictors

        # Bring in the model configuration from _model_setup()
        model = self._model_setup(response_var_shape_prior, response_var_scale_prior,
                                  level_var_shape_prior, level_var_scale_prior,
                                  slope_var_shape_prior, slope_var_scale_prior,
                                  season_var_shape_prior, season_var_scale_prior,
                                  trig_season_var_shape_prior, trig_season_var_scale_prior,
                                  reg_coeff_mean_prior, reg_coeff_var_prior)

        response_var_scale_prior = model.response_var_scale_prior
        response_var_shape_post = model.response_var_shape_post
        state_var_scale_prior = model.state_var_scale_prior
        state_var_shape_post = model.state_var_shape_post
        init_error_variances = model.init_error_variances
        init_state_covariance = model.init_state_covariance

        # Initialize output arrays
        if q > 0:
            state_error_variance = np.empty((num_samp, q, q), dtype=np.float64)
        else:
            state_error_variance = np.empty((num_samp, 0, 0))

        response_error_variance = np.empty((num_samp, 1, 1), dtype=np.float64)
        smoothed_errors = np.empty((num_samp, n, 1 + q, 1), dtype=np.float64)
        smoothed_state = np.empty((num_samp, n + 1, m, 1), dtype=np.float64)
        smoothed_prediction = np.empty((num_samp, n, 1), dtype=np.float64)
        filtered_state = np.empty((num_samp, n + 1, m, 1), dtype=np.float64)
        filtered_prediction = np.empty((num_samp, n, 1), dtype=np.float64)
        state_covariance = np.empty((num_samp, n + 1, m, m), dtype=np.float64)
        response_variance = np.empty((num_samp, n, 1, 1), dtype=np.float64)
        init_plus_state_values = np.zeros((m, 1))
        init_state_values0 = np.zeros((m, 1))

        # Set initial values for response and state error variances
        response_error_variance[0] = np.array([[init_error_variances[0]]])
        if q > 0:
            state_error_variance[0] = np.diag(init_error_variances[1:])

        # Helper matrices
        q_eye = np.eye(q)
        n_ones = np.ones((n, 1))

        if self.num_predictors > 0:
            y_nan_indicator = np.isnan(y) * 1.
            y_no_nan = ao.replace_nan(y)
            init_state_values0[-1] = 1.
            reg_coeff_mean_prior = model.reg_coeff_mean_prior
            reg_coeff_var_prior = model.reg_coeff_var_prior
            reg_coeff_var_inv_prior = solve(reg_coeff_var_prior, np.eye(k))
            reg_coeff_var_inv_post = dot(X.T, X) + solve(reg_coeff_var_prior, np.eye(k))
            reg_coeff_var_post = solve(reg_coeff_var_inv_post, np.eye(k))
            regression_coefficients = np.empty((num_samp, k, 1), dtype=np.float64)
            regression_coefficients[0] = dot(reg_coeff_var_post,
                                             (dot(X.T, y) + dot(reg_coeff_var_inv_prior, reg_coeff_mean_prior)))
        else:
            regression_coefficients = np.array([[[]]])

        # Run Gibbs sampler
        for s in range(1, num_samp):
            response_err_var = response_error_variance[s - 1]
            state_err_var = state_error_variance[s - 1]

            if self.num_predictors > 0:
                reg_coeff = regression_coefficients[s - 1]
                Z[:, :, -1] = X.dot(reg_coeff)

            if s == 1:
                init_state_values = init_state_values0
            else:
                init_state_values = smoothed_state[s - 1, 0]

            # Filtered state
            y_kf = kf(y,
                      Z,
                      T,
                      R,
                      response_err_var,
                      state_err_var,
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
                     state_err_var,
                     init_plus_state_values=init_plus_state_values,
                     init_state_values=init_state_values,
                     init_state_covariance=init_state_covariance,
                     static_regression=(self.static_regression * (self.num_predictors > 0)))

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
                state_error_variance[s] = q_eye * state_var_post

            # Get new draw for observation variance
            smooth_one_step_ahead_prediction_residual = smoothed_errors[s, :, 0]
            response_var_scale_post = (response_var_scale_prior
                                       + 0.5 * dot(smooth_one_step_ahead_prediction_residual.T,
                                                   smooth_one_step_ahead_prediction_residual))
            response_error_variance[s] = dist.vec_ig(response_var_shape_post, response_var_scale_post)

            if self.num_predictors > 0:
                # Get new draw for regression coefficients
                y_adj = y_no_nan + y_nan_indicator * smoothed_prediction[s]
                smooth_time_prediction = smoothed_prediction[s] - Z[:, :, -1]
                y_tilde = y_adj - smooth_time_prediction  # y with smooth time prediction subtracted out
                reg_coeff_mean_post = dot(reg_coeff_var_post,
                                          (dot(X.T, y_tilde) + dot(reg_coeff_var_inv_prior, reg_coeff_mean_prior)))

                cov_post = response_error_variance[s][0, 0] * reg_coeff_var_post
                regression_coefficients[s] = (np
                                              .random
                                              .multivariate_normal(mean=reg_coeff_mean_post.flatten(),
                                                                   cov=cov_post).reshape(-1, 1))

        results = post(num_samp, smoothed_state, smoothed_errors, smoothed_prediction,
                       filtered_state, filtered_prediction, response_variance, state_covariance,
                       response_error_variance, state_error_variance, regression_coefficients)

        return results

    def forecast(self, posterior, num_periods, burn=0,
                 future_predictors: Union[np.ndarray, pd.Series, pd.DataFrame] = np.array([[]])):
        Z = self.observation_matrix(num_rows=num_periods)
        T = self.state_transition_matrix()
        R = self.state_error_transformation_matrix()
        X_fut = future_predictors.copy()

        # TODO: Same as __init__, do a data check on future_predictors
        if self.num_predictors > 0:
            # Check and prepare future predictors data
            if not isinstance(X_fut, (pd.Series, pd.DataFrame, np.ndarray)):
                raise ValueError("The future_predictors array must be a Numpy array, Pandas Series, "
                                 "or Pandas DataFrame.")
            else:
                if isinstance(X_fut, (pd.Series, pd.DataFrame)):
                    X_fut = X_fut.to_numpy()
                    X_fut_index = X_fut.index
                else:
                    X_fut_index = None

            # -- dimensions
            if X_fut.ndim == 0:
                raise ValueError('The future_predictors array must have dimension 1 or 2.')
            elif X_fut.ndim == 1:
                X_fut = X_fut.reshape(-1, 1)
            elif X_fut.ndim > 2:
                raise ValueError('The future_predictors array must have dimension 1 or 2.')
            else:
                if np.isnan(X_fut).any():
                    raise ValueError('The future_predictors array cannot have null values.')
                if np.isinf(X_fut).any():
                    raise ValueError('The future_predictors array cannot have Inf and/or -Inf values.')
                if 1 in X_fut.shape:
                    X_fut = X_fut.reshape(-1, 1)

            num_fut_obs, num_fut_predictors = X_fut.shape

            if X_fut_index is None:
                X_fut_index = np.arange(self.num_obs, self.num_obs + num_fut_obs)

            if self.num_predictors != num_fut_predictors:
                raise ValueError(f'The number of predictors used for historical estimation {self.num_predictors} '
                                 f'does not match the number of predictors specified for forecasting '
                                 f'{num_fut_predictors}. The same set of predictors must be used.')

            if num_periods > num_fut_obs:
                raise ValueError(f'The number of requested forecast periods {num_periods} exceeds the '
                                 f'number of observations provided in future_predictors {num_fut_obs}. '
                                 f'The former must be no larger than the latter.')
            else:
                if num_periods < num_fut_obs:
                    warnings.warn(f'The number of requested forecast periods {num_periods} is less than the '
                                  f'number of observations provided in future_predictors {num_fut_obs}. '
                                  f'Only the first {num_periods} observations will be used '
                                  f'in future_predictors.')

        y_forecast = _forecast(posterior,
                               num_periods,
                               burn,
                               Z,
                               T,
                               R,
                               X_fut)

        return y_forecast

    def plot_components(self, posterior, burn=0, num_fit_ignore=-1, conf_int_level=0.05,
                        random_sample_size_prop=1., smoothed=True):

        if num_fit_ignore == -1:
            num_fit_ignore = self.num_fit_ignore

        if self.num_predictors > 0:
            X = self.predictors[num_fit_ignore:, :]
            reg_coeff = posterior.regression_coefficients[burn:, :, 0].T

        y = self.y[num_fit_ignore:, 0]
        n = self.num_obs
        Z = self.observation_matrix(num_rows=n - num_fit_ignore)
        model = self.model_setup
        components = model.components
        index = np.arange(n - num_fit_ignore)
        conf_int_lb = 0.5 * conf_int_level
        conf_int_ub = 1. - 0.5 * conf_int_level

        filtered_prediction = _simulate_posterior_predictive_response(posterior,
                                                                      burn,
                                                                      num_fit_ignore,
                                                                      random_sample_size_prop)
        smoothed_prediction = posterior.smoothed_prediction[burn:, num_fit_ignore:, 0]

        if smoothed:
            prediction = smoothed_prediction
            state = posterior.smoothed_state[burn:, num_fit_ignore:n, :, :]
        else:
            prediction = filtered_prediction
            state = _simulate_posterior_predictive_state(posterior,
                                                         burn,
                                                         num_fit_ignore,
                                                         random_sample_size_prop,
                                                         self.has_predictors,
                                                         self.static_regression)
        fig, ax = plt.subplots(1 + len(components))
        fig.set_size_inches(12, 10)
        ax[0].plot(y)
        ax[0].plot(np.mean(filtered_prediction, axis=0))
        lb = np.quantile(filtered_prediction, conf_int_lb, axis=0)
        ub = np.quantile(filtered_prediction, conf_int_ub, axis=0)
        ax[0].fill_between(index, lb, ub, alpha=0.2)
        ax[0].title.set_text('Predicted vs. observed response')
        ax[0].legend(('Observed', 'One-step-ahead prediction', f'{100 * (1 - conf_int_level)}% prediction interval'),
                     loc='upper left')
        for i, c in enumerate(components):
            if c == 'Irregular':
                resid = y[np.newaxis] - prediction
                ax[i + 1].plot(np.mean(resid, axis=0))
                lb = np.quantile(resid, conf_int_lb, axis=0)
                ub = np.quantile(resid, conf_int_ub, axis=0)
                ax[i + 1].fill_between(index, lb, ub, alpha=0.2)
                ax[i + 1].title.set_text(c)
                pass

            elif c == 'Regression':
                reg_component = X.dot(reg_coeff)
                ax[i + 1].plot(np.mean(reg_component, axis=1))
                lb = np.quantile(reg_component, conf_int_lb, axis=1)
                ub = np.quantile(reg_component, conf_int_ub, axis=1)
                ax[i + 1].fill_between(index, lb, ub, alpha=0.2)
                ax[i + 1].title.set_text(c)

            elif c == 'Trend':
                time_component = state[:, :, 1, 0]
                ax[i + 1].plot(np.mean(time_component, axis=0))
                lb = np.quantile(time_component, conf_int_lb, axis=0)
                ub = np.quantile(time_component, conf_int_ub, axis=0)
                ax[i + 1].fill_between(index, lb, ub, alpha=0.2)
                ax[i + 1].title.set_text(c)

            else:
                idx = components[c]
                start_index, end_index = idx['start_index'], idx['end_index']
                A = Z[:, 0, start_index:end_index]
                B = state[:, :, start_index:end_index, 0]
                time_component = (A[np.newaxis] * B).sum(axis=2)

                ax[i + 1].plot(np.mean(time_component, axis=0))
                lb = np.quantile(time_component, conf_int_lb, axis=0)
                ub = np.quantile(time_component, conf_int_ub, axis=0)
                ax[i + 1].fill_between(index, lb, ub, alpha=0.2)
                ax[i + 1].title.set_text(c)

        fig.tight_layout()
        plt.show()

        return