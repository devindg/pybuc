import numpy as np
from numpy import dot
from numba import njit
from collections import namedtuple
from numpy.linalg import solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
import warnings

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
                   'outcome_variance',
                   'state_covariance',
                   'outcome_error_variance',
                   'state_error_variance',
                   'regression_coefficients'])

model_setup = namedtuple('model_setup',
                         ['components',
                          'outcome_var_scale_prior',
                          'outcome_var_shape_post',
                          'state_var_scale_prior',
                          'state_var_shape_post',
                          'reg_coeff_mean_prior',
                          'reg_coeff_var_prior',
                          'init_error_variances',
                          'init_state_covariance'])


def is_odd(x: int) -> bool:
    return np.mod(x, 2) != 0


@njit
def set_seed(value):
    np.random.seed(value)


@njit(cache=True)
def _simulate_posterior_predictive_outcome(posterior, burn=0, num_fit_ignore=0,
                                           random_sample_size_prop=1.):
    outcome_mean = posterior.filtered_prediction[burn:, num_fit_ignore:, 0]
    outcome_variance = posterior.outcome_variance[burn:, num_fit_ignore:, 0]
    num_posterior_samp = outcome_mean.shape[0]
    n = outcome_mean.shape[1]

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

    y_post = np.empty((num_samp, n), dtype=np.float64)
    i = 0
    for s in S:
        y_post[i] = dist.vec_norm(outcome_mean[s],
                                  np.sqrt(outcome_variance[s][:, 0]))
        i += 1

    return y_post


def _simulate_posterior_predictive_state_worker(state_mean, state_covariance):
    state_post = (np.random.default_rng()
                  .multivariate_normal(mean=state_mean,
                                       cov=state_covariance,
                                       method='cholesky').reshape(-1, 1))

    return state_post


def _simulate_posterior_predictive_state(posterior, burn=0, num_fit_ignore=0, random_sample_size_prop=1.,
                                         has_regressors=False, static_regression=False):
    if has_regressors and static_regression:
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
              state_error_transformation_matrix, future_regressors=np.array([[]])):
    Z = state_observation_matrix
    T = state_transition_matrix
    R = state_error_transformation_matrix
    X_fut = future_regressors
    m = R.shape[0]
    q = R.shape[1]

    var_eps = posterior.outcome_error_variance[burn:]
    if q > 0:
        var_eta = posterior.state_error_variance[burn:]
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
                                  num_periods_ones * np.sqrt(var_eps[s, 0, 0]))
        if q > 0:
            state_error = dist.vec_norm(num_periods_u_zeros,
                                        num_periods_u_ones * np.sqrt(ao.diag_2d(var_eta[s])))

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
                 outcome: np.ndarray,
                 level: bool = True,
                 stochastic_level: bool = True,
                 slope: bool = False,
                 stochastic_slope: bool = True,
                 seasonal: int = 0,
                 stochastic_seasonal: bool = True,
                 trig_seasonal: tuple[tuple] = (),
                 stochastic_trig_seasonal: tuple = (),
                 standardize: bool = False,
                 regressors: np.ndarray = np.array([[]]),
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
        self.regressors = regressors
        self.static_regression = static_regression

        self.model_setup = None
        self.num_fit_ignore = None

        # TODO: (1) Add functionality for multiple dummy seasonality. (2) Add functionality for autoregressive slope.

        if outcome.ndim != 2:
            raise ValueError('The outcome array must have a row and column dimension. '
                             'Flat/1D arrays or arrays with more than 2 dimensions are not valid.')
        else:
            if outcome.shape[0] > 1 and outcome.shape[1] > 1:
                raise ValueError('The outcome array has at least 2 rows and 2 columns. '
                                 'Only 1 row or 1 column is allowed if there are more than '
                                 '2 columns or 2 rows, respectively. That is, the outcome '
                                 'array must take the form of a vector.')
            else:
                self.outcome = outcome.reshape(-1, 1)

        if self.has_regressors:
            if regressors.ndim != 2:
                raise ValueError('The regressors array must have a row and column dimension. '
                                 'Flat/1D arrays or arrays with more than 2 dimensions are not valid.')
            else:
                if regressors.shape[0] != self.outcome.shape[0]:
                    raise ValueError('The number of observations in the regressor matrix does not match '
                                     'the number of observations in the outcome matrix. The number of '
                                     'observations must be consistent.')
                if np.isnan(regressors).any():
                    raise ValueError('The regressor matrix contains null values, which are not permissible.')
                if np.isinf(regressors).any():
                    raise ValueError('The regressor matrix contains Inf and/or -Inf values, which are not permissible.')

        if seasonal == 1:
            raise ValueError('The seasonal argument takes 0 and integers greater than 1 as valid inputs. '
                             'A value of 1 is not valid.')

        if self.seasonal == 0 and not self.level:
            raise ValueError('At least a level or seasonal component must be specified.')

        if self.slope and not self.level:
            raise ValueError('Slope cannot be specified without a level component.')

        self.check_trig_seasonal()

    def check_trig_seasonal(self):
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
                if is_odd(period):
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
                if is_odd(freq):
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
    def has_regressors(self):
        if self.regressors.size != 0:
            return True
        else:
            return False

    @property
    def num_obs(self):
        return self.outcome.shape[0]

    @property
    def num_seasonal_state_eqs(self):
        return self.num_indicator_seasonal_state_eqs + self.num_trig_seasonal_state_eqs

    @property
    def num_regressors(self):
        if self.regressors.size == 0:
            return 0
        else:
            return self.regressors.shape[1]

    @property
    def num_state_eqs(self):
        return ((self.level + self.slope) * 1
                + self.num_seasonal_state_eqs
                + (self.num_regressors > 0) * 1)

    @property
    def num_stochastic_states(self):
        return (self.level * self.stochastic_level
                + self.slope * self.stochastic_slope
                + (self.seasonal > 1) * self.stochastic_seasonal
                + self.num_stochastic_trig_seasonal_states) * 1

    @property
    def mean_outcome(self):
        return np.nanmean(self.outcome)

    @property
    def sd_outcome(self):
        return np.nanstd(self.outcome)

    @property
    def z_outcome(self):
        return (self.outcome - self.mean_outcome) / self.sd_outcome

    @property
    def y(self):
        if self.standardize:
            return self.z_outcome
        else:
            return self.outcome

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

        # Note that if a regressors component is specified, the observation matrix
        # will vary with time (assuming the regressors vary with time).
        # At this stage of the Kalman Filter setup, however, the regressors
        # component in the observation matrix will be set to 0 for each observation.
        # The 0's will be reassigned based on the prior and posterior means for
        # the regression coefficients.

        V = []
        if self.level and not self.slope:
            V.append(np.array([[1.]]))
        if self.level and self.slope:
            V.append(np.array([[1., 0.]]))
        if self.seasonal > 1:
            s = np.zeros((1, self.seasonal - 1))
            s[0, 0] = 1.
            V.append(s)
        if len(self.trig_seasonal) > 0:
            s = np.zeros((1, self.num_trig_seasonal_state_eqs))
            s[0, 0::2] = 1.
            V.append(s)
        if self.num_regressors > 0:
            V.append(np.array([[0.]]))

        c = 0
        for v in V:
            _, num_cols = v.shape
            Z[:, :, c:c + num_cols] = v
            c += num_cols

        return Z

    def state_transition_matrix(self):
        m = self.num_state_eqs
        T = np.zeros((m, m), dtype=np.float64)

        V = []
        if self.level and not self.slope:
            V.append(np.array([[1.]]))
        if self.level and self.slope:
            V.append(np.array([[1., 1.], [0., 1.]]))
        if self.seasonal > 1:
            s = np.eye(self.seasonal - 1, k=-1)
            s[0, :] = -1.
            V.append(s)
        if len(self.trig_seasonal) > 0:
            for c, w in enumerate(self.trig_seasonal):
                period, freq = w
                for j in range(1, freq + 1):
                    V.append(self.trig_transition_matrix(2. * np.pi * j / period))
        if self.num_regressors > 0:
            V.append(np.array([[1.]]))

        r, c = 0, 0
        for v in V:
            num_rows, num_cols = v.shape
            T[r:r + num_rows, c:c + num_cols] = v
            r += num_rows
            c += num_cols

        return T

    def state_error_transformation_matrix(self):
        m = self.num_state_eqs
        q = self.num_stochastic_states
        R = np.zeros((m, q), dtype=np.float64)

        if q == 0:
            pass
        else:
            V = []
            if self.level:
                if self.stochastic_level:
                    V.append(np.array([[1.]]))
                else:
                    V.append(np.array([[]]))
            if self.slope:
                if self.stochastic_slope:
                    V.append(np.array([[1.]]))
                else:
                    V.append(np.array([[]]))
            if self.seasonal > 1:
                if self.stochastic_seasonal:
                    V.append(np.array([[1.]]))
                else:
                    V.append(np.array([[]]))
            if len(self.trig_seasonal) > 0:
                for c, w in enumerate(self.trig_seasonal):
                    _, freq = w
                    if self.stochastic_trig_seasonal[c]:
                        V.append(np.eye(2 * freq))
                    else:
                        V.append(np.array([[]]))

            r, c = 0, 0
            for v in V:
                if v.size == 0:
                    num_rows, num_cols = 1, 0
                else:
                    num_rows, num_cols = v.shape
                    R[r:r + num_rows, c:c + num_cols] = v
                r += num_rows
                c += num_cols

        return R

    def posterior_state_error_variance_transformation_matrix(self):
        q = self.num_stochastic_states

        if q == 0:
            return np.array([[]])
        else:
            if self.num_stochastic_trig_seasonal_states == 0:
                A = np.eye(q)
            else:
                A = np.zeros((q, q), dtype=np.float64)
                V = []
                if self.level:
                    if self.stochastic_level:
                        V.append(np.array([[1.]]))
                    else:
                        V.append(np.array([[]]))
                if self.slope:
                    if self.stochastic_slope:
                        V.append(np.array([[1.]]))
                    else:
                        V.append(np.array([[]]))
                if self.seasonal > 1:
                    if self.stochastic_seasonal:
                        V.append(np.array([[1.]]))
                    else:
                        V.append(np.array([[]]))
                if len(self.trig_seasonal) > 0:
                    for c, w in enumerate(self.trig_seasonal):
                        _, freq = w
                        if self.stochastic_trig_seasonal[c]:
                            V.append(np.ones((2 * freq, 2 * freq)) / (2 * freq))
                        else:
                            V.append(np.array([[]]))

                r, c = 0, 0
                for v in V:
                    if v.size == 0:
                        num_rows, num_cols = 0, 0
                    else:
                        num_rows, num_cols = v.shape
                        A[r:r + num_rows, c:c + num_cols] = v
                    r += num_rows
                    c += num_cols

            return A

    def _model_setup(self, outcome_var_shape_prior, outcome_var_scale_prior,
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
        if np.isnan(outcome_var_shape_prior):
            outcome_var_shape_prior = 1.
        if np.isnan(outcome_var_scale_prior):
            outcome_var_scale_prior = 0.01

        outcome_var_shape_post = np.array([[outcome_var_shape_prior + 0.5 * n]])
        init_outcome_error_var = [0.01 * self.sd_outcome ** 2]

        state_var_scale_prior = []
        state_var_shape_post = []
        init_state_error_var = []
        init_state_variances = []
        num_fit_ignore = [1]
        j = 0
        if self.level:
            if self.stochastic_level:
                if np.isnan(level_var_shape_prior):
                    level_var_shape_prior = 1.

                if np.isnan(level_var_scale_prior):
                    level_var_scale_prior = 0.01

                state_var_shape_post.append(level_var_shape_prior + 0.5 * n)
                state_var_scale_prior.append(level_var_scale_prior)
                init_state_error_var.append(0.01 * self.sd_outcome ** 2)

            init_state_variances.append(1e6)
            components['Level'] = dict(start_index=j, end_index=j + 1)
            j += 1

        if self.slope:
            num_fit_ignore.append(1)
            if self.stochastic_slope:
                if np.isnan(slope_var_shape_prior):
                    slope_var_shape_prior = 1.

                if np.isnan(slope_var_scale_prior):
                    slope_var_scale_prior = 0.01

                state_var_shape_post.append(slope_var_shape_prior + 0.5 * n)
                state_var_scale_prior.append(slope_var_scale_prior)
                init_state_error_var.append(0.01 * self.sd_outcome ** 2)

            components['Trend'] = dict()
            init_state_variances.append(1e6)
            j += 1

        if self.seasonal > 1:
            num_fit_ignore.append(self.seasonal)
            if self.stochastic_seasonal:
                if np.isnan(season_var_shape_prior):
                    season_var_shape_prior = 1.

                if np.isnan(season_var_scale_prior):
                    season_var_scale_prior = 0.01

                state_var_shape_post.append(season_var_shape_prior + 0.5 * n)
                state_var_scale_prior.append(season_var_scale_prior)
                init_state_error_var.append(0.01 * self.sd_outcome ** 2)

            for k in range(self.seasonal - 1):
                init_state_variances.append(1e6)

            components[f'Seasonal.{self.seasonal}'] = dict(start_index=j,
                                                           end_index=j + (self.seasonal - 1) + 1)
            j += self.seasonal - 1

        if len(self.trig_seasonal) > 0:
            if True in self.stochastic_trig_seasonal:
                if np.isnan(trig_season_var_shape_prior):
                    trig_season_var_shape_prior = 1.

                if np.isnan(trig_season_var_scale_prior):
                    trig_season_var_scale_prior = 0.01

            i = j
            for c, v in enumerate(self.trig_seasonal):
                f, h = v
                num_terms = 2 * h
                num_fit_ignore.append(num_terms)
                if self.stochastic_trig_seasonal[c]:
                    for k in range(num_terms):
                        state_var_shape_post.append(trig_season_var_shape_prior + 0.5 * n)
                        state_var_scale_prior.append(trig_season_var_scale_prior)
                        init_state_error_var.append(0.01 * self.sd_outcome ** 2)

                for k in range(num_terms):
                    init_state_variances.append(1e6)
                components[f'Trigonometric-Seasonal.{f}.{h}'] = dict(start_index=i,
                                                                     end_index=i + num_terms + 1)
                i += 2 * h
            j += self.num_trig_seasonal_state_eqs

        if self.num_regressors > 0:
            components['Regression'] = dict()
            X = self.regressors
            init_state_variances.append(0.)

            if reg_coeff_mean_prior.size == 0:
                print('No mean prior was provided for the regression coefficient vector. '
                      'A 0-mean prior will be enforced.')
                reg_coeff_mean_prior = np.zeros((self.num_regressors, 1))

            if reg_coeff_var_prior.size == 0:
                kappa = 1.
                g = kappa / n
                print('No variance prior was provided for the regression coefficient vector. '
                      'A g=1/n Zellner g-prior will be enforced.')
                reg_coeff_var_prior = g * (0.5 * dot(X.T, X) + 0.5 * np.diag(dot(X.T, X)))

        self.num_fit_ignore = 1 + max(num_fit_ignore)
        state_var_shape_post = np.array(state_var_shape_post).reshape(-1, 1)
        state_var_scale_prior = np.array(state_var_scale_prior).reshape(-1, 1)
        init_error_variances = np.concatenate((init_outcome_error_var, init_state_error_var))
        init_state_covariance = np.diag(init_state_variances)

        self.model_setup = model_setup(components,
                                       outcome_var_scale_prior,
                                       outcome_var_shape_post,
                                       state_var_scale_prior,
                                       state_var_shape_post,
                                       reg_coeff_mean_prior,
                                       reg_coeff_var_prior,
                                       init_error_variances,
                                       init_state_covariance)

        return self.model_setup

    def sample(self, num_samp=1000,
               outcome_var_shape_prior=np.nan, outcome_var_scale_prior=np.nan,
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
        X = self.regressors
        k = self.num_regressors

        model = self._model_setup(outcome_var_shape_prior, outcome_var_scale_prior,
                                  level_var_shape_prior, level_var_scale_prior,
                                  slope_var_shape_prior, slope_var_scale_prior,
                                  season_var_shape_prior, season_var_scale_prior,
                                  trig_season_var_shape_prior, trig_season_var_scale_prior,
                                  reg_coeff_mean_prior, reg_coeff_var_prior)

        outcome_var_scale_prior = model.outcome_var_scale_prior
        outcome_var_shape_post = model.outcome_var_shape_post
        state_var_scale_prior = model.state_var_scale_prior
        state_var_shape_post = model.state_var_shape_post
        init_error_variances = model.init_error_variances
        init_state_covariance = model.init_state_covariance

        # Initialize arrays for variances and state vector
        if q > 0:
            state_error_variance = np.empty((num_samp, q, q), dtype=np.float64)
        else:
            state_error_variance = np.empty((num_samp, 0, 0))

        outcome_error_variance = np.empty((num_samp, 1, 1), dtype=np.float64)
        smoothed_errors = np.empty((num_samp, n, 1 + q, 1), dtype=np.float64)
        smoothed_state = np.empty((num_samp, n + 1, m, 1), dtype=np.float64)
        smoothed_prediction = np.empty((num_samp, n, 1), dtype=np.float64)
        filtered_state = np.empty((num_samp, n + 1, m, 1), dtype=np.float64)
        filtered_prediction = np.empty((num_samp, n, 1), dtype=np.float64)
        state_covariance = np.empty((num_samp, n + 1, m, m), dtype=np.float64)
        outcome_variance = np.empty((num_samp, n, 1, 1), dtype=np.float64)
        init_plus_state_values = np.zeros((m, 1))
        init_state_values0 = np.zeros((m, 1))

        outcome_error_variance[0] = np.array([[init_error_variances[0]]])
        if q > 0:
            state_error_variance[0] = np.diag(init_error_variances[1:])

        # Helper matrices
        q_eye = np.eye(q)
        n_ones = np.ones((n, 1))

        if self.num_regressors > 0:
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
            outcome_err_var = outcome_error_variance[s - 1]
            state_err_var = state_error_variance[s - 1]

            if self.num_regressors > 0:
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
                      outcome_err_var,
                      state_err_var,
                      init_state=init_state_values,
                      init_state_covariance=init_state_covariance)

            filtered_state[s] = y_kf.filtered_state
            state_covariance[s] = y_kf.state_covariance
            filtered_prediction[s] = y - y_kf.one_step_ahead_prediction_residual[:, :, 0]
            outcome_variance[s] = y_kf.outcome_variance

            # Get smoothed state from DK smoother
            dk = dks(y,
                     Z,
                     T,
                     R,
                     outcome_err_var,
                     state_err_var,
                     init_plus_state_values=init_plus_state_values,
                     init_state_values=init_state_values,
                     init_state_covariance=init_state_covariance,
                     static_regression=(self.static_regression * (self.num_regressors > 0)))

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
            outcome_var_scale_post = (outcome_var_scale_prior
                                      + 0.5 * dot(smooth_one_step_ahead_prediction_residual.T,
                                                  smooth_one_step_ahead_prediction_residual))
            outcome_error_variance[s] = dist.vec_ig(outcome_var_shape_post, outcome_var_scale_post)

            if self.num_regressors > 0:
                # Get new draw for regression coefficients
                y_adj = y_no_nan + y_nan_indicator * smoothed_prediction[s]
                smooth_time_prediction = smoothed_prediction[s] - Z[:, :, -1]
                y_tilde = y_adj - smooth_time_prediction  # y with smooth time prediction subtracted out
                reg_coeff_mean_post = dot(reg_coeff_var_post,
                                          (dot(X.T, y_tilde) + dot(reg_coeff_var_inv_prior, reg_coeff_mean_prior)))

                regression_coefficients[s] = (np
                                              .random.default_rng()
                                              .multivariate_normal(mean=reg_coeff_mean_post.flatten(),
                                                                   cov=outcome_error_variance[
                                                                           s, 0, 0] * reg_coeff_var_post,
                                                                   method='cholesky').reshape(-1, 1))

        results = post(num_samp, smoothed_state, smoothed_errors, smoothed_prediction,
                       filtered_state, filtered_prediction, outcome_variance, state_covariance,
                       outcome_error_variance, state_error_variance, regression_coefficients)

        return results

    def forecast(self, posterior, num_periods, burn=0, future_regressors: np.ndarray = np.array([[]])):
        Z = self.observation_matrix(num_rows=num_periods)
        T = self.state_transition_matrix()
        R = self.state_error_transformation_matrix()
        X_fut = future_regressors

        if self.num_regressors > 0:
            num_fut_obs, num_fut_regressors = X_fut.shape

            if self.num_regressors != num_fut_regressors:
                raise ValueError(f'The number of regressors used for historical estimation {self.num_regressors} '
                                 f'does not match the number of regressors specified for forecasting '
                                 f'{num_fut_regressors}. The same set of regressors must be used.')

            if num_periods > num_fut_obs:
                raise ValueError(f'The number of requested forecast periods {num_periods} exceeds the '
                                 f'number of observations provided in future_regressors {num_fut_obs}. '
                                 f'The former must be no larger than the latter.')
            else:
                if num_periods < num_fut_obs:
                    warnings.warn(f'The number of requested forecast periods {num_periods} is less than the '
                                  f'number of observations provided in future_regressors {num_fut_obs}. '
                                  f'Therefore, only the first {num_periods} observations will be used '
                                  f'in future_regressors.')

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

        if self.num_regressors > 0:
            X = self.regressors[num_fit_ignore:, :]
            reg_coeff = posterior.regression_coefficients[burn:, :, 0].T

        y = self.y[num_fit_ignore:, 0]
        n = self.num_obs
        Z = self.observation_matrix(num_rows=n - num_fit_ignore)
        model = self.model_setup
        components = model.components
        index = np.arange(n - num_fit_ignore)
        conf_int_lb = 0.5 * conf_int_level
        conf_int_ub = 1. - 0.5 * conf_int_level

        filtered_prediction = _simulate_posterior_predictive_outcome(posterior,
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
                                                         self.has_regressors,
                                                         self.static_regression)
        fig, ax = plt.subplots(1 + len(components))
        fig.set_size_inches(12, 10)
        ax[0].plot(y)
        ax[0].plot(np.mean(filtered_prediction, axis=0))
        lb = np.quantile(filtered_prediction, conf_int_lb, axis=0)
        ub = np.quantile(filtered_prediction, conf_int_ub, axis=0)
        ax[0].fill_between(index, lb, ub, alpha=0.2)
        ax[0].title.set_text('Observed Outcome')
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
