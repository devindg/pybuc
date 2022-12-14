import numpy as np
from numpy import dot
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
    damped_level_coefficient: np.ndarray
    damped_trend_coefficient: np.ndarray
    damped_season_coefficients: np.ndarray
    regression_coefficients: np.ndarray


class ModelSetup(NamedTuple):
    components: dict
    response_var_scale_prior: float
    response_var_shape_post: np.ndarray
    state_var_scale_prior: np.ndarray
    state_var_shape_post: np.ndarray
    gibbs_iter0_init_state: np.ndarray
    gibbs_iter0_response_error_variance: np.ndarray
    gibbs_iter0_state_error_covariance: np.ndarray
    init_state_plus_values: np.ndarray
    init_state_covariance: np.ndarray
    damped_level_coeff_mean_prior: np.ndarray
    damped_level_coeff_prec_prior: np.ndarray
    damped_level_coeff_cov_prior: np.ndarray
    gibbs_iter0_damped_level_coeff: np.ndarray
    damped_trend_coeff_mean_prior: np.ndarray
    damped_trend_coeff_prec_prior: np.ndarray
    damped_trend_coeff_cov_prior: np.ndarray
    gibbs_iter0_damped_trend_coeff: np.ndarray
    damped_lag_season_coeff_mean_prior: np.ndarray
    damped_lag_season_coeff_prec_prior: np.ndarray
    damped_lag_season_coeff_cov_prior: np.ndarray
    gibbs_iter0_damped_season_coeff: np.ndarray
    reg_coeff_mean_prior: np.ndarray
    reg_coeff_prec_prior: np.ndarray
    reg_coeff_cov_prior: np.ndarray
    reg_coeff_cov_post: np.ndarray
    reg_coeff_prec_post: np.ndarray
    zellner_prior_obs: float
    gibbs_iter0_reg_coeff: np.ndarray


@njit
def _set_seed(value):
    np.random.seed(value)


def _ar_state_post_upd(smoothed_state: np.ndarray,
                       lag: tuple,
                       mean_prior: np.ndarray,
                       precision_prior: np.ndarray,
                       state_err_var_post: np.ndarray,
                       state_eqn_index: list,
                       state_err_var_post_index: list):
    """

    :param smoothed_state: ndarray, float64. Array that represents that smoothed values for each state
    in an unobserved components model.

    :param lag: tuple of positive integers. An integer represents an autoregressive lag of the
    state variable.

    :param mean_prior: ndarray, float64. An element represents the mean of a Gaussian
    random variable. Row length must match the number of lags.

    :param precision_prior: ndarray, float64. An element represents the precision of a Gaussian
    random variable. Row length must match the number of lags.

    :param state_err_var_post: ndarray, float64. An element represents the posterior variance
    of a state's error.

    :param state_eqn_index: list of non-negative integers. An integer represents the row index of a
    state within an unobserved components model.

    :param state_err_var_post_index: list of non-negative integers. An integer represents the row index
    of a state's posterior error variance (i.e., a row index for state_err_var_post).


    :return: ndarray of dimension (L, 1), where L is the number of lags in the lag tuple. An
    element represents the posterior autoregressive coefficient for an autoregressive process of the
    form y(t) = rho * y(t - l) + epsilon(t), where l is some lag.
    """

    ar_coeff_mean_post = np.empty((len(lag), 1))
    cov_post = np.empty((len(lag), 1))

    c = 0
    for i in lag:
        y = smoothed_state[:, state_eqn_index[c]][i:]
        y_lag = np.roll(smoothed_state[:, state_eqn_index[c]], i)[i:]
        ar_coeff_cov_post = ao.mat_inv(dot(y_lag.T, y_lag) + precision_prior[c])
        ar_coeff_mean_post[c] = dot(ar_coeff_cov_post,
                                    dot(y_lag.T, y)
                                    + dot(precision_prior[c], mean_prior[c]))
        cov_post[c] = state_err_var_post[state_err_var_post_index[c], 0] * ar_coeff_cov_post
        c += 1

    ar_coeff_post = dist.vec_norm(ar_coeff_mean_post, np.sqrt(cov_post))

    return ar_coeff_post


@njit(cache=True)
def _simulate_posterior_predictive_response(posterior: Posterior,
                                            burn: int = 0,
                                            num_first_obs_ignore: int = 0,
                                            random_sample_size_prop: float = 1.) -> np.ndarray:
    """

    :param posterior: NamedTuple with attributes:

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
    damped_trend_coefficient: np.ndarray
    damped_season_coefficients: np.ndarray
    regression_coefficients: np.ndarray

    :param burn: non-negative integer. Represents how many of the first posterior samples to
    ignore for computing statistics like the mean and variance of a parameter. Default value is 0.

    :param num_first_obs_ignore: non-negative integer. Represents how many of the first observations
    of a response variable to ignore for computing the posterior predictive distribution of the
    response. The number of observations to ignore depends on the state specification of the unobserved
    components model. Some observations are ignored because diffuse initialization is used for the
    state vector in the Kalman filter. Default value is 0.

    :param random_sample_size_prop: float in interval (0, 1]. Represents the proportion of the
    posterior samples to take for constructing the posterior predictive distribution. Sampling is
    done without replacement. Default value is 1.

    :return: ndarray of dimension (S, n), where S is the number of posterior samples and
    n is the number of observations used for constructing the posterior predictive distribution.
    Note that S = random_sample_size_prop * S_total, where S_total is total number of
    posterior samples, and n = n_tot - num_first_obs_ignore, where n_tot is the total number
    of observations for the response.
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
    """

    :param state_mean: ndarray of dimension (m, 1), where m is the number of state equations.
    Data type if float64.
    :param state_covariance: ndarray of dimension (m, m). Data type if float64.
    :return:
    """
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

    :param posterior: NamedTuple with attributes:

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
    damped_level_coefficient: np.ndarray
    damped_trend_coefficient: np.ndarray
    damped_season_coefficients: np.ndarray
    regression_coefficients: np.ndarray

    :param burn: non-negative integer. Represents how many of the first posterior samples to
    ignore for computing statistics like the mean and variance of a parameter. Default value is 0.

    :param num_first_obs_ignore: non-negative integer. Represents how many of the first observations
    of a response variable to ignore for computing the posterior predictive distribution of the
    response. The number of observations to ignore depends on the state specification of the unobserved
    components model. Some observations are ignored because diffuse initialization is used for the
    state vector in the Kalman filter. Default value is 0.

    :param random_sample_size_prop: float in interval (0, 1]. Represents the proportion of the
    posterior samples to take for constructing the posterior predictive distribution. Sampling is
    done without replacement. Default value is 1.

    :param has_predictors: bool. If True,

    :return: ndarray with shape (S, n, m, 1), where S is the number of posterior samples
    drawn from the full posterior distribution, n is the number of observations for the response
    variable, and m is the number of state equations. For a given sample s in S, there are n
    predictions for each of the m state equations. This array represents the posterior predictive
    distribution for each of the m state equations.
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
              state_intercept_matrix: np.ndarray,
              state_error_transformation_matrix: np.ndarray,
              future_predictors: np.ndarray = np.array([[]]),
              burn: int = 0,
              damped_level_transition_index: tuple = (),
              damped_trend_transition_index: tuple = (),
              damped_season_transition_index: tuple = ()) -> np.ndarray:
    """

    :param posterior: NamedTuple with attributes:

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
    damped_level_coefficient: np.ndarray
    damped_trend_coefficient: np.ndarray
    damped_season_coefficients: np.ndarray
    regression_coefficients: np.ndarray

    :param num_periods: positive integer. Defines how many future periods to forecast.

    :param state_observation_matrix: ndarray of dimension (n, 1, m), where m is the
    number of state equations. Data type must be float64. Matrix that maps the
    state to the response.

    :param state_transition_matrix: ndarray of dimension (m, m). Data type must
    be float64. Matrix that maps previous state values to the current state.

    :param state_error_transformation_matrix: ndarray of dimension (m, q), where
    q is the number of state equations that evolve stochastically. Data type must
    be float64. Matrix that maps state equations to stochastic or non-stochastic form.

    :param future_predictors: ndarray of dimension (k, 1), where k is the number of predictors.
    Data type is float64. Default is an empty array.

    :param burn: non-negative integer. Represents how many of the first posterior samples to
    ignore for computing statistics like the mean and variance of a parameter. Default value is 0.

    :param damped_level_transition_index: tuple of the form (i, j), where i and j index
    the row and column, respectively, of the state transition matrix corresponding to the
    address of the level component's autoregressive coefficient. This index is used when a
    damped level is specified. Specifically, the element in the state transition matrix at
    address (i, j) is assigned the values from the posterior distribution of the level's
    AR(1) coefficient. Default is an empty tuple.

    :param damped_trend_transition_index: tuple of the form (i, j), where i and j index
    the row and column, respectively, of the state transition matrix corresponding to the
    address of the trend component's autoregressive coefficient. This index is used when a
    damped trend is specified. Specifically, the element in the state transition matrix at
    address (i, j) is assigned the values from the posterior distribution of the trend's
    AR(1) coefficient. Default is an empty tuple.

    :param damped_season_transition_index: tuple of the form ((i_0, j_0), (i_1, j_1), ..., (i_P, j_P)),
    where i_p and j_p index the row and column, respectively, of the state transition matrix
    corresponding to the address of the p-th periodic-lag seasonal component's autoregressive coefficient.
    This index is used when damped periodic-lag seasonality is specified. Specifically, the element in
    the state transition matrix at address (i_p, j_p) is assigned the values from the posterior distribution
    of the p-th periodic-lag seasonal component's AR(1) coefficient. Default is an empty tuple.

    :return: ndarray of dimension (S, h, 1), where S is the number of posterior samples and
    h is the number of future periods to be forecast. This array represents the posterior
    forecast distribution for the response.
    """

    Z = state_observation_matrix
    T = state_transition_matrix
    C = state_intercept_matrix
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

    if len(damped_level_transition_index) > 0:
        damped_level_coeff = posterior.damped_level_coefficient[burn:]

    if len(damped_trend_transition_index) > 0:
        damped_trend_coeff = posterior.damped_trend_coefficient[burn:]

    if len(damped_season_transition_index) > 0:
        damped_season_coeff = posterior.damped_season_coefficients[burn:]

    y_forecast = np.empty((num_samp, num_periods, 1), dtype=np.float64)
    num_periods_zeros = np.zeros((num_periods, 1))
    num_periods_u_zeros = np.zeros((num_periods, q, 1))
    num_periods_ones = np.ones((num_periods, 1))
    num_periods_u_ones = np.ones((num_periods, q, 1))

    for s in range(num_samp):
        obs_error = dist.vec_norm(num_periods_zeros,
                                  num_periods_ones
                                  * np.sqrt(response_error_variance[s][0, 0]))
        if q > 0:
            state_error = dist.vec_norm(num_periods_u_zeros,
                                        num_periods_u_ones
                                        * np.sqrt(ao.diag_2d(state_error_covariance[s])))

        if len(damped_level_transition_index) > 0:
            T[damped_level_transition_index] = damped_level_coeff[s][0, 0]

        if len(damped_trend_transition_index) > 0:
            T[damped_trend_transition_index] = damped_trend_coeff[s][0, 0]

        if len(damped_season_transition_index) > 0:
            c = 0
            for j in damped_season_transition_index:
                T[j] = damped_season_coeff[s][c, 0]
                c += 1

        if X_fut.size > 0:
            Z[:, :, -1] = X_fut.dot(reg_coeff[s])

        alpha = np.empty((num_periods + 1, m, 1), dtype=np.float64)
        alpha[0] = smoothed_state[s, -1]
        for t in range(num_periods):
            y_forecast[s, t] = Z[t].dot(alpha[t]) + obs_error[t]
            if q > 0:
                alpha[t + 1] = C + T.dot(alpha[t]) + R.dot(state_error[t])
            else:
                alpha[t + 1] = C + T.dot(alpha[t])

    return y_forecast


class BayesianUnobservedComponents:
    def __init__(self,
                 response: Union[np.ndarray, pd.Series, pd.DataFrame],
                 predictors: Union[np.ndarray, pd.Series, pd.DataFrame] = np.array([[]]),
                 level: bool = False,
                 stochastic_level: bool = True,
                 damped_level: bool = False,
                 trend: bool = False,
                 stochastic_trend: bool = True,
                 damped_trend: bool = False,
                 lag_seasonal: tuple = (),
                 stochastic_lag_seasonal: tuple = (),
                 damped_lag_seasonal: tuple = (),
                 dummy_seasonal: tuple = (),
                 stochastic_dummy_seasonal: tuple = (),
                 trig_seasonal: tuple = (),
                 stochastic_trig_seasonal: tuple = (),
                 seed: int = None):

        """
        Test this
        :param response: Numpy array, Pandas Series, or Pandas DataFrame, float64. Array that represents
        the response variable.

        :param predictors: Numpy array, Pandas Series, or Pandas DataFrame, float64. Array that represents
        the predictors, if any, to be used for predicting the response variable. Default is an empty array.

        :param level: bool. If true, a level component is added to the model. Default is false.

        :param stochastic_level: bool. If true, the level component evolves stochastically. Default is true.

        :param damped_level: bool. If true, the level obeys an autoregressive process of order 1.
        Note that stochastic_level must be true if damped_level is true. Default is false.

        :param trend: bool. If true, a trend component is added to the model. Note, that a trend
        is applicable only when a level component is specified. Default is false.

        :param stochastic_trend: bool. If true, the trend component evolves stochastically. Default is true.

        :param damped_trend: bool. If true, the trend obeys an autoregressive process of order 1.
        Note that stochastic_trend must be true if damped_trend is true. Default is false.

        :param lag_seasonal: tuple of integers. Each integer in the tuple represents a distinct
        form of seasonality/periodicity in the data. Default is an empty tuple, i.e., no
        seasonality.

        :param stochastic_lag_seasonal: tuple of bools. Each boolean in the tuple specifies whether the
        corresponding periodicities in lag_seasonal evolve stochastically. Default is an empty tuple,
        which will be converted to true bools if lag_seasonal is not empty.

        :param damped_lag_seasonal: tuple of bools. Each boolean in the tuple specifies whether the
        corresponding periodicities in lag_seasonal evolve according to an autoregressive process of order 1.
        Note that any periodicity in lag_seasonal that is specified with a damping factor must also be
        stochastic. E.g., if lag_seasonal = (12, 4) and damped_lag_seasonal = (True, False), then
        stochastic_lag_seasonal = (True, False) or stochastic_lag_seasonal = (True, True). Default is false
        for all periodicities.

        :param dummy_seasonal: tuple of integers. Each integer in the tuple represents a distinct
        form of dummy seasonality/periodicity in the data. Default is an empty tuple, i.e., no
        dummy seasonality.

        :param stochastic_dummy_seasonal: tuple of bools. Each boolean in the tuple specifies whether the
        corresponding periodicities in dummy_seasonal evolve stochastically. Default is an empty tuple,
        which will be converted to true bools if dummy_seasonal is not empty.

        :param trig_seasonal: tuple of 2-tuples that takes the form ((periodicity_1, num_harmonics_1),
        (periodicity_2, num_harmonics_2), ...). Each (periodicity, num_harmonics) pair in trig_seasonal
        specifies a periodicity and number of harmonics associated with the periodicity. For example,
        (12, 6) would specify a periodicity of 12 with 6 harmonics. The number of harmonics must be an
        integer in [1, periodicity / 2] if periodicity is even, or in [1, (periodicity - 1) / 2]
        if periodicity is odd. Each period specified must be distinct. For any given periodicity specified,
        if 0 is entered for the number of harmonics, then the maximum number of harmonics will be used
        (e.g., if ((7, 0),) is passed to trig_seasonal, then (7 - 1) / 2 = 3 harmonics will be used).

        :param stochastic_trig_seasonal: tuple of bools. Each boolean in the tuple specifies whether the
        corresponding (periodicity, num_harmonics) in trig_seasonal evolve stochastically. Default is an
        empty tuple, which will be converted to true bools if trig_seasonal is not empty.
        """

        self._response = response
        self._predictors = predictors
        self.model_setup = None
        self.response_name = None
        self.predictors_names = None
        self.historical_time_index = None
        self.future_time_index = None
        self.num_first_obs_ignore = None
        self.posterior = None
        self.damped_level_transition_index = ()
        self.damped_trend_transition_index = ()
        self.damped_lag_season_transition_index = ()

        # CHECK AND PREPARE RESPONSE DATA
        # -- data types, name, and index
        if not isinstance(response, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError("The response array must be a Numpy array, Pandas Series, or Pandas DataFrame.")
        else:
            resp = response.copy()
            if isinstance(response, (pd.Series, pd.DataFrame)):
                if not isinstance(response.index, pd.DatetimeIndex):
                    raise TypeError("Pandas' DatetimeIndex is currently the only supported "
                                    "index type for Pandas objects.")
                else:
                    if response.index.freq is None:
                        warnings.warn('Frequency of DatetimeIndex is None. Frequency will be inferred for response.')
                        response.index.freq = pd.infer_freq(response.index)

                if isinstance(response, pd.Series):
                    self.response_name = [response.name]
                else:
                    self.response_name = response.columns.values.tolist()

                self.historical_time_index = response.index
                resp = resp.sort_index().to_numpy()

        # -- dimensions
        if resp.dtype != 'float64':
            raise TypeError('All values in the response array must be of type float.')
        if resp.ndim == 0:
            raise ValueError('The response array must have dimension 1 or 2.')
        if resp.ndim == 1:
            resp = resp.reshape(-1, 1)
        if resp.ndim > 2:
            raise ValueError('The response array must have dimension 1 or 2.')
        if resp.ndim == 2:
            if all(i > 1 for i in resp.shape):
                raise ValueError('The response array must have shape (1, n) or (n, 1), '
                                 'where n is the number of observations. Both the row and column '
                                 'count exceed 1.')
            else:
                resp = resp.reshape(-1, 1)

        # Null check
        if np.all(np.isnan(resp)):
            raise ValueError('All values in the response array are null. At least one value must be non-null.')

        if resp.shape[0] < 2:
            raise ValueError('At least two observations are required to fit a model.')

        if np.sum(np.isnan(resp) * 1) / resp.shape[0] >= 0.1:
            warnings.warn('At least 10% of values in the response array are null. Predictions from the model may be '
                          'significantly compromised.')

        # CHECK AND PREPARE PREDICTORS DATA, IF APPLICABLE
        # -- check if correct data type
        if not isinstance(predictors, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError("The predictors array must be a Numpy array, Pandas Series, or Pandas DataFrame.")
        else:
            pred = predictors.copy()
            if predictors.size > 0:
                # -- check if response and predictors are same date type.
                # if not isinstance(predictors, type(response)):
                #     raise ValueError('Object types for response and predictors arrays must match.')
                if isinstance(response, np.ndarray) and not isinstance(predictors, np.ndarray):
                    raise TypeError('The response array provided is a NumPy array, but the predictors '
                                    'array is not. Object types must match.')

                if (isinstance(response, (pd.Series, pd.DataFrame)) and
                        not isinstance(predictors, (pd.Series, pd.DataFrame))):
                    raise TypeError('The response array provided is a Pandas Series/DataFrame, but the predictors '
                                    'array is not. Object types must match.')

                # -- get predictor names if a Pandas object and sort index
                if isinstance(predictors, (pd.Series, pd.DataFrame)):
                    if not isinstance(predictors.index, pd.DatetimeIndex):
                        raise TypeError("Pandas' DatetimeIndex is currently the only supported "
                                        "index type for Pandas objects.")
                    else:
                        if not (predictors.index == self.historical_time_index).all():
                            raise ValueError('The response and predictors indexes must match.')

                    if isinstance(predictors, pd.Series):
                        self.predictors_names = [predictors.name]
                    else:
                        self.predictors_names = predictors.columns.values.tolist()

                    pred = pred.sort_index().to_numpy()

                # -- dimension and null/inf checks
                if pred.dtype != 'float64':
                    raise TypeError('All values in the predictors array must be of type float.')
                if pred.ndim == 0:
                    raise ValueError('The predictors array must have dimension 1 or 2. Dimension is 0.')
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                if pred.ndim > 2:
                    raise ValueError('The predictors array must have dimension 1 or 2. Dimension exceeds 2.')
                if pred.ndim == 2:
                    if 1 in pred.shape:
                        pred = pred.reshape(-1, 1)
                if np.any(np.isnan(pred)):
                    raise ValueError('The predictors array cannot have null values.')
                if np.any(np.isinf(pred)):
                    raise ValueError('The predictors array cannot have Inf and/or -Inf values.')

                # -- conformable number of observations
                if pred.shape[0] != resp.shape[0]:
                    raise ValueError('The number of observations in the predictors array must match '
                                     'the number of observations in the response array.')

                # -- check if design matrix has a constant
                pred_column_mean = np.mean(pred, axis=0)
                pred_offset = pred - pred_column_mean[np.newaxis, :]
                diag_pred_offset_squared = np.diag(dot(pred_offset.T, pred_offset))
                if np.any(diag_pred_offset_squared == 0.):
                    raise ValueError('The predictors array cannot have a column with a constant value. Note that '
                                     'the inclusion of a constant/intercept in the predictors array will confound '
                                     'with the level if specified. If a constant level without trend or seasonality '
                                     'is desired, pass as arguments level=True, stochastic_level=False, trend=False, '
                                     'dum_seasonal=(), and trig_seasonal=(). This will replicate standard regression.')

                # -- warn about model stability if the number of predictors exceeds number of observations
                if pred.shape[1] > pred.shape[0]:
                    warnings.warn('The number of predictors exceeds the number of observations. '
                                  'Results will be sensitive to choice of priors.')

            # CHECK AND PREPARE LAG SEASONAL
            if not isinstance(lag_seasonal, tuple):
                raise TypeError('lag_seasonal must be a tuple.')

            if not isinstance(stochastic_lag_seasonal, tuple):
                raise TypeError('stochastic_lag_seasonal must be a tuple.')

            if not isinstance(damped_lag_seasonal, tuple):
                raise TypeError('damped_lag_seasonal must be a tuple.')

            if len(lag_seasonal) > 0:
                if not all(isinstance(v, int) for v in lag_seasonal):
                    raise TypeError('The period for a lag_seasonal component must be an integer.')

                if len(lag_seasonal) != len(set(lag_seasonal)):
                    raise ValueError('Each specified period in lag_seasonal must be distinct.')

                if any(v < 2 for v in lag_seasonal):
                    raise ValueError('The period for a lag_seasonal component must be an integer greater than 1.')

                if len(stochastic_lag_seasonal) > 0:
                    if not all(isinstance(v, bool) for v in stochastic_lag_seasonal):
                        raise TypeError('If a non-empty tuple is passed for the stochastic specification '
                                        'of the lag_seasonal components, all elements must be of boolean type.')

                    if len(lag_seasonal) > len(stochastic_lag_seasonal):
                        raise ValueError('Some of the lag_seasonal components were given a stochastic '
                                         'specification, but not all. Partial specification of the stochastic '
                                         'profile is not allowed. Either leave the stochastic specification blank '
                                         'by passing an empty tuple (), which will default to True '
                                         'for all components, or pass a stochastic specification '
                                         'for each lag_seasonal component.')

                    if len(lag_seasonal) < len(stochastic_lag_seasonal):
                        raise ValueError('The tuple which specifies the number of stochastic lag_seasonal components '
                                         'has greater length than the tuple that specifies the number of lag_seasonal '
                                         'components. Either pass a blank tuple () for the stochastic profile, or a '
                                         'boolean tuple of same length as the tuple that specifies the number of '
                                         'lag_seasonal components.')

                if len(damped_lag_seasonal) > 0:
                    if not all(isinstance(v, bool) for v in damped_lag_seasonal):
                        raise TypeError('If a non-empty tuple is passed for the damped specification '
                                        'of the lag_seasonal components, all elements must be of boolean type.')

                    if len(lag_seasonal) > len(damped_lag_seasonal):
                        raise ValueError('Some of the lag_seasonal components were given a damped specification, but '
                                         'not all. Partial specification of the damped profile is not allowed. '
                                         'Either leave the damped specification blank by passing an empty '
                                         'tuple (), which will default to True for all components, or pass a '
                                         'damped specification for each lag_seasonal component.')

                    if len(lag_seasonal) < len(damped_lag_seasonal):
                        raise ValueError('The tuple which specifies the number of damped lag_seasonal components '
                                         'has greater length than the tuple that specifies the number of lag_seasonal '
                                         'components. Either pass a blank tuple () for the damped profile, or a '
                                         'boolean tuple of same length as the tuple that specifies the number of '
                                         'lag_seasonal components.')
                    if len(stochastic_lag_seasonal) > 0:
                        for k in range(len(lag_seasonal)):
                            if damped_lag_seasonal[k] and not stochastic_lag_seasonal[k]:
                                raise ValueError('Every element in damped_lag_seasonal that is True must also '
                                                 'have a corresponding True in stochastic_lag_seasonal.')

                if resp.shape[0] < 4 * max(lag_seasonal):
                    warnings.warn(
                        f'It is recommended to have an observation count that is at least quadruple the highest '
                        f'periodicity specified. The max periodicity specified in lag_seasonal is {max(lag_seasonal)}.')

            else:
                if len(stochastic_lag_seasonal) > 0:
                    raise ValueError('No lag_seasonal components were specified, but a non-empty '
                                     'stochastic profile was passed for lag seasonality. If '
                                     'lag_seasonal components are desired, specify the period for each '
                                     'component via a tuple passed to the lag_seasonal argument.')
                if len(damped_lag_seasonal) > 0:
                    raise ValueError('No lag_seasonal components were specified, but a non-empty '
                                     'damped profile was passed for lag seasonality. If lag_seasonal '
                                     'components are desired, specify the period for each '
                                     'component via a tuple passed to the lag_seasonal argument.')

        # CHECK AND PREPARE DUMMY SEASONAL
        if not isinstance(dummy_seasonal, tuple):
            raise TypeError('dummy_seasonal must be a tuple.')

        if not isinstance(stochastic_dummy_seasonal, tuple):
            raise TypeError('stochastic_dummy_seasonal must be a tuple.')

        if len(dummy_seasonal) > 0:
            if not all(isinstance(v, int) for v in dummy_seasonal):
                raise TypeError('The period for a dummy seasonal component must be an integer.')

            if len(dummy_seasonal) != len(set(dummy_seasonal)):
                raise ValueError('Each specified period in dummy_seasonal must be distinct.')

            if any(v < 2 for v in dummy_seasonal):
                raise ValueError('The period for a dummy seasonal component must be an integer greater than 1.')

            if len(stochastic_dummy_seasonal) > 0:
                if not all(isinstance(v, bool) for v in stochastic_dummy_seasonal):
                    raise TypeError('If a non-empty tuple is passed for the stochastic specification '
                                    'of the dummy seasonal components, all elements must be of boolean type.')

                if len(dummy_seasonal) > len(stochastic_dummy_seasonal):
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

            if resp.shape[0] < 4 * max(dummy_seasonal):
                warnings.warn(f'It is recommended to have an observation count that is at least quadruple the highest '
                              f'periodicity specified. The max periodicity specified in dummy_seasonal is '
                              f'{max(dummy_seasonal)}.')

        else:
            if len(stochastic_dummy_seasonal) > 0:
                raise ValueError('No dummy seasonal components were specified, but a non-empty '
                                 'stochastic profile was passed for dummy seasonality. If dummy '
                                 'seasonal components are desired, specify the period for each '
                                 'component via a tuple passed to the dummy_seasonal argument.')

        # CHECK AND PREPARE TRIGONOMETRIC SEASONAL
        if not isinstance(trig_seasonal, tuple):
            raise TypeError('trig_seasonal must be a tuple.')

        if not isinstance(stochastic_trig_seasonal, tuple):
            raise TypeError('stochastic_trig_seasonal must be a tuple.')

        if len(trig_seasonal) > 0:
            if not all(isinstance(v, tuple) for v in trig_seasonal):
                raise TypeError('Each element in trig_seasonal must be a tuple.')

            if not all(len(v) == 2 for v in trig_seasonal):
                raise ValueError('A (period, num_harmonics) tuple must be provided for each specified trigonometric '
                                 'seasonal component.')

            if not all(isinstance(v[0], int) for v in trig_seasonal):
                raise TypeError('The period for a specified trigonometric seasonal component must be an integer.')

            if not all(isinstance(v[1], int) for v in trig_seasonal):
                raise TypeError('The number of harmonics for a specified trigonometric seasonal component must '
                                'be an integer.')

            if any(v[0] < 2 for v in trig_seasonal):
                raise ValueError('The period for a trigonometric seasonal component must be an integer greater than 1.')

            if any(v[1] < 1 and v[1] != 0 for v in trig_seasonal):
                raise ValueError('The number of harmonics for a trigonometric seasonal component can take 0 or '
                                 'integers at least as large as 1 as valid options. A value of 0 will enforce '
                                 'the highest possible number of harmonics for the given period, which is period / 2 '
                                 'if period is even, or (period - 1) / 2 if period is odd.')

            trig_periodicities = ()
            for v in trig_seasonal:
                period, num_harmonics = v
                trig_periodicities += (period,)
                if ao.is_odd(period):
                    if num_harmonics > int(period - 1) / 2:
                        raise ValueError('The number of harmonics for a trigonometric seasonal component cannot '
                                         'exceed (period - 1) / 2 when period is odd.')
                else:
                    if num_harmonics > int(period / 2):
                        raise ValueError('The number of harmonics for a trigonometric seasonal component cannot '
                                         'exceed period / 2 when period is even.')

            if len(trig_seasonal) != len(set(trig_periodicities)):
                raise ValueError('Each specified period in trig_seasonal must be distinct.')

            if len(stochastic_trig_seasonal) > 0:
                if not all(isinstance(v, bool) for v in stochastic_trig_seasonal):
                    raise TypeError('If a non-empty tuple is passed for the stochastic specification '
                                    'of the trigonometric seasonal components, all elements must be of boolean type.')

                if len(trig_seasonal) > len(stochastic_trig_seasonal):
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

            if resp.shape[0] < 4 * max(trig_periodicities):
                warnings.warn(f'It is recommended to have an observation count that is at least quadruple the highest '
                              f'periodicity specified. The max periodicity specified in trig_seasonal is '
                              f'{max(trig_periodicities)}.')

        else:
            if len(stochastic_trig_seasonal) > 0:
                raise ValueError('No trigonometric seasonal components were specified, but a non-empty '
                                 'stochastic profile was passed for trigonometric seasonality. If trigonometric '
                                 'seasonal components are desired, specify the period for each '
                                 'component via a tuple passed to the trig_seasonal argument.')

        # FINAL VALIDITY CHECKS
        if not isinstance(level, bool) or not isinstance(stochastic_level, bool):
            raise TypeError('level and stochastic_level must be of boolean type.')

        if level and not stochastic_level and damped_level:
            raise ValueError('stochastic_level must be true if level and damped_level are true.')

        if not isinstance(trend, bool) or not isinstance(stochastic_trend, bool):
            raise TypeError('trend and stochastic_trend must be of boolean type.')

        if trend and not stochastic_trend and damped_trend:
            raise ValueError('stochastic_trend must be true if trend and damped_trend are true.')

        if len(lag_seasonal) == 0 and len(dummy_seasonal) == 0 and len(trig_seasonal) == 0 and not level:
            raise ValueError('At least a level or seasonal component must be specified.')

        if trend and not level:
            raise ValueError('trend cannot be specified without a level component.')

        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError('seed must be an integer.')
            if not 0 < seed < 2 ** 32 - 1:
                raise ValueError('seed must be an integer between 0 and 2**32 - 1.')
            _set_seed(seed)  # for Numba JIT functions
            np.random.seed(seed)

        # Check if there are redundant periodicities across lag_seasonal, dummy_seasonal, and trig_seasonal
        if len(lag_seasonal + dummy_seasonal + trig_seasonal) > 0:
            trig_periodicities = ()
            if len(trig_seasonal) > 0:
                for v in trig_seasonal:
                    period, _ = v
                    trig_periodicities += (period,)

            if len(set(lag_seasonal).intersection(set(trig_periodicities))) > 0:
                raise ValueError('lag_seasonal and trig_seasonal cannot have periodicities in common.')

            if len(set(lag_seasonal).intersection(set(dummy_seasonal))) > 0:
                raise ValueError('lag_seasonal and dummy_seasonal cannot have periodicities in common.')

            if len(set(dummy_seasonal).intersection(set(trig_periodicities))) > 0:
                raise ValueError('dummy_seasonal and trig_seasonal cannot have periodicities in common.')

        # ASSIGN CLASS ATTRIBUTES IF ALL VALIDITY CHECKS ARE PASSED
        self.response = resp
        self.predictors = pred
        self.level = level
        self.stochastic_level = stochastic_level
        self.damped_level = damped_level
        self.trend = trend
        self.stochastic_trend = stochastic_trend
        self.damped_trend = damped_trend
        self.lag_seasonal = lag_seasonal
        self.stochastic_lag_seasonal = stochastic_lag_seasonal
        self.damped_lag_seasonal = damped_lag_seasonal
        self.dummy_seasonal = dummy_seasonal
        self.stochastic_dummy_seasonal = stochastic_dummy_seasonal
        self.trig_seasonal = trig_seasonal
        self.stochastic_trig_seasonal = stochastic_trig_seasonal

        # Set default values for stochastic_lag_seasonal and damped_lag_seasonal, if applicable
        if len(lag_seasonal) > 0:
            if len(stochastic_lag_seasonal) == 0:
                self.stochastic_lag_seasonal = (True,) * len(lag_seasonal)
            if len(damped_lag_seasonal) == 0:
                self.damped_lag_seasonal = (False,) * len(lag_seasonal)

        # Set default values for stochastic_dummy_seasonal, if applicable
        if len(dummy_seasonal) > 0 and len(stochastic_dummy_seasonal) == 0:
            self.stochastic_dummy_seasonal = (True,) * len(dummy_seasonal)

        # Set default values for stochastic_trig_seasonal and number of harmonics for
        # each periodicity, if applicable
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

        # Create historical time index if one hasn't already been created
        if self.historical_time_index is None:
            self.historical_time_index = np.arange(resp.shape[0])

        # Create variable names for predictors, if applicable
        if self.has_predictors:
            if self.predictors_names is None:
                self.predictors_names = [f"x{i + 1}" for i in range(self.num_predictors)]

        if resp.shape[0] <= self.num_state_eqs:
            warnings.warn('The number of state equations implied by the model specification '
                          'is at least as large as the number of observations in the response '
                          'array. Predictions from the model may be significantly compromised.')

    @property
    def num_lag_season_state_eqs(self) -> int:
        if len(self.lag_seasonal) == 0:
            return 0
        else:
            num_eqs = 0
            for v in self.lag_seasonal:
                num_eqs += v

        return num_eqs

    @property
    def num_stoch_lag_season_state_eqs(self) -> int:
        num_stoch = 0
        for c, v in enumerate(self.lag_seasonal):
            num_stoch += 1 * self.stochastic_lag_seasonal[c]

        return num_stoch

    @property
    def num_damped_lag_season(self) -> int:
        return sum(self.damped_lag_seasonal)

    @property
    def lag_season_damped_lags(self) -> tuple:
        if len(self.lag_seasonal) == 0:
            return ()
        else:
            lags = ()
            for c, v in enumerate(self.lag_seasonal):
                if self.damped_lag_seasonal[c]:
                    lags += (v,)
            return lags

    @property
    def num_dum_season_state_eqs(self) -> int:
        if len(self.dummy_seasonal) == 0:
            return 0
        else:
            num_eqs = 0
            for v in self.dummy_seasonal:
                num_eqs += v - 1

        return num_eqs

    @property
    def num_stoch_dum_season_state_eqs(self) -> int:
        num_stoch = 0
        for c, v in enumerate(self.dummy_seasonal):
            num_stoch += 1 * self.stochastic_dummy_seasonal[c]

        return num_stoch

    @property
    def num_trig_season_state_eqs(self) -> int:
        if len(self.trig_seasonal) == 0:
            return 0
        else:
            num_eqs = 0
            for v in self.trig_seasonal:
                period, num_harmonics = v
                if period / num_harmonics == 2:
                    num_eqs += 2 * num_harmonics - 1
                else:
                    num_eqs += 2 * num_harmonics

        return num_eqs

    @property
    def num_stoch_trig_season_state_eqs(self) -> int:
        num_stoch = 0
        for c, v in enumerate(self.trig_seasonal):
            period, num_harmonics = v
            if period / num_harmonics == 2:
                num_stoch += (2 * num_harmonics - 1) * self.stochastic_trig_seasonal[c]
            else:
                num_stoch += 2 * num_harmonics * self.stochastic_trig_seasonal[c]

        return num_stoch

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
        return ((self.level + self.trend) * 1
                + self.num_lag_season_state_eqs
                + self.num_dum_season_state_eqs
                + self.num_trig_season_state_eqs
                + self.has_predictors * 1)

    @property
    def num_stoch_states(self) -> int:
        return ((self.level * self.stochastic_level
                 + self.trend * self.stochastic_trend) * 1
                + self.num_stoch_lag_season_state_eqs
                + self.num_stoch_dum_season_state_eqs
                + self.num_stoch_trig_season_state_eqs)

    @property
    def num_obs(self) -> int:
        return self.response.shape[0]

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
        Z = np.zeros((num_rows, 1, m))

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

        if self.trend:
            j += 1

        if len(self.lag_seasonal) > 0:
            i = j
            for v in self.lag_seasonal:
                Z[:, :, i] = 1.
                i += v

            j += self.num_lag_season_state_eqs

        if len(self.dummy_seasonal) > 0:
            i = j
            for v in self.dummy_seasonal:
                Z[:, :, i] = 1.
                i += v - 1

            j += self.num_dum_season_state_eqs

        if len(self.trig_seasonal) > 0:
            Z[:, :, j::2] = 1.
            j += self.num_trig_season_state_eqs

        if self.has_predictors:
            Z[:, :, j] = 0.

        return Z

    @property
    def state_transition_matrix(self) -> np.ndarray:
        m = self.num_state_eqs
        T = np.zeros((m, m))

        i, j = 0, 0
        if self.level:
            T[i, j] = 1.

            # Get transition index for level if damping specified
            if self.damped_level:
                self.damped_level_transition_index = (i, j)

            i += 1
            j += 1

        if self.trend:
            T[i - 1, j] = 1.
            T[i, j] = 1.

            # Get transition index for trend if damping specified
            if self.damped_trend:
                self.damped_trend_transition_index = (i, j)

            i += 1
            j += 1

        if len(self.lag_seasonal) > 0:
            w = ()
            for c, v in enumerate(self.lag_seasonal):
                T[i, j + v - 1] = 1.

                # Get transition index for lag_seasonal component if damping specified
                if self.damped_lag_seasonal[c]:
                    w += ((i, j + v - 1),)

                for k in range(1, v):
                    T[i + k, j + k - 1] = 1.

                i += v
                j += v

            self.damped_lag_season_transition_index = w

        if len(self.dummy_seasonal) > 0:
            for v in self.dummy_seasonal:
                T[i, j:j + v - 1] = -1.
                for k in range(1, v - 1):
                    T[i + k, j + k - 1] = 1.

                i += v - 1
                j += v - 1

        if len(self.trig_seasonal) > 0:
            for v in self.trig_seasonal:
                period, num_harmonics = v
                if period / num_harmonics == 2:
                    for k in range(1, num_harmonics + 1):
                        if k < num_harmonics:
                            T[i:i + 2, j:j + 2] = self.trig_transition_matrix(2. * np.pi * k / period)
                            i += 2
                            j += 2
                        else:
                            T[i:i + 1, j:j + 1] = self.trig_transition_matrix(2. * np.pi * k / period)[0, 0]
                            i += 1
                            j += 1
                else:
                    for k in range(1, num_harmonics + 1):
                        T[i:i + 2, j:j + 2] = self.trig_transition_matrix(2. * np.pi * k / period)
                        i += 2
                        j += 2

        if self.has_predictors:
            T[i, j] = 1.

        return T

    @property
    def state_intercept_matrix(self) -> np.ndarray:
        m = self.num_state_eqs
        C = np.zeros((m, 1))

        return C

    @property
    def state_error_transformation_matrix(self) -> np.ndarray:
        m = self.num_state_eqs
        q = self.num_stoch_states
        R = np.zeros((m, q))

        if q == 0:
            pass
        else:
            i, j = 0, 0
            if self.level:
                if self.stochastic_level:
                    R[i, j] = 1.
                    j += 1

                i += 1

            if self.trend:
                if self.stochastic_trend:
                    R[i, j] = 1.
                    j += 1

                i += 1

            if len(self.lag_seasonal) > 0:
                for c, v in enumerate(self.lag_seasonal):
                    if self.stochastic_lag_seasonal[c]:
                        R[i, j] = 1.
                        j += 1

                    i += v

            if len(self.dummy_seasonal) > 0:
                for c, v in enumerate(self.dummy_seasonal):
                    if self.stochastic_dummy_seasonal[c]:
                        R[i, j] = 1.
                        j += 1

                    i += v - 1

            if len(self.trig_seasonal) > 0:
                for c, v in enumerate(self.trig_seasonal):
                    period, num_harmonics = v
                    if period / num_harmonics == 2:
                        num_eqs = 2 * num_harmonics - 1
                    else:
                        num_eqs = 2 * num_harmonics

                    if self.stochastic_trig_seasonal[c]:
                        for k in range(num_eqs):
                            R[i + k, j + k] = 1.

                        j += num_eqs
                    i += num_eqs

        return R

    @property
    def state_sse_transformation_matrix(self) -> np.ndarray:
        q = self.num_stoch_states
        q_trig = self.num_stoch_trig_season_state_eqs

        if q == 0:
            H = None
        else:
            if q_trig == 0:
                H = np.eye(q)
            else:
                q_restrict = q - q_trig + sum(self.stochastic_trig_seasonal)
                H = np.zeros((q_restrict, q))

                i = 0
                if self.level:
                    if self.stochastic_level:
                        H[i, i] = 1.
                        i += 1

                if self.trend:
                    if self.stochastic_trend:
                        H[i, i] = 1.
                        i += 1

                if len(self.lag_seasonal) > 0:
                    for c, v in enumerate(self.lag_seasonal):
                        if self.stochastic_lag_seasonal[c]:
                            H[i, i] = 1.
                            i += 1

                if len(self.dummy_seasonal) > 0:
                    for c, v in enumerate(self.dummy_seasonal):
                        if self.stochastic_dummy_seasonal[c]:
                            H[i, i] = 1.
                            i += 1

                if len(self.trig_seasonal) > 0:
                    j = i
                    for c, v in enumerate(self.trig_seasonal):
                        period, num_harmonics = v

                        if period / num_harmonics == 2:
                            num_eqs = 2 * num_harmonics - 1
                        else:
                            num_eqs = 2 * num_harmonics

                        if self.stochastic_trig_seasonal[c]:
                            H[i, j:j + num_eqs] = 1.
                            i += 1
                            j += num_eqs

        return H

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
            first_y = self._first_value(self.response)[0]
            return first_y
        else:
            return

    def _gibbs_iter0_init_trend(self):
        if self.trend:
            first_y = self._first_value(self.response)
            last_y = self._last_value(self.response)
            num_steps = (self.response.size - last_y[1][0][0] - 1) - first_y[1][0][0]
            if num_steps == 0:
                trend = 0.
            else:
                trend = (last_y[0] - first_y[0]) / num_steps
            return trend
        else:
            return

    def _gibbs_iter0_init_lag_season(self):
        if len(self.lag_seasonal) > 0:
            num_eqs = self.num_lag_season_state_eqs
            return np.zeros(num_eqs)
        else:
            return

    def _gibbs_iter0_init_dum_season(self):
        if len(self.dummy_seasonal) > 0:
            num_eqs = self.num_dum_season_state_eqs
            return np.zeros(num_eqs)
        else:
            return

    def _gibbs_iter0_init_trig_season(self):
        if len(self.trig_seasonal) > 0:
            num_eqs = self.num_trig_season_state_eqs
            return np.zeros(num_eqs)
        else:
            return

    def _model_setup(self,
                     response_var_shape_prior, response_var_scale_prior,
                     level_var_shape_prior, level_var_scale_prior,
                     damped_level_coeff_mean_prior, damped_level_coeff_prec_prior,
                     trend_var_shape_prior, trend_var_scale_prior,
                     damped_trend_coeff_mean_prior, damped_trend_coeff_prec_prior,
                     lag_season_var_shape_prior, lag_season_var_scale_prior,
                     damped_lag_season_coeff_mean_prior, damped_lag_season_coeff_prec_prior,
                     dum_season_var_shape_prior, dum_season_var_scale_prior,
                     trig_season_var_shape_prior, trig_season_var_scale_prior,
                     reg_coeff_mean_prior, reg_coeff_prec_prior,
                     zellner_prior_obs) -> ModelSetup:

        n = self.num_obs
        q = self.num_stoch_states

        # Initialize outputs
        if q > 0:
            state_var_scale_prior = []
            state_var_shape_post = []
            gibbs_iter0_state_error_var = []
        else:
            # Can't assign to NoneType because Numba function expects an array type.
            state_var_shape_post = np.array([[]])
            state_var_scale_prior = np.array([[]])
            gibbs_iter0_state_error_covariance = np.array([[]])

        gibbs_iter0_init_state = []
        init_state_variances = []
        init_state_plus_values = []
        seasonal_period = [0]

        if not self.damped_level:
            damped_level_coeff_mean_prior = None
            damped_level_coeff_prec_prior = None
            damped_level_coeff_cov_prior = None
            gibbs_iter0_damped_level_coeff = None

        if not self.damped_trend:
            damped_trend_coeff_mean_prior = None
            damped_trend_coeff_prec_prior = None
            damped_trend_coeff_cov_prior = None
            gibbs_iter0_damped_trend_coeff = None

        if not self.num_damped_lag_season > 0:
            damped_lag_season_coeff_mean_prior = None
            damped_lag_season_coeff_prec_prior = None
            damped_lag_season_coeff_cov_prior = None
            gibbs_iter0_damped_season_coeff = None

        if not self.has_predictors:
            reg_coeff_mean_prior = None
            reg_coeff_prec_prior = None
            reg_coeff_cov_prior = None
            reg_coeff_cov_post = None
            reg_coeff_prec_post = None
            gibbs_iter0_reg_coeff = None

        # Record the specification of the model.
        # Irregular component will always be a part of the model.
        components = dict()
        components['Irregular'] = dict()

        # RESPONSE ERROR
        if response_var_shape_prior is None:
            response_var_shape_prior = 1e-6
        if response_var_scale_prior is None:
            response_var_scale_prior = 1e-6

        response_var_shape_post = np.array([[response_var_shape_prior + 0.5 * n]])
        gibbs_iter0_response_error_variance = np.array([[response_var_scale_prior
                                                         / (response_var_shape_prior + 1.)]])

        # Record the state specification of the state space model

        # LEVEL STATE
        j, s = 0, 0  # j indexes the state equation, and s indexes stochastic equations
        if self.level:
            if self.stochastic_level:
                if level_var_shape_prior is None:
                    level_var_shape_prior = 1e-6

                if level_var_scale_prior is None:
                    level_var_scale_prior = 1e-6

                state_var_shape_post.append(level_var_shape_prior + 0.5 * n)
                state_var_scale_prior.append(level_var_scale_prior)
                gibbs_iter0_state_error_var.append(level_var_scale_prior
                                                   / (level_var_shape_prior + 1.))
                stochastic_index = s
                s += 1
            else:
                stochastic_index = None

            # The level state equation represents a random walk if
            # no damping is specified. Diffuse initialization is required
            # in this case. The fake states used in Durbin-Koopman for
            # simulating the smoothed states can take any arbitrary value
            # under diffuse initialization. Zero is chosen.
            # For the initial state covariance matrix, an
            # approximate diffuse initialization method is adopted.
            # That is, a diagonal matrix with large values along
            # the diagonal. Large in this setting is defined as 1e6.

            # If damping is specified, then the fake state is drawn
            # from a Gaussian distribution with mean zero and variance
            # equal to the variance of its stationary distribution, which
            # in general for an AR(1) process is
            # Residual Variance / (1 - [AR(1) Coefficient]^2).

            if self.damped_level:
                if damped_level_coeff_mean_prior is None:
                    damped_level_coeff_mean_prior = np.array([[0.]])

                if damped_level_coeff_prec_prior is None:
                    damped_level_coeff_prec_prior = np.array([[1.]])

                damped_level_coeff_cov_prior = ao.mat_inv(damped_level_coeff_prec_prior)
                gibbs_iter0_damped_level_coeff = damped_level_coeff_mean_prior
                init_state_variances.append(gibbs_iter0_state_error_var[stochastic_index] /
                                            (1 - gibbs_iter0_damped_level_coeff[0, 0] ** 2))
                init_state_plus_values.append(dist.vec_norm(0., np.sqrt(init_state_variances[j])))
                damped_index = self.damped_level_transition_index
            else:
                damped_index = None
                init_state_variances.append(1e6)
                init_state_plus_values.append(0.)

            gibbs_iter0_init_state.append(self._gibbs_iter0_init_level())
            components['Level'] = dict(start_state_eqn_index=j,
                                       end_state_eqn_index=j + 1,
                                       stochastic=self.stochastic_level,
                                       stochastic_index=stochastic_index,
                                       damped=self.damped_level,
                                       damped_transition_index=damped_index)
            j += 1

        # TREND STATE
        if self.trend:
            if self.stochastic_trend:
                if trend_var_shape_prior is None:
                    trend_var_shape_prior = 1e-6

                if trend_var_scale_prior is None:
                    trend_var_scale_prior = 1e-6

                state_var_shape_post.append(trend_var_shape_prior + 0.5 * n)
                state_var_scale_prior.append(trend_var_scale_prior)
                gibbs_iter0_state_error_var.append(trend_var_scale_prior
                                                   / (trend_var_shape_prior + 1.))

                stochastic_index = s
                s += 1
            else:
                stochastic_index = None

            # The trend state equation represents a random walk if
            # no damping is specified. Diffuse initialization is required
            # in this case. The fake states used in Durbin-Koopman for
            # simulating the smoothed states can take any arbitrary value
            # under diffuse initialization. Zero is chosen.
            # For the initial state covariance matrix, an
            # approximate diffuse initialization method is adopted.
            # That is, a diagonal matrix with large values along
            # the diagonal. Large in this setting is defined as 1e6.

            # If damping is specified, then the fake state is drawn
            # from a Gaussian distribution with mean zero and variance
            # equal to the variance of its stationary distribution, which
            # in general for an AR(1) process is
            # Residual Variance / (1 - [AR(1) Coefficient]^2).

            if self.damped_trend:
                if damped_trend_coeff_mean_prior is None:
                    damped_trend_coeff_mean_prior = np.array([[0.]])

                if damped_trend_coeff_prec_prior is None:
                    damped_trend_coeff_prec_prior = np.array([[1.]])

                damped_trend_coeff_cov_prior = ao.mat_inv(damped_trend_coeff_prec_prior)
                gibbs_iter0_damped_trend_coeff = damped_trend_coeff_mean_prior
                init_state_variances.append(gibbs_iter0_state_error_var[stochastic_index] /
                                            (1 - gibbs_iter0_damped_trend_coeff[0, 0] ** 2))
                init_state_plus_values.append(dist.vec_norm(0., np.sqrt(init_state_variances[j])))
                damped_index = self.damped_trend_transition_index
            else:
                damped_index = None
                init_state_variances.append(1e6)
                init_state_plus_values.append(0.)

            gibbs_iter0_init_state.append(self._gibbs_iter0_init_trend())
            components['Trend'] = dict(start_state_eqn_index=j,
                                       end_state_eqn_index=j + 1,
                                       stochastic=self.stochastic_trend,
                                       stochastic_index=stochastic_index,
                                       damped=self.damped_trend,
                                       damped_transition_index=damped_index)
            j += 1

        # LAG SEASONAL STATE
        if len(self.lag_seasonal) > 0:
            if self.num_damped_lag_season > 0:
                if damped_lag_season_coeff_mean_prior is None:
                    damped_lag_season_coeff_mean_prior = np.zeros((self.num_damped_lag_season, 1))
                if damped_lag_season_coeff_prec_prior is None:
                    damped_lag_season_coeff_prec_prior = np.ones((self.num_damped_lag_season, 1))

                damped_lag_season_coeff_cov_prior = damped_lag_season_coeff_prec_prior ** (-1)
                gibbs_iter0_damped_season_coeff = damped_lag_season_coeff_mean_prior

            i = j
            d = 0  # indexes damped periodicities
            for c, v in enumerate(self.lag_seasonal):
                seasonal_period.append(v)
                if self.stochastic_lag_seasonal[c]:
                    if lag_season_var_shape_prior is None:
                        shape_prior = 1e-6
                    else:
                        shape_prior = lag_season_var_shape_prior[c]
                    state_var_shape_post.append(shape_prior + 0.5 * n)

                    if lag_season_var_scale_prior is None:
                        scale_prior = 1e-6
                    else:
                        scale_prior = lag_season_var_scale_prior[c]
                    state_var_scale_prior.append(scale_prior)

                    gibbs_iter0_state_error_var.append(scale_prior / (shape_prior + 1.))

                    stochastic_index = s
                    s += 1
                else:
                    stochastic_index = None

                # A period-lag seasonal state equation represents a random walk if
                # no damping is specified. Diffuse initialization is required
                # in this case. The fake states used in Durbin-Koopman for
                # simulating the smoothed states can take any arbitrary value
                # under diffuse initialization. Zero is chosen.
                # For the initial state covariance matrix, an
                # approximate diffuse initialization method is adopted.
                # That is, a diagonal matrix with large values along
                # the diagonal. Large in this setting is defined as 1e6.

                # If damping is specified, then the fake state is drawn
                # from a Gaussian distribution with mean zero and variance
                # equal to the variance of its stationary distribution, which
                # in general for an AR(1) process is
                # Residual Variance / (1 - [AR(1) Coefficient]^2).

                if self.damped_lag_seasonal[c]:
                    init_state_variances.append(gibbs_iter0_state_error_var[stochastic_index] /
                                                (1 - gibbs_iter0_damped_season_coeff[d, 0] ** 2))
                    init_state_plus_values.append(dist.vec_norm(0., np.sqrt(init_state_variances[j])))
                    damped_index = d
                    d += 1
                else:
                    damped_index = None
                    init_state_variances.append(1e6)
                    init_state_plus_values.append(0.)

                for k in range(v - 1):
                    init_state_variances.append(1e6)
                    init_state_plus_values.append(0.)

                components[f'Lag-Seasonal.{v}'] = dict(start_state_eqn_index=i,
                                                       end_state_eqn_index=i + v,
                                                       stochastic=self.stochastic_lag_seasonal[c],
                                                       stochastic_index=stochastic_index,
                                                       damped=self.damped_lag_seasonal[c],
                                                       damped_transition_index=damped_index)
                i += v

            for k in self._gibbs_iter0_init_lag_season():
                gibbs_iter0_init_state.append(k)

            j += self.num_lag_season_state_eqs

        # DUMMY SEASONAL STATE
        if len(self.dummy_seasonal) > 0:
            i = j
            for c, v in enumerate(self.dummy_seasonal):
                seasonal_period.append(v)
                if self.stochastic_dummy_seasonal[c]:
                    if dum_season_var_shape_prior is None:
                        shape_prior = 1e-6
                    else:
                        shape_prior = dum_season_var_shape_prior[c]
                    state_var_shape_post.append(shape_prior + 0.5 * n)

                    if dum_season_var_scale_prior is None:
                        scale_prior = 1e-6
                    else:
                        scale_prior = dum_season_var_scale_prior[c]
                    state_var_scale_prior.append(scale_prior)

                    gibbs_iter0_state_error_var.append(scale_prior / (shape_prior + 1.))

                    stochastic_index = s
                    s += 1
                else:
                    stochastic_index = None

                # A dummy seasonal state equation represents a random walk.
                # Diffuse initialization is required in this case. The fake states
                # used in Durbin-Koopman for simulating the smoothed states can take
                # any arbitrary value under diffuse initialization. Zero is chosen.
                # For the initial state covariance matrix, an
                # approximate diffuse initialization method is adopted.
                # That is, a diagonal matrix with large values along
                # the diagonal. Large in this setting is defined as 1e6.

                for k in range(v - 1):
                    init_state_plus_values.append(0.)
                    init_state_variances.append(1e6)

                components[f'Dummy-Seasonal.{v}'] = dict(start_state_eqn_index=i,
                                                         end_state_eqn_index=i + (v - 1),
                                                         stochastic=self.stochastic_dummy_seasonal[c],
                                                         stochastic_index=stochastic_index,
                                                         damped=None,
                                                         damped_transition_index=None)
                i += v - 1

            for k in self._gibbs_iter0_init_dum_season():
                gibbs_iter0_init_state.append(k)

            j += self.num_dum_season_state_eqs

        # TRIGONOMETRIC SEASONAL STATE
        if len(self.trig_seasonal) > 0:
            i = j
            for c, v in enumerate(self.trig_seasonal):
                period, num_harmonics = v

                if period / num_harmonics == 2:
                    num_eqs = 2 * num_harmonics - 1
                else:
                    num_eqs = 2 * num_harmonics

                seasonal_period.append(period)
                if self.stochastic_trig_seasonal[c]:
                    if trig_season_var_shape_prior is None:
                        shape_prior = 1e-6
                    else:
                        shape_prior = trig_season_var_shape_prior[c]
                    state_var_shape_post.append(shape_prior + 0.5 * n * num_eqs)

                    if trig_season_var_scale_prior is None:
                        scale_prior = 1e-6
                    else:
                        scale_prior = trig_season_var_scale_prior[c]
                    state_var_scale_prior.append(scale_prior)

                    for k in range(num_eqs):
                        gibbs_iter0_state_error_var.append(scale_prior / (shape_prior + 1.))

                    stochastic_index = s
                    s += num_eqs
                else:
                    stochastic_index = None

                # A trigonometric seasonal state equation represents a random walk.
                # Diffuse initialization is required in this case. The fake states
                # used in Durbin-Koopman for simulating the smoothed states can take
                # any arbitrary value under diffuse initialization. Zero is chosen.
                # For the initial state covariance matrix, an
                # approximate diffuse initialization method is adopted.
                # That is, a diagonal matrix with large values along
                # the diagonal. Large in this setting is defined as 1e6.

                for k in range(num_eqs):
                    init_state_plus_values.append(0.)
                    init_state_variances.append(1e6)

                components[f'Trigonometric-Seasonal'
                           f'.{period}.{num_harmonics}'] = dict(start_state_eqn_index=i,
                                                                end_state_eqn_index=i + num_eqs,
                                                                stochastic=self.stochastic_trig_seasonal[c],
                                                                stochastic_index=stochastic_index,
                                                                damped=None,
                                                                damped_transition_index=None)
                i += num_eqs

            for k in self._gibbs_iter0_init_trig_season():
                gibbs_iter0_init_state.append(k)

            j += self.num_trig_season_state_eqs

        # REGRESSION
        if self.has_predictors:
            components['Regression'] = dict(start_state_eqn_index=j,
                                            end_state_eqn_index=j + 1,
                                            stochastic=False,
                                            stochastic_index=None,
                                            damped=None,
                                            damped_transition_index=None)
            X = self.predictors

            # Static regression coefficients are modeled
            # by appending X*beta to the observation matrix,
            # appending a 1 to the state vector, and forcing
            # the relevant state equation to be non-stochastic.
            # Thus, the variance for the static regression
            # state equation is zero and the initial state
            # value is 1.
            init_state_plus_values.append(0.)
            init_state_variances.append(0.)
            gibbs_iter0_init_state.append(1.)

            if zellner_prior_obs is None:
                zellner_prior_obs = 1e-6

            if reg_coeff_mean_prior is None:
                reg_coeff_mean_prior = np.zeros((self.num_predictors, 1))

            if reg_coeff_prec_prior is None:
                reg_coeff_prec_prior = zellner_prior_obs / n * (0.5 * dot(X.T, X)
                                                                + 0.5 * np.diag(np.diag(dot(X.T, X))))
                reg_coeff_cov_prior = ao.mat_inv(reg_coeff_prec_prior)
            else:
                reg_coeff_cov_prior = ao.mat_inv(reg_coeff_prec_prior)

            reg_coeff_prec_post = dot(X.T, X) + reg_coeff_prec_prior
            reg_coeff_cov_post = ao.mat_inv(reg_coeff_prec_post)
            gibbs_iter0_reg_coeff = reg_coeff_mean_prior

        if q > 0:
            state_var_shape_post = np.vstack(state_var_shape_post)
            state_var_scale_prior = np.vstack(state_var_scale_prior)
            gibbs_iter0_state_error_covariance = np.diag(gibbs_iter0_state_error_var)

        gibbs_iter0_init_state = np.vstack(gibbs_iter0_init_state)
        init_state_plus_values = np.vstack(init_state_plus_values)
        init_state_covariance = np.diag(init_state_variances)
        self.num_first_obs_ignore = max(1 + max(seasonal_period), self.num_state_eqs)

        self.model_setup = ModelSetup(components,
                                      response_var_scale_prior,
                                      response_var_shape_post,
                                      state_var_scale_prior,
                                      state_var_shape_post,
                                      gibbs_iter0_init_state,
                                      gibbs_iter0_response_error_variance,
                                      gibbs_iter0_state_error_covariance,
                                      init_state_plus_values,
                                      init_state_covariance,
                                      damped_level_coeff_mean_prior,
                                      damped_level_coeff_prec_prior,
                                      damped_level_coeff_cov_prior,
                                      gibbs_iter0_damped_level_coeff,
                                      damped_trend_coeff_mean_prior,
                                      damped_trend_coeff_prec_prior,
                                      damped_trend_coeff_cov_prior,
                                      gibbs_iter0_damped_trend_coeff,
                                      damped_lag_season_coeff_mean_prior,
                                      damped_lag_season_coeff_prec_prior,
                                      damped_lag_season_coeff_cov_prior,
                                      gibbs_iter0_damped_season_coeff,
                                      reg_coeff_mean_prior,
                                      reg_coeff_prec_prior,
                                      reg_coeff_cov_prior,
                                      reg_coeff_cov_post,
                                      reg_coeff_prec_post,
                                      zellner_prior_obs,
                                      gibbs_iter0_reg_coeff)

        return self.model_setup

    def sample(self,
               num_samp: int,
               response_var_shape_prior: float = None,
               response_var_scale_prior: float = None,
               level_var_shape_prior: float = None,
               level_var_scale_prior: float = None,
               damped_level_coeff_mean_prior: np.ndarray = None,
               damped_level_coeff_prec_prior: np.ndarray = None,
               trend_var_shape_prior: float = None,
               trend_var_scale_prior: float = None,
               damped_trend_coeff_mean_prior: np.ndarray = None,
               damped_trend_coeff_prec_prior: np.ndarray = None,
               lag_season_var_shape_prior: tuple = None,
               lag_season_var_scale_prior: tuple = None,
               damped_lag_season_coeff_mean_prior: np.ndarray = None,
               damped_lag_season_coeff_prec_prior: np.ndarray = None,
               dum_season_var_shape_prior: tuple = None,
               dum_season_var_scale_prior: tuple = None,
               trig_season_var_shape_prior: tuple = None,
               trig_season_var_scale_prior: tuple = None,
               reg_coeff_mean_prior: np.ndarray = None,
               reg_coeff_prec_prior: np.ndarray = None,
               zellner_prior_obs: float = None) -> Posterior:

        """

        :param num_samp: integer > 0. Specifies the number of posterior samples to draw.

        :param response_var_shape_prior: float > 0. Specifies the inverse-Gamma shape prior for the
        response error variance. Default is 1e-6.

        :param response_var_scale_prior: float > 0. Specifies the inverse-Gamma scale prior for the
        response error variance. Default is 1e-6.

        :param level_var_shape_prior: float > 0. Specifies the inverse-Gamma shape prior for the
        level state equation error variance. Default is 1e-6.

        :param level_var_scale_prior: float > 0. Specifies the inverse-Gamma scale prior for the
        level state equation error variance. Default is 1e-6.

        :param damped_level_coeff_mean_prior: Numpy array of dimension (1, 1). Specifies the prior
        mean for the coefficient governing the level's AR(1) process without drift. Default is [[0.]].

        :param damped_level_coeff_prec_prior: Numpy array of dimension (1, 1). Specifies the prior
        precision matrix for the coefficient governing the level's an AR(1) process without drift.
        Default is [[1.]].

        :param trend_var_shape_prior: float > 0. Specifies the inverse-Gamma shape prior for the
        trend state equation error variance. Default is 1e-6.

        :param trend_var_scale_prior: float > 0. Specifies the inverse-Gamma scale prior for the
        trend state equation error variance. Default is 1e-6.

        :param damped_trend_coeff_mean_prior: Numpy array of dimension (1, 1). Specifies the prior
        mean for the coefficient governing the trend's AR(1) process without drift. Default is [[0.]].

        :param damped_trend_coeff_prec_prior: Numpy array of dimension (1, 1). Specifies the prior
        precision matrix for the coefficient governing the trend's an AR(1) process without drift.
        Default is [[1.]].

        :param lag_season_var_shape_prior: tuple of floats > 0. Specifies the inverse-Gamma shape priors
        for each periodicity in lag_seasonal. Default is 1e-6 for each periodicity.

        :param lag_season_var_scale_prior: tuple of floats > 0. Specifies the inverse-Gamma scale priors
        for each periodicity in lag_seasonal. Default is 1e-6 for each periodicity.

        :param damped_lag_season_coeff_mean_prior: Numpy array of dimension (1, 1). Specifies the prior
        mean for the coefficient governing a lag_seasonal AR(1) process without drift. Default is [[0.]].

        :param damped_lag_season_coeff_prec_prior: Numpy array of dimension (1, 1). Specifies the prior
        precision matrix for the coefficient governing a lag_seasonal AR(1) process without drift.
        Default is [[1.]].

        :param dum_season_var_shape_prior: tuple of floats > 0. Specifies the inverse-Gamma shape priors
        for each periodicity in dummy_seasonal. Default is 1e-6 for each periodicity.

        :param dum_season_var_scale_prior: tuple of floats > 0. Specifies the inverse-Gamma scale priors
        for each periodicity in dummy_seasonal. Default is 1e-6 for each periodicity.

        :param trig_season_var_shape_prior: tuple of floats > 0. Specifies the inverse-Gamma shape priors
        for each periodicity in trig_seasonal. For example, if trig_seasonal = ((12, 3), (10, 2)) and
        stochastic_trig_seasonal = (True, True), only two shape priors need to be specified: one for periodicity 12
        and one for periodicity 10. Default is 1e-6 for each periodicity.

        :param trig_season_var_scale_prior: tuple of floats > 0. Specifies the inverse-Gamma scale priors
        for each periodicity in trig_seasonal. For example, if trig_seasonal = ((12, 3), (10, 2)) and
        stochastic_trig_seasonal = (True, True), only two scale priors need to be specified: one for periodicity 12
        and one for periodicity 10. Default is 1e-6 for each periodicity.

        :param reg_coeff_mean_prior: Numpy array of dimension (k, 1), where k is the number of predictors.
        Data type must be float64. If predictors are specified without a mean prior, a k-dimensional zero
        vector will be assumed.

        :param reg_coeff_prec_prior: Numpy array of dimension (k, k), where k is the number of predictors.
        Data type must be float64. If predictors are specified without a precision prior, Zellner's g-prior will
        be enforced. Specifically, 1 / g * (w * dot(X.T, X) + (1 - w) * diag(dot(X.T, X))), where g = n / prior_obs,
        prior_obs is the number of prior observations given to the regression coefficient mean prior (i.e.,
        it controls how much weight is given to the mean prior), n is the number of observations, X is the design
        matrix, and diag(dot(X.T, X)) is a diagonal matrix with the diagonal elements matching those of
        dot(X.T, X). The addition of the diagonal matrix to dot(X.T, X) is to guard against singularity
        (i.e., a design matrix that is not full rank). The weighting controlled by w is set to 0.5.

        :param zellner_prior_obs: float > 0. Relevant only if no regression precision matrix is provided.
        It controls how precise one believes their priors are for the regression coefficients, assuming no regression
        precision matrix is provided. Default value is 1e-6, which gives little weight to the regression coefficient
        mean prior. This should approximate maximum likelihood estimation.

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
                    damped_level_coefficient: Posterior AR(1) coefficient for level
                    damped_trend_coefficient: Posterior AR(1) coefficient for trend
                    damped_lag_season_coefficients: Posterior AR(1) coefficients for lag_seasonal components
        """

        if not isinstance(num_samp, int):
            raise TypeError('num_samp must be of type integer.')
        if not num_samp > 0:
            raise ValueError('num_samp must a strictly positive integer.')

        # Response prior check
        if response_var_shape_prior is not None:
            if not isinstance(response_var_shape_prior, float):
                raise TypeError('response_var_shape_prior must be of type float.')
            if np.isnan(response_var_shape_prior):
                raise ValueError('response_var_shape_prior cannot be NaN.')
            if np.isinf(response_var_shape_prior):
                raise ValueError('response_var_shape_prior cannot be Inf/-Inf.')
            if not response_var_shape_prior > 0:
                raise ValueError('response_var_shape_prior must be a strictly positive float.')

        if response_var_scale_prior is not None:
            if not isinstance(response_var_scale_prior, float):
                raise TypeError('response_var_scale_prior must be of type float.')
            if np.isnan(response_var_scale_prior):
                raise ValueError('response_var_scale_prior cannot be NaN.')
            if np.isinf(response_var_scale_prior):
                raise ValueError('response_var_scale_prior cannot be Inf/-Inf.')
            if not response_var_scale_prior > 0:
                raise ValueError('response_var_scale_prior must be a strictly positive float.')

        # Level prior check
        if self.level and self.stochastic_level:
            if level_var_shape_prior is not None:
                if not isinstance(level_var_shape_prior, float):
                    raise TypeError('level_var_shape_prior must be of type float.')
                if np.isnan(level_var_shape_prior):
                    raise ValueError('level_var_shape_prior cannot be NaN.')
                if np.isinf(level_var_shape_prior):
                    raise ValueError('level_var_shape_prior cannot be Inf/-Inf.')
                if not level_var_shape_prior > 0:
                    raise ValueError('level_var_shape_prior must be a strictly positive float.')

            if level_var_scale_prior is not None:
                if not isinstance(level_var_scale_prior, float):
                    raise TypeError('level_var_scale_prior must be of type float.')
                if np.isnan(level_var_scale_prior):
                    raise ValueError('level_var_scale_prior cannot be NaN.')
                if np.isinf(level_var_scale_prior):
                    raise ValueError('level_var_scale_prior cannot be Inf/-Inf.')
                if not level_var_scale_prior > 0:
                    raise ValueError('level_var_scale_prior must be a strictly positive float.')

            # Damped level prior check
            if self.damped_level:
                if damped_level_coeff_mean_prior is not None:
                    if not isinstance(damped_level_coeff_mean_prior, np.ndarray):
                        raise TypeError('damped_level_coeff_mean_prior must be a Numpy array.')
                    if damped_level_coeff_mean_prior.dtype != 'float64':
                        raise TypeError('All values in damped_level_coeff_mean_prior must be of type float.')
                    if not damped_level_coeff_mean_prior.shape == (1, 1):
                        raise ValueError('damped_level_coeff_mean_prior must have shape (1, 1).')
                    if np.any(np.isnan(damped_level_coeff_mean_prior)):
                        raise ValueError('damped_level_coeff_mean_prior cannot have NaN values.')
                    if np.any(np.isinf(damped_level_coeff_mean_prior)):
                        raise ValueError('damped_level_coeff_mean_prior cannot have Inf/-Inf values.')
                    if not abs(damped_level_coeff_mean_prior[0, 0]) < 1:
                        warnings.warn('The mean damped level coefficient is greater than 1 in absolute value, '
                                      'which implies an explosive process. Note that an explosive process '
                                      'can be stationary, but it implies that the future is needed to '
                                      'predict the past.')

                if damped_level_coeff_prec_prior is not None:
                    if not isinstance(damped_level_coeff_prec_prior, np.ndarray):
                        raise TypeError('damped_level_coeff_prec_prior must be a Numpy array.')
                    if damped_level_coeff_prec_prior.dtype != 'float64':
                        raise TypeError('All values in damped_level_coeff_prec_prior must be of type float.')
                    if not damped_level_coeff_prec_prior.shape == (1, 1):
                        raise ValueError('damped_level_coeff_prec_prior must have shape (1, 1).')
                    if np.any(np.isnan(damped_level_coeff_prec_prior)):
                        raise ValueError('damped_level_coeff_prec_prior cannot have NaN values.')
                    if np.any(np.isinf(damped_level_coeff_prec_prior)):
                        raise ValueError('damped_level_coeff_prec_prior cannot have Inf/-Inf values.')
                    if not ao.is_positive_definite(damped_level_coeff_prec_prior):
                        raise ValueError('damped_level_coeff_prec_prior must be a positive definite matrix.')
                    if not ao.is_symmetric(damped_level_coeff_prec_prior):
                        raise ValueError('damped_level_coeff_prec_prior must be a symmetric matrix.')

        # Trend prior check
        if self.trend and self.stochastic_trend:
            if trend_var_shape_prior is not None:
                if not isinstance(trend_var_shape_prior, float):
                    raise TypeError('trend_var_shape_prior must be of type float.')
                if np.isnan(trend_var_shape_prior):
                    raise ValueError('trend_var_shape_prior cannot be NaN.')
                if np.isinf(trend_var_shape_prior):
                    raise ValueError('trend_var_shape_prior cannot be Inf/-Inf.')
                if not trend_var_shape_prior > 0:
                    raise ValueError('trend_var_shape_prior must be a strictly positive float.')

            if trend_var_scale_prior is not None:
                if not isinstance(trend_var_scale_prior, float):
                    raise TypeError('trend_var_scale_prior must be of type float.')
                if np.isnan(trend_var_scale_prior):
                    raise ValueError('trend_var_scale_prior cannot be NaN.')
                if np.isinf(trend_var_scale_prior):
                    raise ValueError('trend_var_scale_prior cannot be Inf/-Inf.')
                if not trend_var_scale_prior > 0:
                    raise ValueError('trend_var_scale_prior must be a strictly positive float.')

            # Damped trend prior check
            if self.damped_trend:
                if damped_trend_coeff_mean_prior is not None:
                    if not isinstance(damped_trend_coeff_mean_prior, np.ndarray):
                        raise TypeError('damped_trend_coeff_mean_prior must be a Numpy array.')
                    if damped_trend_coeff_mean_prior.dtype != 'float64':
                        raise TypeError('All values in damped_trend_coeff_mean_prior must be of type float.')
                    if not damped_trend_coeff_mean_prior.shape == (1, 1):
                        raise ValueError('damped_trend_coeff_mean_prior must have shape (1, 1).')
                    if np.any(np.isnan(damped_trend_coeff_mean_prior)):
                        raise ValueError('damped_trend_coeff_mean_prior cannot have NaN values.')
                    if np.any(np.isinf(damped_trend_coeff_mean_prior)):
                        raise ValueError('damped_trend_coeff_mean_prior cannot have Inf/-Inf values.')
                    if not abs(damped_trend_coeff_mean_prior[0, 0]) < 1:
                        warnings.warn('The mean damped trend coefficient is greater than 1 in absolute value, '
                                      'which implies an explosive process. Note that an explosive process '
                                      'can be stationary, but it implies that the future is needed to '
                                      'predict the past.')

                if damped_trend_coeff_prec_prior is not None:
                    if not isinstance(damped_trend_coeff_prec_prior, np.ndarray):
                        raise TypeError('damped_trend_coeff_prec_prior must be a Numpy array.')
                    if damped_trend_coeff_prec_prior.dtype != 'float64':
                        raise TypeError('All values in damped_trend_coeff_prec_prior must be of type float.')
                    if not damped_trend_coeff_prec_prior.shape == (1, 1):
                        raise ValueError('damped_trend_coeff_prec_prior must have shape (1, 1).')
                    if np.any(np.isnan(damped_trend_coeff_prec_prior)):
                        raise ValueError('damped_trend_coeff_prec_prior cannot have NaN values.')
                    if np.any(np.isinf(damped_trend_coeff_prec_prior)):
                        raise ValueError('damped_trend_coeff_prec_prior cannot have Inf/-Inf values.')
                    if not ao.is_positive_definite(damped_trend_coeff_prec_prior):
                        raise ValueError('damped_trend_coeff_prec_prior must be a positive definite matrix.')
                    if not ao.is_symmetric(damped_trend_coeff_prec_prior):
                        raise ValueError('damped_trend_coeff_prec_prior must be a symmetric matrix.')

        # Lag seasonal prior check
        if len(self.lag_seasonal) > 0 and sum(self.stochastic_lag_seasonal) > 0:
            if lag_season_var_shape_prior is not None:
                if not isinstance(lag_season_var_shape_prior, tuple):
                    raise TypeError('lag_season_var_shape_prior must be a tuple '
                                    'to accommodate potentially multiple seasonality.')
                if len(lag_season_var_shape_prior) != sum(self.stochastic_lag_seasonal):
                    raise ValueError('The number of elements in lag_season_var_shape_prior must match the '
                                     'number of stochastic periodicities in lag_seasonal. That is, for each '
                                     'stochastic periodicity in lag_seasonal, there must be a corresponding '
                                     'shape prior.')
                if not all(isinstance(i, float) for i in lag_season_var_shape_prior):
                    raise TypeError('All values in lag_season_var_shape_prior must be of type float.')
                if any(np.isnan(i) for i in lag_season_var_shape_prior):
                    raise ValueError('No values in lag_season_var_shape_prior can be NaN.')
                if any(np.isinf(i) for i in lag_season_var_shape_prior):
                    raise ValueError('No values in lag_season_var_shape_prior can be Inf/-Inf.')
                if not all(i > 0 for i in lag_season_var_shape_prior):
                    raise ValueError('All values in lag_season_var_shape_prior must be strictly positive floats.')

            if lag_season_var_scale_prior is not None:
                if not isinstance(lag_season_var_scale_prior, tuple):
                    raise TypeError('lag_season_var_scale_prior must be a tuple '
                                    'to accommodate potentially multiple seasonality.')
                if len(lag_season_var_scale_prior) != sum(self.stochastic_lag_seasonal):
                    raise ValueError('The number of elements in lag_season_var_scale_prior must match the '
                                     'number of stochastic periodicities in lag_seasonal. That is, for each '
                                     'stochastic periodicity in lag_seasonal, there must be a corresponding '
                                     'scale prior.')
                if not all(isinstance(i, float) for i in lag_season_var_scale_prior):
                    raise TypeError('All values in lag_season_var_scale_prior must be of type float.')
                if any(np.isnan(i) for i in lag_season_var_scale_prior):
                    raise ValueError('No values in lag_season_var_scale_prior can be NaN.')
                if any(np.isinf(i) for i in lag_season_var_scale_prior):
                    raise ValueError('No values in lag_season_var_scale_prior can be Inf/-Inf.')
                if not all(i > 0 for i in lag_season_var_scale_prior):
                    raise ValueError('All values in lag_season_var_scale_prior must be strictly positive floats.')

            # Damped trend prior check
            if self.num_damped_lag_season > 0:
                if damped_lag_season_coeff_mean_prior is not None:
                    if not isinstance(damped_lag_season_coeff_mean_prior, np.ndarray):
                        raise TypeError('damped_lag_season_coeff_mean_prior must be a Numpy array.')
                    if damped_lag_season_coeff_mean_prior.dtype != 'float64':
                        raise TypeError('All values in damped_lag_season_coeff_mean_prior must be of type float.')
                    if not damped_lag_season_coeff_mean_prior.shape == (self.num_damped_lag_season, 1):
                        raise ValueError(f'damped_lag_season_coeff_mean_prior must have shape '
                                         f'({self.num_damped_lag_season}, 1), where the row count '
                                         f'corresponds to the number of lags with damping.')
                    if np.any(np.isnan(damped_lag_season_coeff_mean_prior)):
                        raise ValueError('damped_trend_coeff_mean_prior cannot have NaN values.')
                    if np.any(np.isinf(damped_lag_season_coeff_mean_prior)):
                        raise ValueError('damped_lag_season_coeff_mean_prior cannot have Inf/-Inf values.')
                    for j in range(self.num_damped_lag_season):
                        if not abs(damped_lag_season_coeff_mean_prior[j, 0]) < 1:
                            warnings.warn(f'The mean damped coefficient for seasonal lag {j} is greater than 1 in '
                                          f'absolute value, which implies an explosive process. Note that an '
                                          f'explosive process can be stationary, but it implies that the future '
                                          f'is needed to predict the past.')

                if damped_lag_season_coeff_prec_prior is not None:
                    if not isinstance(damped_lag_season_coeff_prec_prior, np.ndarray):
                        raise TypeError('damped_lag_season_coeff_prec_prior must be a Numpy array.')
                    if damped_lag_season_coeff_prec_prior.dtype != 'float64':
                        raise TypeError('All values in damped_lag_season_coeff_prec_prior must be of type float.')
                    if not damped_lag_season_coeff_prec_prior.shape == (self.num_damped_lag_season, 1):
                        raise ValueError(f'damped_lag_season_coeff_prec_prior must have shape '
                                         f'({self.num_damped_lag_season}, 1), where the row count '
                                         f'corresponds to the number of lags with damping.')
                    if np.any(np.isnan(damped_lag_season_coeff_prec_prior)):
                        raise ValueError('damped_lag_season_coeff_prec_prior cannot have NaN values.')
                    if np.any(np.isinf(damped_lag_season_coeff_prec_prior)):
                        raise ValueError('damped_lag_season_coeff_prec_prior cannot have Inf/-Inf values.')
                    if not ao.is_positive_definite(damped_lag_season_coeff_prec_prior):
                        raise ValueError('damped_lag_season_coeff_prec_prior must be a positive definite matrix.')
                    if not ao.is_symmetric(damped_lag_season_coeff_prec_prior):
                        raise ValueError('damped_lag_season_coeff_prec_prior must be a symmetric matrix.')

        # Dummy seasonal prior check
        if len(self.dummy_seasonal) > 0 and sum(self.stochastic_dummy_seasonal) > 0:
            if dum_season_var_shape_prior is not None:
                if not isinstance(dum_season_var_shape_prior, tuple):
                    raise TypeError('dum_seasonal_var_shape_prior must be a tuple '
                                    'to accommodate potentially multiple seasonality.')
                if len(dum_season_var_shape_prior) != sum(self.stochastic_dummy_seasonal):
                    raise ValueError('The number of elements in dum_season_var_shape_prior must match the '
                                     'number of stochastic periodicities in dummy_seasonal. That is, for each '
                                     'stochastic periodicity in dummy_seasonal, there must be a corresponding '
                                     'shape prior.')
                if not all(isinstance(i, float) for i in dum_season_var_shape_prior):
                    raise TypeError('All values in dum_season_var_shape_prior must be of type float.')
                if any(np.isnan(i) for i in dum_season_var_shape_prior):
                    raise ValueError('No values in dum_season_var_shape_prior can be NaN.')
                if any(np.isinf(i) for i in dum_season_var_shape_prior):
                    raise ValueError('No values in dum_season_var_shape_prior can be Inf/-Inf.')
                if not all(i > 0 for i in dum_season_var_shape_prior):
                    raise ValueError('All values in dum_season_var_shape_prior must be strictly positive floats.')

            if dum_season_var_scale_prior is not None:
                if not isinstance(dum_season_var_scale_prior, tuple):
                    raise TypeError('dum_seasonal_var_scale_prior must be a tuple '
                                    'to accommodate potentially multiple seasonality.')
                if len(dum_season_var_shape_prior) != sum(self.stochastic_dummy_seasonal):
                    raise ValueError('The number of elements in dum_season_var_scale_prior must match the '
                                     'number of stochastic periodicities in dummy_seasonal. That is, for each '
                                     'stochastic periodicity in dummy_seasonal, there must be a corresponding '
                                     'scale prior.')
                if not all(isinstance(i, float) for i in dum_season_var_scale_prior):
                    raise TypeError('All values in dum_season_var_scale_prior must be of type float.')
                if any(np.isnan(i) for i in dum_season_var_scale_prior):
                    raise ValueError('No values in dum_season_var_scale_prior can be NaN.')
                if any(np.isinf(i) for i in dum_season_var_scale_prior):
                    raise ValueError('No values in dum_season_var_scale_prior can be Inf/-Inf.')
                if not all(i > 0 for i in dum_season_var_scale_prior):
                    raise ValueError('All values in dum_season_var_scale_prior must be strictly positive floats.')

        # Trigonometric seasonal prior check
        if len(self.trig_seasonal) > 0 and sum(self.stochastic_trig_seasonal) > 0:
            if trig_season_var_shape_prior is not None:
                if not isinstance(trig_season_var_shape_prior, tuple):
                    raise TypeError('trig_seasonal_var_shape_prior must be a tuple '
                                    'to accommodate potentially multiple seasonality.')
                if len(trig_season_var_shape_prior) != sum(self.stochastic_trig_seasonal):
                    raise ValueError('The number of elements in trig_season_var_shape_prior must match the '
                                     'number of stochastic periodicities in trig_seasonal. That is, for each '
                                     'stochastic periodicity in trig_seasonal, there must be a corresponding '
                                     'shape prior.')
                if not all(isinstance(i, float) for i in trig_season_var_shape_prior):
                    raise TypeError('All values in trig_season_var_shape_prior must be of type float.')
                if any(np.isnan(i) for i in trig_season_var_shape_prior):
                    raise ValueError('No values in trig_season_var_shape_prior can be NaN.')
                if any(np.isinf(i) for i in trig_season_var_shape_prior):
                    raise ValueError('No values in trig_season_var_shape_prior can be Inf/-Inf.')
                if not all(i > 0 for i in trig_season_var_shape_prior):
                    raise ValueError('All values in trig_season_var_shape_prior must be strictly positive floats.')

            if trig_season_var_scale_prior is not None:
                if not isinstance(trig_season_var_scale_prior, tuple):
                    raise TypeError('trig_seasonal_var_scale_prior must be a tuple '
                                    'to accommodate potentially multiple seasonality.')
                if len(trig_season_var_scale_prior) != sum(self.stochastic_trig_seasonal):
                    raise ValueError('The number of elements in trig_season_var_scale_prior must match the '
                                     'number of stochastic periodicities in trig_seasonal. That is, for each '
                                     'stochastic periodicity in trig_seasonal, there must be a corresponding '
                                     'scale prior.')
                if not all(isinstance(i, float) for i in trig_season_var_scale_prior):
                    raise TypeError('All values in trig_season_var_scale_prior must be of type float.')
                if any(np.isinf(i) for i in trig_season_var_scale_prior):
                    raise ValueError('No values in trig_season_var_scale_prior can be NaN.')
                if any(np.isinf(i) for i in trig_season_var_scale_prior):
                    raise ValueError('No values in trig_season_var_scale_prior can be Inf/-Inf.')
                if not all(i > 0 for i in trig_season_var_scale_prior):
                    raise ValueError('All values in trig_season_var_scale_prior must be strictly positive floats.')

        # Predictors prior check
        if self.has_predictors:
            if reg_coeff_mean_prior is not None:
                if not isinstance(reg_coeff_mean_prior, np.ndarray):
                    raise TypeError('reg_coeff_mean_prior must be of type Numpy ndarray.')
                if reg_coeff_mean_prior.dtype != 'float64':
                    raise TypeError('All values in reg_coeff_mean_prior must be of type float.')
                if not reg_coeff_mean_prior.shape == (self.num_predictors, 1):
                    raise ValueError(f'reg_coeff_mean_prior must have shape ({self.num_predictors}, 1).')
                if np.any(np.isnan(reg_coeff_mean_prior)):
                    raise ValueError('reg_coeff_mean_prior cannot have NaN values.')
                if np.any(np.isinf(reg_coeff_mean_prior)):
                    raise ValueError('reg_coeff_mean_prior cannot have Inf and/or -Inf values.')

            if reg_coeff_prec_prior is not None:
                if not isinstance(reg_coeff_prec_prior, np.ndarray):
                    raise TypeError('reg_coeff_prec_prior must be of type Numpy ndarray.')
                if reg_coeff_prec_prior.dtype != 'float64':
                    raise TypeError('All values in reg_coeff_prec_prior must be of type float.')
                if not reg_coeff_prec_prior.shape == (self.num_predictors, self.num_predictors):
                    raise ValueError(f'reg_coeff_prec_prior must have shape ({self.num_predictors}, '
                                     f'{self.num_predictors}).')
                if not ao.is_positive_definite(reg_coeff_prec_prior):
                    raise ValueError('reg_coeff_prec_prior must be a positive definite matrix.')
                if not ao.is_symmetric(reg_coeff_prec_prior):
                    raise ValueError('reg_coeff_prec_prior must be a symmetric matrix.')

            if zellner_prior_obs is not None:
                if not isinstance(zellner_prior_obs, float):
                    raise TypeError('zellner_prior_obs must be of type float')
                if not 0 < zellner_prior_obs:
                    raise ValueError('zellner_prior_obs must be a strictly positive float.')

        # Define variables
        y = self.response
        n = self.num_obs
        q = self.num_stoch_states
        m = self.num_state_eqs
        Z = self.observation_matrix()
        T = self.state_transition_matrix
        C = self.state_intercept_matrix
        R = self.state_error_transformation_matrix
        H = self.state_sse_transformation_matrix
        X = self.predictors

        # Bring in the model configuration from _model_setup()
        model = self._model_setup(response_var_shape_prior, response_var_scale_prior,
                                  level_var_shape_prior, level_var_scale_prior,
                                  damped_level_coeff_mean_prior, damped_level_coeff_prec_prior,
                                  trend_var_shape_prior, trend_var_scale_prior,
                                  damped_trend_coeff_mean_prior, damped_trend_coeff_prec_prior,
                                  lag_season_var_shape_prior, lag_season_var_scale_prior,
                                  damped_lag_season_coeff_mean_prior, damped_lag_season_coeff_prec_prior,
                                  dum_season_var_shape_prior, dum_season_var_scale_prior,
                                  trig_season_var_shape_prior, trig_season_var_scale_prior,
                                  reg_coeff_mean_prior, reg_coeff_prec_prior,
                                  zellner_prior_obs)

        # Bring in model configuration
        components = model.components
        response_var_scale_prior = model.response_var_scale_prior
        response_var_shape_post = model.response_var_shape_post
        state_var_scale_prior = model.state_var_scale_prior
        state_var_shape_post = model.state_var_shape_post
        gibbs_iter0_init_state = model.gibbs_iter0_init_state
        gibbs_iter0_response_error_variance = model.gibbs_iter0_response_error_variance
        gibbs_iter0_state_error_covariance = model.gibbs_iter0_state_error_covariance
        init_state_plus_values = model.init_state_plus_values
        init_state_covariance = model.init_state_covariance
        damped_level_coeff_mean_prior = model.damped_level_coeff_mean_prior
        damped_level_coeff_prec_prior = model.damped_level_coeff_prec_prior
        gibbs_iter0_damped_level_coeff = model.gibbs_iter0_damped_level_coeff
        damped_trend_coeff_mean_prior = model.damped_trend_coeff_mean_prior
        damped_trend_coeff_prec_prior = model.damped_trend_coeff_prec_prior
        gibbs_iter0_damped_trend_coeff = model.gibbs_iter0_damped_trend_coeff
        damped_lag_season_coeff_mean_prior = model.damped_lag_season_coeff_mean_prior
        damped_lag_season_coeff_prec_prior = model.damped_lag_season_coeff_prec_prior
        gibbs_iter0_damped_season_coeff = model.gibbs_iter0_damped_season_coeff

        # Identify NaN values in y, if any
        y_nan_indicator = np.isnan(y) * 1.
        y_no_nan = ao.replace_nan(y)

        # Initialize output arrays
        if q > 0:
            state_error_covariance = np.empty((num_samp, q, q))
        else:
            state_error_covariance = np.empty((num_samp, 0, 0))

        response_error_variance = np.empty((num_samp, 1, 1))
        smoothed_errors = np.empty((num_samp, n, 1 + q, 1))
        smoothed_state = np.empty((num_samp, n + 1, m, 1))
        smoothed_prediction = np.empty((num_samp, n, 1))
        filtered_state = np.empty((num_samp, n + 1, m, 1))
        filtered_prediction = np.empty((num_samp, n, 1))
        state_covariance = np.empty((num_samp, n + 1, m, m))
        response_variance = np.empty((num_samp, n, 1, 1))

        # Initialize damped level coefficient output, if applicable
        if self.level and self.damped_level:
            damped_level_coefficient = np.empty((num_samp, 1, 1))
            level_stoch_idx = [components['Level']['stochastic_index']]
            level_state_eqn_idx = [components['Level']['start_state_eqn_index']]
            ar_level_tran_idx = self.damped_level_transition_index
        else:
            damped_level_coefficient = np.array([[[]]])

        # Initialize damped trend coefficient output, if applicable
        if self.trend and self.damped_trend:
            damped_trend_coefficient = np.empty((num_samp, 1, 1))
            trend_stoch_idx = [components['Trend']['stochastic_index']]
            trend_state_eqn_idx = [components['Trend']['start_state_eqn_index']]
            ar_trend_tran_idx = self.damped_trend_transition_index
        else:
            damped_trend_coefficient = np.array([[[]]])

        # Initialize damped lag_seasonal coefficient output, if applicable
        if len(self.lag_seasonal) > 0 and self.num_damped_lag_season > 0:
            damped_season_coefficients = np.empty((num_samp, self.num_damped_lag_season, 1))
            season_stoch_idx = []
            season_state_eqn_idx = []
            ar_season_tran_idx = self.damped_lag_season_transition_index
            for j in self.lag_seasonal:
                season_stoch_idx.append(components[f'Lag-Seasonal.{j}']['stochastic_index'])
                season_state_eqn_idx.append(components[f'Lag-Seasonal.{j}']['start_state_eqn_index'])
        else:
            damped_season_coefficients = np.array([[[]]])

        # Helper matrices
        q_eye = np.eye(q)
        n_ones = np.ones((n, 1))

        if self.has_predictors:
            reg_coeff_mean_prior = model.reg_coeff_mean_prior
            reg_coeff_prec_prior = model.reg_coeff_prec_prior
            reg_coeff_cov_post = model.reg_coeff_cov_post
            gibbs_iter0_reg_coeff = model.gibbs_iter0_reg_coeff
            regression_coefficients = np.empty((num_samp, self.num_predictors, 1))
        else:
            regression_coefficients = np.array([[[]]])

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

            if self.level and self.damped_level:
                if s < 1:
                    damped_level_coeff = gibbs_iter0_damped_level_coeff
                else:
                    damped_level_coeff = damped_level_coefficient[s - 1]

                ar_level_coeff = damped_level_coeff[0, 0]
                if abs(ar_level_coeff) < 1:
                    damped_level_var = (state_err_cov[level_stoch_idx[0], level_stoch_idx[0]]
                                        / (1. - ar_level_coeff ** 2))
                    init_state_plus_values[level_state_eqn_idx[0]] = dist.vec_norm(0., np.sqrt(damped_level_var))
                    init_state_covariance[level_state_eqn_idx[0], level_state_eqn_idx[0]] = damped_level_var
                else:
                    init_state_plus_values[level_state_eqn_idx[0]] = 0.
                    init_state_covariance[level_state_eqn_idx[0], level_state_eqn_idx[0]] = 1e6

                T[ar_level_tran_idx] = ar_level_coeff

            if self.trend and self.damped_trend:
                if s < 1:
                    damped_trend_coeff = gibbs_iter0_damped_trend_coeff
                else:
                    damped_trend_coeff = damped_trend_coefficient[s - 1]

                ar_trend_coeff = damped_trend_coeff[0, 0]
                if abs(ar_trend_coeff) < 1:
                    damped_trend_var = (state_err_cov[trend_stoch_idx[0], trend_stoch_idx[0]]
                                        / (1. - ar_trend_coeff ** 2))
                    init_state_plus_values[trend_state_eqn_idx[0]] = dist.vec_norm(0., np.sqrt(damped_trend_var))
                    init_state_covariance[trend_state_eqn_idx[0], trend_state_eqn_idx[0]] = damped_trend_var
                else:
                    init_state_plus_values[trend_state_eqn_idx[0]] = 0.
                    init_state_covariance[trend_state_eqn_idx[0], trend_state_eqn_idx[0]] = 1e6

                T[ar_trend_tran_idx] = ar_trend_coeff

            if len(self.lag_seasonal) > 0 and self.num_damped_lag_season > 0:
                if s < 1:
                    damped_season_coeff = gibbs_iter0_damped_season_coeff
                else:
                    damped_season_coeff = damped_season_coefficients[s - 1]

                for j in range(self.num_damped_lag_season):
                    ar_season_coeff = damped_season_coeff[j, 0]
                    if abs(ar_season_coeff) < 1:
                        damped_season_var = (state_err_cov[season_stoch_idx[j], season_stoch_idx[j]]
                                             / (1. - ar_season_coeff ** 2))
                        init_state_plus_values[season_state_eqn_idx[j]] = dist.vec_norm(0., np.sqrt(damped_season_var))
                        init_state_covariance[season_state_eqn_idx[j], season_state_eqn_idx[j]] = damped_season_var
                    else:
                        init_state_plus_values[season_state_eqn_idx[j]] = 0.
                        init_state_covariance[season_state_eqn_idx[j], season_state_eqn_idx[j]] = 1e6

                    T[ar_season_tran_idx[j]] = ar_season_coeff

            # Filtered state
            y_kf = kf(y=y,
                      observation_matrix=Z,
                      state_transition_matrix=T,
                      state_intercept_matrix=C,
                      state_error_transformation_matrix=R,
                      response_error_variance_matrix=response_err_var,
                      state_error_covariance_matrix=state_err_cov,
                      init_state=init_state_values,
                      init_state_covariance=init_state_covariance)

            filtered_state[s] = y_kf.filtered_state
            state_covariance[s] = y_kf.state_covariance
            filtered_prediction[s] = y_kf.one_step_ahead_prediction
            response_variance[s] = y_kf.response_variance

            # Smoothed state from D-K smoother
            dk = dks(y=y,
                     observation_matrix=Z,
                     state_transition_matrix=T,
                     state_intercept_matrix=C,
                     state_error_transformation_matrix=R,
                     response_error_variance_matrix=response_err_var,
                     state_error_covariance_matrix=state_err_cov,
                     init_state_plus=init_state_plus_values,
                     init_state=init_state_values,
                     init_state_covariance=init_state_covariance,
                     has_predictors=self.has_predictors)

            # Smoothed disturbances and state
            smoothed_errors[s] = dk.simulated_smoothed_errors
            smoothed_state[s] = dk.simulated_smoothed_state
            smoothed_prediction[s] = dk.simulated_smoothed_prediction

            # Get new draws for state variances
            if q > 0:
                state_resid = smoothed_errors[s][:, 1:, 0]
                state_sse = dot(state_resid.T ** 2, n_ones)
                state_var_scale_post = state_var_scale_prior + 0.5 * dot(H, state_sse)
                state_err_var_post = dist.vec_ig(state_var_shape_post, state_var_scale_post)
                state_error_covariance[s] = q_eye * dot(H.T, state_err_var_post)

            # Get new draw for the level's AR(1) coefficient, if applicable
            if self.level and self.damped_level:
                ar_state_post_upd_args = dict(smoothed_state=smoothed_state[s],
                                              lag=(1,),
                                              mean_prior=damped_level_coeff_mean_prior,
                                              precision_prior=damped_level_coeff_prec_prior,
                                              state_err_var_post=state_err_var_post,
                                              state_eqn_index=level_state_eqn_idx,
                                              state_err_var_post_index=level_stoch_idx)
                damped_level_coefficient[s] = _ar_state_post_upd(**ar_state_post_upd_args)

            # Get new draw for the trend's AR(1) coefficient, if applicable
            if self.trend and self.damped_trend:
                ar_state_post_upd_args = dict(smoothed_state=smoothed_state[s],
                                              lag=(1,),
                                              mean_prior=damped_trend_coeff_mean_prior,
                                              precision_prior=damped_trend_coeff_prec_prior,
                                              state_err_var_post=state_err_var_post,
                                              state_eqn_index=trend_state_eqn_idx,
                                              state_err_var_post_index=trend_stoch_idx)
                damped_trend_coefficient[s] = _ar_state_post_upd(**ar_state_post_upd_args)

            # Get new draw for lag_seasonal AR(1) coefficients, if applicable
            if len(self.lag_seasonal) > 0 and self.num_damped_lag_season > 0:
                ar_state_post_upd_args = dict(smoothed_state=smoothed_state[s],
                                              lag=self.lag_season_damped_lags,
                                              mean_prior=damped_lag_season_coeff_mean_prior,
                                              precision_prior=damped_lag_season_coeff_prec_prior,
                                              state_err_var_post=state_err_var_post,
                                              state_eqn_index=season_state_eqn_idx,
                                              state_err_var_post_index=season_stoch_idx)
                damped_season_coefficients[s] = _ar_state_post_upd(**ar_state_post_upd_args)

            # Get new draw for observation variance
            smooth_one_step_ahead_prediction_resid = smoothed_errors[s, :, 0]
            response_var_scale_post = (response_var_scale_prior
                                       + 0.5 * dot(smooth_one_step_ahead_prediction_resid.T,
                                                   smooth_one_step_ahead_prediction_resid))
            response_error_variance[s] = dist.vec_ig(response_var_shape_post, response_var_scale_post)

            # Get new draw for regression coefficients
            if self.has_predictors:
                y_adj = y_no_nan + y_nan_indicator * smoothed_prediction[s]
                smooth_time_prediction = smoothed_prediction[s] - Z[:, :, -1]
                y_tilde = y_adj - smooth_time_prediction  # y with smooth time prediction subtracted out
                reg_coeff_mean_post = dot(reg_coeff_cov_post,
                                          (dot(X.T, y_tilde) + dot(reg_coeff_prec_prior,
                                                                   reg_coeff_mean_prior)))

                cov_post = response_error_variance[s][0, 0] * reg_coeff_cov_post
                regression_coefficients[s] = (np
                                              .random
                                              .multivariate_normal(mean=reg_coeff_mean_post.flatten(),
                                                                   cov=cov_post).reshape(-1, 1))

        self.posterior = Posterior(num_samp, smoothed_state,
                                   smoothed_errors, smoothed_prediction,
                                   filtered_state, filtered_prediction,
                                   response_variance, state_covariance,
                                   response_error_variance, state_error_covariance,
                                   damped_level_coefficient, damped_trend_coefficient,
                                   damped_season_coefficients, regression_coefficients)

        return self.posterior

    def forecast(self,
                 num_periods: int,
                 burn: int = 0,
                 future_predictors: Union[np.ndarray, pd.Series, pd.DataFrame] = np.array([[]])):

        """

        :param num_periods: positive integer. Defines how many future periods to forecast.

        :param burn: non-negative integer. Represents how many of the first posterior samples to
        ignore for computing statistics like the mean and variance of a parameter. Default value is 0.

        :param future_predictors: ndarray of dimension (k, 1), where k is the number of predictors.
        Data type is float64. Default is an empty array.

        :return: ndarray of dimension (S, h, 1), where S is the number of posterior samples and
        h is the number of future periods to be forecast. This array represents the posterior
        forecast distribution for the response.
        """

        Z = self.observation_matrix(num_rows=num_periods)
        T = self.state_transition_matrix
        C = self.state_intercept_matrix
        R = self.state_error_transformation_matrix

        if not isinstance(num_periods, int):
            raise TypeError('num_periods must be of type integer.')
        else:
            if not num_periods > 0:
                raise ValueError('num_periods must be a strictly positive integer.')

        if not isinstance(burn, int):
            raise TypeError('burn must be of type integer.')
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

        # -- check if object type is valid
        if not isinstance(future_predictors, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError("The future_predictors array must be a NumPy array, Pandas Series, "
                            "or Pandas DataFrame.")
        else:
            fut_pred = future_predictors.copy()
            if self.has_predictors:
                # Check and prepare future predictor data
                # -- data types match across predictors and future_predictors
                if not isinstance(future_predictors, type(self._predictors)):
                    raise TypeError('Object types for predictors and future_predictors must match.')

                else:
                    # -- if Pandas type, grab index and column names
                    if isinstance(future_predictors, (pd.Series, pd.DataFrame)):
                        if not isinstance(future_predictors.index, type(self.future_time_index)):
                            raise TypeError('The future_predictors and predictors indexes must be of the same type.')

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

                        fut_pred = fut_pred.sort_index().to_numpy()

                # -- dimensions
                if fut_pred.ndim == 0:
                    raise ValueError('The future_predictors array must have dimension 1 or 2. Dimension is 0.')
                if fut_pred.ndim == 1:
                    fut_pred = fut_pred.reshape(-1, 1)
                if fut_pred.ndim > 2:
                    raise ValueError('The future_predictors array must have dimension 1 or 2. Dimension exceeds 2.')
                if fut_pred.ndim == 2:
                    if 1 in fut_pred.shape:
                        fut_pred = fut_pred.reshape(-1, 1)
                if np.isnan(fut_pred).any():
                    raise ValueError('The future_predictors array cannot have null values.')
                if np.isinf(fut_pred).any():
                    raise ValueError('The future_predictors array cannot have Inf and/or -Inf values.')

                # Final sanity checks
                if self.num_predictors != fut_pred.shape[1]:
                    raise ValueError(f'The number of predictors used for historical estimation {self.num_predictors} '
                                     f'does not match the number of predictors specified for forecasting '
                                     f'{fut_pred.shape[1]}. The same set of predictors must be used.')

                if num_periods > fut_pred.shape[0]:
                    raise ValueError(f'The number of requested forecast periods {num_periods} exceeds the '
                                     f'number of observations provided in future_predictors {fut_pred.shape[0]}. '
                                     f'The former must be no larger than the latter.')
                else:
                    if num_periods < fut_pred.shape[0]:
                        warnings.warn(f'The number of requested forecast periods {num_periods} is less than the '
                                      f'number of observations provided in future_predictors {fut_pred.shape[0]}. '
                                      f'Only the first {num_periods} observations will be used '
                                      f'in future_predictors.')

        y_forecast = _forecast(posterior=self.posterior,
                               num_periods=num_periods,
                               state_observation_matrix=Z,
                               state_transition_matrix=T,
                               state_intercept_matrix=C,
                               state_error_transformation_matrix=R,
                               future_predictors=fut_pred,
                               burn=burn,
                               damped_level_transition_index=self.damped_level_transition_index,
                               damped_trend_transition_index=self.damped_trend_transition_index,
                               damped_season_transition_index=self.damped_lag_season_transition_index)

        return y_forecast

    def plot_components(self,
                        burn: int = 0,
                        cred_int_level: float = 0.05,
                        random_sample_size_prop: float = 1.,
                        smoothed: bool = True):

        """

        :param burn: non-negative integer. Represents how many of the first posterior samples to
        ignore for computing statistics like the mean and variance of a parameter. Default value is 0.

        :param cred_int_level: float in (0, 1). Defines the width of the predictive distribution.
        E.g., a value of 0.05 will represent all values of the predicted variable between the
        2.5% and 97.5% quantiles.

        :param random_sample_size_prop: float in interval (0, 1]. Represents the proportion of the
        posterior samples to take for constructing the posterior predictive distribution. Sampling is
        done without replacement. Default value is 1.

        :param smoothed: boolean. If True, the smoothed Kalman values will be plotted (as opposed
        to the filtered Kalman values).

        :return: Return a matplotlib.pyplot figure that plots the posterior predictive distribution
        of the response and the specified states.
        """

        if not isinstance(burn, int):
            raise TypeError('burn must be of type integer.')
        else:
            if burn < 0:
                raise ValueError('burn must be a non-negative integer.')

        if not isinstance(cred_int_level, float):
            raise TypeError('cred_int_level must be of type float.')
        else:
            if cred_int_level <= 0 or cred_int_level >= 1:
                raise ValueError('cred_int_level must be a value in the '
                                 'interval (0, 1).')

        if not isinstance(random_sample_size_prop, float):
            raise TypeError('random_sample_size_prop must be of type float.')
        else:
            if random_sample_size_prop <= 0 or random_sample_size_prop > 1:
                raise ValueError('random_sample_size_prop must be a value in the '
                                 'interval (0, 1].')

        if not isinstance(smoothed, bool):
            raise TypeError('smoothed must be of type bool.')

        num_first_obs_ignore = self.num_first_obs_ignore

        if self.has_predictors:
            X = self.predictors[num_first_obs_ignore:, :]
            reg_coeff = self.posterior.regression_coefficients[burn:, :, 0].T

        y = self.response[num_first_obs_ignore:, 0]
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
        ax[0].legend(('Observed', 'One-step-ahead prediction', f'{100 * (1 - cred_int_level)}% credible interval'),
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
                state_eqn_index = components['Trend']['start_state_eqn_index']
                time_component = state[:, :, state_eqn_index, 0]
                ax[i + 1].plot(historical_time_index, np.mean(time_component, axis=0))
                lb = np.quantile(time_component, cred_int_lb, axis=0)
                ub = np.quantile(time_component, cred_int_ub, axis=0)
                ax[i + 1].fill_between(historical_time_index, lb, ub, alpha=0.2)
                ax[i + 1].title.set_text(c)

            if c not in ('Irregular', 'Regression', 'Trend'):
                v = components[c]
                start_index, end_index = v['start_state_eqn_index'], v['end_state_eqn_index']
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

        :param burn: non-negative integer. Represents how many of the first posterior samples to
        ignore for computing statistics like the mean and variance of a parameter. Default value is 0.

        :param cred_int_level: float in (0, 1). Defines the width of the predictive distribution.
        E.g., a value of 0.05 will represent all values of the predicted variable between the
        2.5% and 97.5% quantiles.

        :return: A dictionary that summarizes the statistics of each of the model's parameters.
        Statistics include the mean, standard deviation, and lower and upper quantiles defined by
        cred_int_level.
        """

        if not isinstance(burn, int):
            raise TypeError('burn must be of type integer.')
        else:
            if burn < 0:
                raise ValueError('burn must be a non-negative integer.')

        if not isinstance(cred_int_level, float):
            raise TypeError('cred_int_level must be of type float.')
        else:
            if not 0 < cred_int_level < 1:
                raise ValueError('cred_int_level must be a value in the '
                                 'interval (0, 1).')
            lb = 0.5 * cred_int_level
            ub = 1. - lb

        components = self.model_setup.components
        resp_err_var = self.posterior.response_error_variance[burn:]
        state_err_cov = self.posterior.state_error_covariance[burn:]
        res = dict()
        res['Number of posterior samples (after burn)'] = self.posterior.num_samp - burn

        d = 0  # index for damped seasonal components
        for c in components:
            if c == 'Irregular':
                res[f"Posterior.Mean[{c}.Var]"] = np.mean(resp_err_var)
                res[f"Posterior.StdDev[{c}.Var]"] = np.std(resp_err_var)
                res[f"Posterior.CredInt.LB[{c}.Var]"] = np.quantile(resp_err_var, lb)
                res[f"Posterior.CredInt.UB[{c}.Var]"] = np.quantile(resp_err_var, ub)

            if c not in ('Irregular', 'Regression'):
                v = components[c]
                stochastic = v['stochastic']
                damped = v['damped']
                idx = v['stochastic_index']

                if stochastic:
                    res[f"Posterior.Mean[{c}.Var]"] = np.mean(state_err_cov[:, idx, idx])
                    res[f"Posterior.StdDev[{c}.Var]"] = np.std(state_err_cov[:, idx, idx])
                    res[f"Posterior.CredInt.LB[{c}.Var]"] = np.quantile(state_err_cov[:, idx, idx], lb)
                    res[f"Posterior.CredInt.UB[{c}.Var]"] = np.quantile(state_err_cov[:, idx, idx], ub)

                if c == 'Level' and damped:
                    ar_coeff = self.posterior.damped_level_coefficient[burn:, 0, 0]
                    res[f"Posterior.Mean[{c}.AR]"] = np.mean(ar_coeff)
                    res[f"Posterior.StdDev[{c}.AR]"] = np.std(ar_coeff)
                    res[f"Posterior.CredInt.LB[{c}.AR]"] = np.quantile(ar_coeff, lb)
                    res[f"Posterior.CredInt.LB[{c}.AR]"] = np.quantile(ar_coeff, ub)

                if c == 'Trend' and damped:
                    ar_coeff = self.posterior.damped_trend_coefficient[burn:, 0, 0]
                    res[f"Posterior.Mean[{c}.AR]"] = np.mean(ar_coeff)
                    res[f"Posterior.StdDev[{c}.AR]"] = np.std(ar_coeff)
                    res[f"Posterior.CredInt.LB[{c}.AR]"] = np.quantile(ar_coeff, lb)
                    res[f"Posterior.CredInt.LB[{c}.AR]"] = np.quantile(ar_coeff, ub)

                if 'Lag-Seasonal' in c:
                    if damped:
                        ar_coeff = self.posterior.damped_season_coefficients[burn:, d, 0]
                        res[f"Posterior.Mean[{c}.AR]"] = np.mean(ar_coeff)
                        res[f"Posterior.StdDev[{c}.AR]"] = np.std(ar_coeff)
                        res[f"Posterior.CredInt.LB[{c}.AR]"] = np.quantile(ar_coeff, lb)
                        res[f"Posterior.CredInt.LB[{c}.AR]"] = np.quantile(ar_coeff, ub)
                        d += 1

            if c == 'Regression':
                reg_coeff = self.posterior.regression_coefficients[burn:, :, 0]
                mean_reg_coeff = np.mean(reg_coeff, axis=0)
                std_reg_coeff = np.std(reg_coeff, axis=0)
                ci_lb_reg_coeff = np.quantile(reg_coeff, lb, axis=0)
                ci_ub_reg_coeff = np.quantile(reg_coeff, ub, axis=0)
                for k in range(self.num_predictors):
                    res[f"Posterior.Mean[beta.{self.predictors_names[k]}]"] = mean_reg_coeff[k]
                    res[f"Posterior.StdDev[beta.{self.predictors_names[k]}]"] = std_reg_coeff[k]
                    res[f"Posterior.CredInt.LB[beta.{self.predictors_names[k]}]"] = ci_lb_reg_coeff[k]
                    res[f"Posterior.CredInt.UB[beta.{self.predictors_names[k]}]"] = ci_ub_reg_coeff[k]

        return res
