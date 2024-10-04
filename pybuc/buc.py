import numpy as np
from numpy import dot
from numba import njit
import matplotlib.pyplot as plt
import warnings
from typing import Union, NamedTuple
import pandas as pd
from pybuc.statespace.kalman_filter import kalman_filter as kf
from pybuc.statespace.durbin_koopman_smoother import dk_smoother as dks
from pybuc.utils import array_operations as ao
from pybuc.vectorized import distributions as dist
from pybuc.model_assessment.performance import watanabe_akaike, WAIC
from seaborn import histplot, lineplot
from statsmodels.tsa.statespace.structural import UnobservedComponents as UC


class MaxIterSamplingError(Exception):
    def __init__(self,
                 upper_var_limit,
                 max_samp_iter):
        self.upper_var_limit = upper_var_limit
        self.max_samp_iter = max_samp_iter
        self.message = f"""Maximum number of sampling iterations ({max_samp_iter}) exceeded based on
                       upper acceptable value for model variances ({upper_var_limit}). Consider a
                       combination of the following: (1) scaling your data (e.g., scaling the
                       response by its standard deviation), (2) redefining the specification of
                       the model in terms of components and priors, (3) increasing the upper
                       acceptable value for model variances governed by upper_var_limit, and/or 
                       (4) increasing the maximum number of sampling iterations governed by 
                       max_iter_samp_factor.
                       """
        super().__init__(self.message)


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
    reg_ninvg_coeff_cov_post: np.ndarray
    reg_ninvg_coeff_prec_post: np.ndarray
    zellner_prior_obs: Union[int, float]
    gibbs_iter0_reg_coeff: np.ndarray


class Forecast(NamedTuple):
    response_forecast: np.ndarray
    state_forecast: np.ndarray


@njit
def _set_seed(value):
    np.random.seed(value)


@njit(cache=True)
def _simulate_posterior_predictive_filtered_state(posterior: Posterior,
                                                  burn: int = 0,
                                                  num_first_obs_ignore: int = 0,
                                                  random_sample_size_prop: float = 1.,
                                                  has_predictors: bool = False) -> np.ndarray:
    """
    Generates the posterior predictive density of the filtered state vector. The filtered, as
    opposed to the smoothed, states are used. Using smoothed states assumes we have all
    information available (past, present, future) at any given time t. In contrast, the
    filtered state assumes we only have information up until time t - 1 for predicting
    time t. The posterior distribution for the smoothed state vector is sampled directly
    in BayesianUnobservedComponents.sample(). The posterior of the filtered state vector,
    however, is not. Therefore, this function should be used only if one wants the
    posterior predictive density of the filtered state vector. The state equations are
    represented by

    a(t+1) = T.a(t) + R * state_error(t+1),

    where T, a(t), R, and state_error(t) are the state transition matrix, state value,
    state error transformation matrix, and error at time t, respectively. The
    posterior predictive distribution for the filtered state at any time t is obtained by
    sampling from

        a(t+1) | t ~ N(T.a(t), P(t+1 | t)),

    where

        P(t+1|t) = Var[a(t+1) | t] = T.P(t).L(t)' + R.StateErrCovMat.R'

    is the state covariance matrix. L(t) = T - K(t).Z(t), where K(t) is the Kalman
    gain at time t, and StateErrCovMat = Var[state_error(t+1) | t] is the state
    error covariance matrix.

    Each vector a(t|t-1), t=1,...,n, must be drawn from a multivariate normal
    distribution that obeys the correlation implied by the state covariance matrix.
    Namely, sample

        a(t+1) | T, t ~ N(T.a(t) | a(t), T.P(t-1|t-1).T' + R.StateErrCovMat.R'),

    where the transition matrix T can include autoregressive coefficients. Otherwise,
    T is a constant.

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
        mean = posterior.filtered_state[burn:, num_first_obs_ignore:-1, :-1]
        cov = posterior.state_covariance[burn:, num_first_obs_ignore:-1, :-1, :-1]
    else:
        mean = posterior.filtered_state[burn:, num_first_obs_ignore:-1, :]
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

    state_post = np.empty((num_samp, n, m, 1), dtype=np.float64)
    m_zeros, m_ones = np.zeros((m, 1)), np.ones((m, 1))
    for s in S:
        for t in range(n):
            chol = np.linalg.cholesky(cov[s, t])
            z = dist.vec_norm(m_zeros, m_ones)
            state_post[s, t] = chol.dot(z) + mean[s, t]

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
              damped_season_transition_index: tuple[tuple, ...] = ()) -> tuple:
    """
    The posterior forecast distributions for the response and the state vector.

    To generate the posterior forecast distributions, the posterior smoothed
    states are required. Specifically, the last smoothed state value (i.e.,
    the smoothed state value for time n) is needed to kick off the recursive
    forecast. Also required are the posterior for the response error variance
    and the posterior for the state error covariance matrix.

    Give the posterior samples s=1,...,S and the smoothed state vector at time
    n, the posterior forecast for the response and state vector are computed as:

        for s = 1,...,S:
            for t = 1,...,h:
                y(s, t) = Z(t).a(s, t) + RespErrVar(s, t)
                a(s, t+1) = T.a(s, t) + R.StateErrCovMat(s, t)

    where a(s, 1) is the smoothed state vector value at time n for posterior sample s.

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

    ndarray of dimension (S, h, m, 1), where S is the number of posterior samples,
    h is the number of future periods to be forecast, and m is the number of state equations.
    This array represents the posterior forecast distribution for the state vector.
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
    state_forecast = np.empty((num_samp, num_periods, m, 1), dtype=np.float64)
    num_periods_zeros = np.zeros((num_periods, 1))
    num_periods_q_zeros = np.zeros((num_periods, q, 1))
    num_periods_ones = np.ones((num_periods, 1))
    num_periods_q_ones = np.ones((num_periods, q, 1))

    for s in range(num_samp):
        obs_error = dist.vec_norm(num_periods_zeros,
                                  num_periods_ones
                                  * np.sqrt(response_error_variance[s][0, 0]))
        if q > 0:
            state_error = dist.vec_norm(num_periods_q_zeros,
                                        num_periods_q_ones
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

        state = np.empty((num_periods + 1, m, 1), dtype=np.float64)
        state[0] = smoothed_state[s, -1]
        for t in range(num_periods):
            y_forecast[s, t] = Z[t].dot(state[t]) + obs_error[t]
            if q > 0:
                state[t + 1] = C + T.dot(state[t]) + R.dot(state_error[t])
            else:
                state[t + 1] = C + T.dot(state[t])

        state_forecast[s] = state[1:]

    return y_forecast, state_forecast


class BayesianUnobservedComponents:
    def __init__(self,
                 response: Union[np.ndarray, list, tuple, pd.Series, pd.DataFrame],
                 predictors: Union[np.ndarray, list, tuple, pd.Series, pd.DataFrame] = None,
                 level: bool = False,
                 stochastic_level: bool = True,
                 damped_level: bool = False,
                 trend: bool = False,
                 stochastic_trend: bool = True,
                 damped_trend: bool = False,
                 lag_seasonal: tuple[int, ...] = (),
                 stochastic_lag_seasonal: tuple[bool, ...] = (),
                 damped_lag_seasonal: tuple[bool, ...] = (),
                 dummy_seasonal: tuple[int, ...] = (),
                 stochastic_dummy_seasonal: tuple[bool, ...] = (),
                 trig_seasonal: tuple[tuple[int, int], ...] = (),
                 stochastic_trig_seasonal: tuple[bool, ...] = (),
                 seed: int = None):

        """

        :param response: Numpy array, list, tuple, Pandas Series, or Pandas DataFrame, float64.
        Array that represents the response variable.

        :param predictors: Numpy array, list, tuple, Pandas Series, or Pandas DataFrame, float64.
        Array that represents the predictors, if any, to be used for predicting the response variable.
        Default is None.

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

        self.model_setup = None
        self.response_name = None
        self.predictors_names = None
        self.historical_time_index = None
        self.future_time_index = None
        self.posterior = None
        self.parameters = []
        self.num_sampling_iterations = None
        self.high_posterior_variance = None
        self._components_ppd = None

        # CHECK AND PREPARE RESPONSE DATA
        # -- data types, name, and index
        if not isinstance(response, (np.ndarray, list, tuple, pd.Series, pd.DataFrame)):
            raise TypeError(
                "The response array must be a Numpy array, list, tuple, Pandas Series, "
                "or Pandas DataFrame."
            )
        else:
            if isinstance(response, (list, tuple)):
                resp = np.asarray(response, dtype=np.float64)
            else:
                resp = response.copy()

            self.response_type = type(resp)

            if isinstance(resp, (pd.Series, pd.DataFrame)):
                if isinstance(resp.index, pd.DatetimeIndex):
                    if resp.index.freq is None:
                        warnings.warn(
                            'Frequency of DatetimeIndex is None. Frequency will be inferred '
                            'for response.'
                        )
                        resp.index.freq = pd.infer_freq(resp.index)

                    if not resp.index.is_monotonic_increasing:
                        warnings.warn(
                            "The DatetimeIndex for the response is not in ascending order. "
                            "Data will be sorted.")
                        resp = resp.sort_index()

                    self.historical_time_index = resp.index

                if isinstance(response, pd.Series):
                    self.response_name = [resp.name]
                else:
                    self.response_name = resp.columns.values.tolist()

                resp = resp.to_numpy()

        # -- dimensions
        if resp.dtype != 'float64':
            raise TypeError(
                'All values in the response array must be of type float.')

        if resp.ndim not in (1, 2):
            raise ValueError(
                'The response array must have dimension 1 or 2.')
        elif resp.ndim == 1:
            resp = resp.reshape(-1, 1)
        else:
            if all(i > 1 for i in resp.shape):
                raise ValueError(
                    'The response array must have shape (1, n) or (n, 1), '
                    'where n is the number of observations. Both the row and column '
                    'count exceed 1.'
                )
            else:
                resp = resp.reshape(-1, 1)

        if np.all(np.isnan(resp)):
            raise ValueError(
                'All values in the response array are null. At least one value must be non-null.'
            )

        if resp.shape[0] < 3:
            raise ValueError(
                'At least three observations are required to fit a model. This restriction '
                'may be removed in the future.'
            )

        # CHECK AND PREPARE PREDICTORS DATA, IF APPLICABLE
        if predictors is not None:
            # -- check if correct data type
            if not isinstance(predictors, (np.ndarray, list, tuple, pd.Series, pd.DataFrame)):
                raise TypeError(
                    "The predictors array must be a Numpy array, list, tuple, Pandas Series, "
                    "or Pandas DataFrame."
                )
            else:
                if isinstance(predictors, (list, tuple)):
                    pred = np.asarray(predictors, dtype=np.float64)
                else:
                    pred = predictors.copy()

                self.predictors_type = type(pred)

                # -- check if response and predictors are same date type.
                if isinstance(response, np.ndarray) and not isinstance(pred, np.ndarray):
                    raise TypeError(
                        'The response array provided is a NumPy array, list, or tuple, '
                        'but the predictors array is not. Object types must match.'
                    )

                if (isinstance(response, (pd.Series, pd.DataFrame)) and
                        not isinstance(pred, (pd.Series, pd.DataFrame))):
                    raise TypeError(
                        'The response array provided is a Pandas Series/DataFrame, but the predictors '
                        'array is not. Object types must match.'
                    )

                # -- get predictor names if a Pandas object and sort index
                if isinstance(pred, (pd.Series, pd.DataFrame)):
                    if isinstance(pred.index, pd.DatetimeIndex):
                        if not pred.index.is_monotonic_increasing:
                            warnings.warn(
                                "The DatetimeIndex for predictors is not in ascending order. "
                                "Data will be sorted."
                            )
                            pred = pred.sort_index()

                    if not (pred.index == response.index).all():
                        raise ValueError(
                            'The response and predictors indexes must match.')

                    if isinstance(pred, pd.Series):
                        self.predictors_names = [pred.name]
                    else:
                        self.predictors_names = pred.columns.values.tolist()

                    pred = pred.to_numpy()

                # -- dimension and null/inf checks
                if pred.dtype != 'float64':
                    raise TypeError(
                        'All values in the predictors array must be of type float.')

                if pred.ndim not in (1, 2):
                    raise ValueError(
                        'The predictors array must have dimension 1 or 2.')
                elif pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                else:
                    pass

                if np.any(np.isnan(pred)):
                    raise ValueError(
                        'The predictors array cannot have null values.')
                if np.any(np.isinf(pred)):
                    raise ValueError(
                        'The predictors array cannot have Inf and/or -Inf values.')

                # -- conformable number of observations
                if pred.shape[0] != resp.shape[0]:
                    raise ValueError(
                        'The number of observations in the predictors array must match '
                        'the number of observations in the response array.'
                    )

                # -- check if design matrix has a constant
                sd_pred = np.std(pred, axis=0)
                if np.any(sd_pred <= 1e-6):
                    raise ValueError(
                        'The predictors array cannot have a column with a constant value. Note that '
                        'the inclusion of a constant/intercept in the predictors array will confound '
                        'with the level if specified. If a constant level without trend or seasonality '
                        'is desired, pass as arguments level=True, stochastic_level=False, trend=False, '
                        'dum_seasonal=(), and trig_seasonal=(). This will replicate standard regression.'
                    )

                # -- warn about model stability if the number of predictors exceeds number of observations
                if pred.shape[1] > pred.shape[0]:
                    warnings.warn(
                        'The number of predictors exceeds the number of observations. '
                        'Results will be sensitive to choice of priors.'
                    )
        else:
            pred = np.array([[]])

        # CHECK AND PREPARE LAG SEASONAL
        if not isinstance(lag_seasonal, tuple):
            raise TypeError(
                'lag_seasonal must be a tuple.'
            )

        if not isinstance(stochastic_lag_seasonal, tuple):
            raise TypeError(
                'stochastic_lag_seasonal must be a tuple.'
            )

        if not isinstance(damped_lag_seasonal, tuple):
            raise TypeError(
                'damped_lag_seasonal must be a tuple.'
            )

        if len(lag_seasonal) > 0:
            if not all(isinstance(v, int) for v in lag_seasonal):
                raise TypeError(
                    'The period for a lag_seasonal component must be an integer.'
                )

            if len(lag_seasonal) != len(set(lag_seasonal)):
                raise ValueError(
                    'Each specified period in lag_seasonal must be distinct.'
                )

            if any(v < 2 for v in lag_seasonal):
                raise ValueError(
                    'The period for a lag_seasonal component must be an integer greater than 1.'
                )

            if len(stochastic_lag_seasonal) > 0:
                if not all(isinstance(v, bool) for v in stochastic_lag_seasonal):
                    raise TypeError(
                        'If a non-empty tuple is passed for the stochastic specification '
                        'of the lag_seasonal components, all elements must be of boolean type.'
                    )

                if len(lag_seasonal) > len(stochastic_lag_seasonal):
                    raise ValueError(
                        'Some of the lag_seasonal components were given a stochastic '
                        'specification, but not all. Partial specification of the stochastic '
                        'profile is not allowed. Either leave the stochastic specification blank '
                        'by passing an empty tuple (), which will default to True '
                        'for all components, or pass a stochastic specification '
                        'for each lag_seasonal component.'
                    )

                if len(lag_seasonal) < len(stochastic_lag_seasonal):
                    raise ValueError(
                        'The tuple which specifies the number of stochastic lag_seasonal components '
                        'has greater length than the tuple that specifies the number of lag_seasonal '
                        'components. Either pass a blank tuple () for the stochastic profile, or a '
                        'boolean tuple of same length as the tuple that specifies the number of '
                        'lag_seasonal components.'
                    )

            if len(damped_lag_seasonal) > 0:
                if not all(isinstance(v, bool) for v in damped_lag_seasonal):
                    raise TypeError(
                        'If a non-empty tuple is passed for the damped specification '
                        'of the lag_seasonal components, all elements must be of boolean type.'
                    )

                if len(lag_seasonal) > len(damped_lag_seasonal):
                    raise ValueError(
                        'Some of the lag_seasonal components were given a damped specification, but '
                        'not all. Partial specification of the damped profile is not allowed. '
                        'Either leave the damped specification blank by passing an empty '
                        'tuple (), which will default to True for all components, or pass a '
                        'damped specification for each lag_seasonal component.'
                    )

                if len(lag_seasonal) < len(damped_lag_seasonal):
                    raise ValueError(
                        'The tuple which specifies the number of damped lag_seasonal components '
                        'has greater length than the tuple that specifies the number of lag_seasonal '
                        'components. Either pass a blank tuple () for the damped profile, or a '
                        'boolean tuple of same length as the tuple that specifies the number of '
                        'lag_seasonal components.'
                    )
                if len(stochastic_lag_seasonal) > 0:
                    for k in range(len(lag_seasonal)):
                        if damped_lag_seasonal[k] and not stochastic_lag_seasonal[k]:
                            raise ValueError(
                                'Every element in damped_lag_seasonal that is True must also '
                                'have a corresponding True in stochastic_lag_seasonal.'
                            )

            if resp.shape[0] < 4 * max(lag_seasonal):
                warnings.warn(
                    f'It is recommended to have an observation count that is at least quadruple the highest '
                    f'periodicity specified. The max periodicity specified in lag_seasonal is {max(lag_seasonal)}.'
                )

        else:
            if len(stochastic_lag_seasonal) > 0:
                raise ValueError(
                    'No lag_seasonal components were specified, but a non-empty '
                    'stochastic profile was passed for lag seasonality. If '
                    'lag_seasonal components are desired, specify the period for each '
                    'component via a tuple passed to the lag_seasonal argument.')
            if len(damped_lag_seasonal) > 0:
                raise ValueError(
                    'No lag_seasonal components were specified, but a non-empty '
                    'damped profile was passed for lag seasonality. If lag_seasonal '
                    'components are desired, specify the period for each '
                    'component via a tuple passed to the lag_seasonal argument.')

        # CHECK AND PREPARE DUMMY SEASONAL
        if not isinstance(dummy_seasonal, tuple):
            raise TypeError(
                'dummy_seasonal must be a tuple.')

        if not isinstance(stochastic_dummy_seasonal, tuple):
            raise TypeError(
                'stochastic_dummy_seasonal must be a tuple.')

        if len(dummy_seasonal) > 0:
            if not all(isinstance(v, int) for v in dummy_seasonal):
                raise TypeError(
                    'The period for a dummy seasonal component must be an integer.')

            if len(dummy_seasonal) != len(set(dummy_seasonal)):
                raise ValueError(
                    'Each specified period in dummy_seasonal must be distinct.')

            if any(v < 2 for v in dummy_seasonal):
                raise ValueError(
                    'The period for a dummy seasonal component must be an integer greater than 1.')

            if len(stochastic_dummy_seasonal) > 0:
                if not all(isinstance(v, bool) for v in stochastic_dummy_seasonal):
                    raise TypeError(
                        'If a non-empty tuple is passed for the stochastic specification '
                        'of the dummy seasonal components, all elements must be of boolean type.')

                if len(dummy_seasonal) > len(stochastic_dummy_seasonal):
                    raise ValueError(
                        'Some of the dummy seasonal components were given a stochastic '
                        'specification, but not all. Partial specification of the stochastic '
                        'profile is not allowed. Either leave the stochastic specification blank '
                        'by passing an empty tuple (), which will default to True '
                        'for all components, or pass a stochastic specification '
                        'for each seasonal component.')

                if len(dummy_seasonal) < len(stochastic_dummy_seasonal):
                    raise ValueError(
                        'The tuple which specifies the number of stochastic dummy seasonal components '
                        'has greater length than the tuple that specifies the number of dummy seasonal '
                        'components. Either pass a blank tuple () for the stochastic profile, or a '
                        'boolean tuple of same length as the tuple that specifies the number of '
                        'dummy seasonal components.')

            if resp.shape[0] < 4 * max(dummy_seasonal):
                warnings.warn(
                    f'It is recommended to have an observation count that is at least quadruple the highest '
                    f'periodicity specified. The max periodicity specified in dummy_seasonal is '
                    f'{max(dummy_seasonal)}.')

        else:
            if len(stochastic_dummy_seasonal) > 0:
                raise ValueError(
                    'No dummy seasonal components were specified, but a non-empty '
                    'stochastic profile was passed for dummy seasonality. If dummy '
                    'seasonal components are desired, specify the period for each '
                    'component via a tuple passed to the dummy_seasonal argument.')

        # CHECK AND PREPARE TRIGONOMETRIC SEASONAL
        if not isinstance(trig_seasonal, tuple):
            raise TypeError(
                'trig_seasonal must be a tuple.')

        if not isinstance(stochastic_trig_seasonal, tuple):
            raise TypeError(
                'stochastic_trig_seasonal must be a tuple.')

        if len(trig_seasonal) > 0:
            if not all(isinstance(v, tuple) for v in trig_seasonal):
                raise TypeError(
                    'Each element in trig_seasonal must be a tuple.')

            if not all(len(v) == 2 for v in trig_seasonal):
                raise ValueError(
                    'A (period, num_harmonics) tuple must be provided for each specified trigonometric '
                    'seasonal component.')

            if not all(isinstance(v[0], int) for v in trig_seasonal):
                raise TypeError(
                    'The period for a specified trigonometric seasonal component must be an integer.')

            if not all(isinstance(v[1], int) for v in trig_seasonal):
                raise TypeError(
                    'The number of harmonics for a specified trigonometric seasonal component must '
                    'be an integer.')

            if any(v[0] < 2 for v in trig_seasonal):
                raise ValueError(
                    'The period for a trigonometric seasonal component must be an integer greater than 1.')

            if any(v[1] < 1 and v[1] != 0 for v in trig_seasonal):
                raise ValueError(
                    'The number of harmonics for a trigonometric seasonal component can take 0 or '
                    'integers at least as large as 1 as valid options. A value of 0 will enforce '
                    'the highest possible number of harmonics for the given period, which is period / 2 '
                    'if period is even, or (period - 1) / 2 if period is odd.')

            trig_periodicities = ()
            for v in trig_seasonal:
                period, num_harmonics = v
                trig_periodicities += (period,)
                if ao.is_odd(period):
                    if num_harmonics > int(period - 1) / 2:
                        raise ValueError(
                            'The number of harmonics for a trigonometric seasonal component cannot '
                            'exceed (period - 1) / 2 when period is odd.')
                else:
                    if num_harmonics > int(period / 2):
                        raise ValueError(
                            'The number of harmonics for a trigonometric seasonal component cannot '
                            'exceed period / 2 when period is even.')

            if len(trig_seasonal) != len(set(trig_periodicities)):
                raise ValueError(
                    'Each specified period in trig_seasonal must be distinct.')

            if len(stochastic_trig_seasonal) > 0:
                if not all(isinstance(v, bool) for v in stochastic_trig_seasonal):
                    raise TypeError(
                        'If a non-empty tuple is passed for the stochastic specification '
                        'of the trigonometric seasonal components, all elements must be of boolean type.')

                if len(trig_seasonal) > len(stochastic_trig_seasonal):
                    raise ValueError(
                        'Some of the trigonometric seasonal components '
                        'were given a stochastic specification, but not all. '
                        'Partial specification of the stochastic profile is not '
                        'allowed. Either leave the stochastic specification blank '
                        'by passing an empty tuple (), which will default to True '
                        'for all components, or pass a stochastic specification '
                        'for each seasonal component.')

                if len(trig_seasonal) < len(stochastic_trig_seasonal):
                    raise ValueError(
                        'The tuple which specifies the number of stochastic trigonometric '
                        'seasonal components has greater length than the tuple that specifies '
                        'the number of trigonometric seasonal components. Either pass a blank '
                        'tuple () for the stochastic profile, or a boolean tuple of same length '
                        'as the tuple that specifies the number of trigonometric seasonal components.')

            if resp.shape[0] < 4 * max(trig_periodicities):
                warnings.warn(
                    f'It is recommended to have an observation count that is at least quadruple the highest '
                    f'periodicity specified. The max periodicity specified in trig_seasonal is '
                    f'{max(trig_periodicities)}.')

        else:
            if len(stochastic_trig_seasonal) > 0:
                raise ValueError(
                    'No trigonometric seasonal components were specified, but a non-empty '
                    'stochastic profile was passed for trigonometric seasonality. If trigonometric '
                    'seasonal components are desired, specify the period for each '
                    'component via a tuple passed to the trig_seasonal argument.')

        # FINAL VALIDITY CHECKS
        if not isinstance(level, bool) or not isinstance(stochastic_level, bool):
            raise TypeError(
                'level and stochastic_level must be of boolean type.')

        if level and not stochastic_level and damped_level:
            raise ValueError(
                'stochastic_level must be true if level and damped_level are true.')

        if not isinstance(trend, bool) or not isinstance(stochastic_trend, bool):
            raise TypeError(
                'trend and stochastic_trend must be of boolean type.')

        if trend and not stochastic_trend and damped_trend:
            raise ValueError(
                'stochastic_trend must be true if trend and damped_trend are true.')

        if len(lag_seasonal) == 0 and len(dummy_seasonal) == 0 and len(trig_seasonal) == 0 and not level:
            raise ValueError(
                'At least a level or seasonal component must be specified.')

        if trend and not level:
            raise ValueError(
                'trend cannot be specified without a level component.')

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
                raise ValueError(
                    'lag_seasonal and trig_seasonal cannot have periodicities in common.')

            if len(set(lag_seasonal).intersection(set(dummy_seasonal))) > 0:
                raise ValueError(
                    'lag_seasonal and dummy_seasonal cannot have periodicities in common.')

            if len(set(dummy_seasonal).intersection(set(trig_periodicities))) > 0:
                raise ValueError('dummy_seasonal and trig_seasonal cannot have periodicities in common.')

        # ASSIGN CLASS ATTRIBUTES IF ALL VALIDITY CHECKS ARE PASSED
        # If the response has NaNs, replace with 0's
        if np.any(np.isnan(resp)):
            self.response_nan_indicator = np.isnan(resp) * 1.
            self.response_replace_nan = ao.replace_nan(resp)
            self.response_has_nan = True
        else:
            self.response_has_nan = False

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

    @property
    def num_first_obs_ignore(self) -> int:
        p = 1 + max((0,)
                    + self.lag_seasonal
                    + self.dummy_seasonal
                    + tuple(j[0] for j in self.trig_seasonal))
        return max(p, self.num_state_eqs)

    @staticmethod
    def trig_transition_matrix(freq: int) -> np.ndarray:
        real_part = np.array([[np.cos(freq), np.sin(freq)]])
        imaginary_part = np.array([[-np.sin(freq), np.cos(freq)]])
        return np.concatenate((real_part, imaginary_part), axis=0)

    def observation_matrix(self,
                           num_rows: int = 0) -> np.ndarray:
        """

        :param num_rows: Number of rows that matches the number of observations
        of the response. For a model specified without a regression component,
        this amounts to replication of the observation matrix (row vector) num_row times.
        However, if a time-varying regression component is specified, the observation matrix
        will have distinct rows.
        :return: ndarray with dimension (num_row, num_state_eqs). This represents the observation
        matrix for the response equation.
        """
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
        """
        State transition matrix for the state equations.
        :return: ndarray of dimension (num_state_eqs, num_state_eqs)
        """
        m = self.num_state_eqs
        T = np.zeros((m, m))

        i, j = 0, 0
        if self.level:
            T[i, j] = 1.

            i += 1
            j += 1

        if self.trend:
            T[i - 1, j] = 1.
            T[i, j] = 1.

            i += 1
            j += 1

        if len(self.lag_seasonal) > 0:
            for c, v in enumerate(self.lag_seasonal):
                T[i, j + v - 1] = 1.

                for k in range(1, v):
                    T[i + k, j + k - 1] = 1.

                i += v
                j += v

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
        """
        The state intercept for the state equations.
        * Note that this is not currently being used. It
        * is here as a placeholder for potential future implementation.
        :return: ndarray of dimension (num_state_eqs, 1)
        """
        m = self.num_state_eqs
        C = np.zeros((m, 1))

        return C

    @property
    def state_error_transformation_matrix(self) -> np.ndarray:
        """
        State error transformation matrix. This matrix tracks
        which state equations are stochastic and non-stochastic.
        :return: ndarray of dimension (num_state_eqs, num_stoch_states).
        """
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
        """
        Matrix used for computing posterior scale parameters for
        posterior state variances. In general, this is an identity
        matrix of dimension (num_stoch_states, num_stoch_states).
        However, if a stochastic trigonometric seasonality component
        is specified, then the sum of squared residuals for each
        trigonometric state equation is used to arrive at a single
        variance. In other words, the residuals across all state
        equations corresponding to a given stochastic trigonometric
        component are pooled with the assumption that errors
        across harmonics/frequencies are homoskedastic. This
        is a simplifying assumption that is commonly used.
        :return:
        """
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

    def _posterior_exists_check(self) -> None:
        if self.posterior is None:
            raise AttributeError("No posterior distribution was found. The fit() method must be called.")

        return

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
        """
        Iteration-0 Gibbs value for level component.
        Uses the first non-missing value of the response.
        :return: scalar
        """
        if self.level:
            first_y = self._first_value(self.response)[0]
            return first_y
        else:
            return

    def _gibbs_iter0_init_trend(self):
        """
        Iteration-0 Gibbs value for a trend component.
        Uses the first and last non-missing values of the response to
        compute an initial linear slope.
        :return: scalar
        """
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
        """
        Iteration-0 Gibbs values for a lag-seasonal component.
        A vector of 0's is used.
        :return: Zero-vector of length num_lag_season_state_eqs
        """
        if len(self.lag_seasonal) > 0:
            num_eqs = self.num_lag_season_state_eqs
            return np.zeros(num_eqs)
        else:
            return

    def _gibbs_iter0_init_dum_season(self):
        """
        Iteration-0 Gibbs values for a lag-seasonal component.
        A vector of 0's is used.
        :return: Zero-vector of length num_dum_season_state_eqs
        """
        if len(self.dummy_seasonal) > 0:
            num_eqs = self.num_dum_season_state_eqs
            return np.zeros(num_eqs)
        else:
            return

    def _gibbs_iter0_init_trig_season(self):
        """
        Iteration-0 Gibbs values for a lag-seasonal component.
        A vector of 0's is used.
        :return: Zero-vector of length num_trig_season_state_eqs
        """
        if len(self.trig_seasonal) > 0:
            num_eqs = self.num_trig_season_state_eqs
            return np.zeros(num_eqs)
        else:
            return

    @staticmethod
    def _ar_state_post_upd(smoothed_state: np.ndarray,
                           lag: tuple[int, ...],
                           mean_prior: np.ndarray,
                           precision_prior: np.ndarray,
                           state_err_var_post: np.ndarray,
                           state_eqn_index: list,
                           state_err_var_post_index: list):
        """
        This is a generic function for updating an AR(p) coefficient for a state variable that
        has an autoregressive structure. For example, if trend is specified with damping, then the
        trend state equation will take the form trend(t) = alpha * trend(t-1) + error(t), and the
        coefficient alpha will be updated in the same way coefficients would be updated in a
        standard regression context. The smoothed Kalman values of a state variable are required
        for posterior updating.

        :param smoothed_state: ndarray, float64. Array that represents that smoothed values for each state
        in an unobserved components model.

        :param lag: tuple of positive integers. An integer represents an autoregressive lag of the
        state variable.

        :param mean_prior: ndarray, float64. An element represents the mean of a Gaussian
        random variable. Row length must match the number of lags.

        :param precision_prior: ndarray, float64. An element represents the precision of a Gaussian
        random variable. Row length must match the number of lags.

        :param state_err_var_post: ndarray, float64. An element represents the posterior variance
        of a state variable's error.

        :param state_eqn_index: list of non-negative integers. An integer represents the row index of a
        state within an unobserved components model.

        :param state_err_var_post_index: list of non-negative integers. An integer represents the row index
        of a state's posterior error variance (i.e., a row index for state_err_var_post).


        :return: ndarray of dimension (L, 1), where L is the number of lags in the lag tuple. An
        element represents the posterior autoregressive coefficient for an autoregressive process of the
        form y(t) = rho * y(t - l) + error(t), where l is some lag.
        """

        ar_coeff_mean_post = np.empty((len(lag), 1))
        cov_post = np.empty((len(lag), 1))

        c = 0
        for i in lag:
            y = smoothed_state[:, state_eqn_index[c]][i:]
            y_lag = np.roll(smoothed_state[:, state_eqn_index[c]], i)[i:]
            prec_prior = precision_prior[c]

            ar_coeff_cov_post = ao.mat_inv(y_lag.T @ y_lag + prec_prior)
            ar_coeff_mean_post[c] = ar_coeff_cov_post @ (y_lag.T @ y + prec_prior @ mean_prior[c])
            cov_post[c] = state_err_var_post[state_err_var_post_index[c], 0] * ar_coeff_cov_post
            c += 1

        ar_coeff_post = dist.vec_norm(ar_coeff_mean_post, np.sqrt(cov_post))

        return ar_coeff_post

    def _design_matrix_svd(self):
        X = self.predictors
        n, k = X.shape
        if n >= k:
            _, s, X_SVD_Vt = np.linalg.svd(X, full_matrices=False)
            X_SVD_S = np.diag(s)
            X_SVD_StS = X_SVD_S ** 2
        else:
            _, s, X_SVD_Vt = np.linalg.svd(X, full_matrices=True)
            X_SVD_S = np.zeros((n, k))
            X_SVD_S[:n, :n] = np.diag(s)
            X_SVD_StS = X_SVD_S.T @ X_SVD_S

        return X_SVD_Vt, X_SVD_StS

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
                     zellner_prior_obs
                     ) -> ModelSetup:

        n = self.num_obs
        q = self.num_stoch_states
        var_y = np.nanvar(self.response, ddof=1)
        default_shape_prior = 0.01
        default_root_scale = 0.01 * np.sqrt(var_y)

        # Initialize outputs
        if q > 0:
            state_var_scale_prior = []
            state_var_shape_post = []
            gibbs_iter0_state_error_var = []
        else:
            # Can't assign to NoneType because Numba function expects an array type.
            state_var_shape_post = np.array([[]])
            state_var_scale_prior = np.array([[]])
            gibbs_iter0_state_error_covariance = np.empty((0, 0), dtype=np.float64)

        gibbs_iter0_init_state = []
        init_state_variances = []
        init_state_plus_values = []

        if not self.level or not self.damped_level:
            damped_level_coeff_mean_prior = None
            damped_level_coeff_prec_prior = None
            damped_level_coeff_cov_prior = None
            gibbs_iter0_damped_level_coeff = None

        if not self.trend or not self.damped_trend:
            damped_trend_coeff_mean_prior = None
            damped_trend_coeff_prec_prior = None
            damped_trend_coeff_cov_prior = None
            gibbs_iter0_damped_trend_coeff = None

        if not self.num_lag_season_state_eqs > 0 or not self.num_damped_lag_season > 0:
            damped_lag_season_coeff_mean_prior = None
            damped_lag_season_coeff_prec_prior = None
            damped_lag_season_coeff_cov_prior = None
            gibbs_iter0_damped_season_coeff = None

        if not self.has_predictors:
            reg_coeff_mean_prior = None
            reg_coeff_prec_prior = None
            reg_coeff_cov_prior = None
            reg_ninvg_coeff_cov_post = None
            reg_ninvg_coeff_prec_post = None
            gibbs_iter0_reg_coeff = None

        # Record the specification of the model.
        # Irregular component will always be a part of the model.
        components = {"Irregular": dict(
            params=["Irregular.Var"],
            start_state_eqn_index=None,
            end_state_eqn_index=None,
            stochastic=True,
            stochastic_index=None,
            damped=None,
            damped_transition_index=None
        )}

        # RESPONSE ERROR
        if response_var_shape_prior is None:
            response_var_shape_prior = default_shape_prior
        if response_var_scale_prior is None:
            response_var_scale_prior = default_root_scale ** 2

        response_var_shape_post = np.array([[response_var_shape_prior + 0.5 * n]])
        gibbs_iter0_response_error_variance = np.array([[response_var_scale_prior]])

        # Record the state specification of the state space model

        # LEVEL STATE
        j, s = 0, 0  # j indexes the state equation, and s indexes stochastic equations
        if self.level:
            level_params = []
            if self.stochastic_level:
                level_params.append("Level.Var")

                if level_var_shape_prior is None:
                    level_var_shape_prior = default_shape_prior

                if level_var_scale_prior is None:
                    level_var_scale_prior = default_root_scale ** 2

                state_var_shape_post.append(level_var_shape_prior + 0.5 * n)
                state_var_scale_prior.append(level_var_scale_prior)
                gibbs_iter0_state_error_var.append(level_var_scale_prior)
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
                level_params.append("Level.AR")

                if damped_level_coeff_mean_prior is None:
                    damped_level_coeff_mean_prior = np.array([[1.]])

                if damped_level_coeff_prec_prior is None:
                    damped_level_coeff_prec_prior = np.array([[1.]])

                damped_level_coeff_cov_prior = ao.mat_inv(damped_level_coeff_prec_prior)
                gibbs_iter0_damped_level_coeff = damped_level_coeff_mean_prior

                if abs(damped_level_coeff_mean_prior[0, 0]) >= 1:
                    init_state_variances.append(1e6)
                else:
                    init_state_variances.append(gibbs_iter0_state_error_var[stochastic_index] /
                                                (1. - gibbs_iter0_damped_level_coeff[0, 0] ** 2))

                init_state_plus_values.append(dist.vec_norm(0., np.sqrt(init_state_variances[j])))
                damped_index = (j, j)
            else:
                damped_index = ()
                init_state_variances.append(1e6)
                init_state_plus_values.append(0.)

            gibbs_iter0_init_state.append(self._gibbs_iter0_init_level())
            components["Level"] = dict(
                params=level_params,
                start_state_eqn_index=j,
                end_state_eqn_index=j + 1,
                stochastic=self.stochastic_level,
                stochastic_index=stochastic_index,
                damped=self.damped_level,
                damped_transition_index=damped_index
            )
            j += 1

        # TREND STATE
        if self.trend:
            trend_params = []
            if self.stochastic_trend:
                trend_params.append("Trend.Var")

                if trend_var_shape_prior is None:
                    trend_var_shape_prior = default_shape_prior

                if trend_var_scale_prior is None:
                    trend_var_scale_prior = (0.2 * default_root_scale) ** 2

                state_var_shape_post.append(trend_var_shape_prior + 0.5 * n)
                state_var_scale_prior.append(trend_var_scale_prior)
                gibbs_iter0_state_error_var.append(trend_var_scale_prior)

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
                trend_params.append("Trend.AR")

                if damped_trend_coeff_mean_prior is None:
                    damped_trend_coeff_mean_prior = np.array([[1.]])

                if damped_trend_coeff_prec_prior is None:
                    damped_trend_coeff_prec_prior = np.array([[1.]])

                damped_trend_coeff_cov_prior = ao.mat_inv(damped_trend_coeff_prec_prior)
                gibbs_iter0_damped_trend_coeff = damped_trend_coeff_mean_prior

                if abs(damped_trend_coeff_mean_prior[0, 0]) >= 1:
                    init_state_variances.append(1e6)
                else:
                    init_state_variances.append(gibbs_iter0_state_error_var[stochastic_index] /
                                                (1. - gibbs_iter0_damped_trend_coeff[0, 0] ** 2))

                init_state_plus_values.append(dist.vec_norm(0., np.sqrt(init_state_variances[j])))
                damped_index = (j, j)
            else:
                damped_index = ()
                init_state_variances.append(1e6)
                init_state_plus_values.append(0.)

            gibbs_iter0_init_state.append(self._gibbs_iter0_init_trend())
            components["Trend"] = dict(
                params=trend_params,
                start_state_eqn_index=j,
                end_state_eqn_index=j + 1,
                stochastic=self.stochastic_trend,
                stochastic_index=stochastic_index,
                damped=self.damped_trend,
                damped_transition_index=damped_index
            )
            j += 1

        # LAG SEASONAL STATE
        if len(self.lag_seasonal) > 0:
            if self.num_damped_lag_season > 0:
                if damped_lag_season_coeff_mean_prior is None:
                    damped_lag_season_coeff_mean_prior = np.ones((self.num_damped_lag_season, 1))

                if damped_lag_season_coeff_prec_prior is None:
                    damped_lag_season_coeff_prec_prior = np.ones((self.num_damped_lag_season, 1))

                damped_lag_season_coeff_cov_prior = damped_lag_season_coeff_prec_prior ** (-1)
                gibbs_iter0_damped_season_coeff = damped_lag_season_coeff_mean_prior

            i = j
            d = 0  # indexes damped periodicities
            for c, v in enumerate(self.lag_seasonal):
                lag_season_params = []

                if self.stochastic_lag_seasonal[c]:
                    lag_season_params.append(f"Lag-Seasonal.{v}.Var")

                    if lag_season_var_shape_prior is None:
                        shape_prior = default_shape_prior
                    else:
                        shape_prior = lag_season_var_shape_prior[c]
                    state_var_shape_post.append(shape_prior + 0.5 * n)

                    if lag_season_var_scale_prior is None:
                        scale_prior = default_root_scale ** 2
                    else:
                        scale_prior = lag_season_var_scale_prior[c]
                    state_var_scale_prior.append(scale_prior)

                    gibbs_iter0_state_error_var.append(scale_prior)

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
                    lag_season_params.append(f"Lag-Seasonal.{v}.AR")

                    if abs(damped_lag_season_coeff_mean_prior[d, 0]) >= 1:
                        init_state_variances.append(1e6)
                    else:
                        init_state_variances.append(gibbs_iter0_state_error_var[stochastic_index] /
                                                    (1. - gibbs_iter0_damped_season_coeff[d, 0] ** 2))

                    init_state_plus_values.append(dist.vec_norm(0., np.sqrt(init_state_variances[j])))
                    damped_index = (i, i + v - 1)
                    damped_ar_coeff_col_index = d
                    d += 1
                else:
                    damped_index = ()
                    damped_ar_coeff_col_index = None
                    init_state_variances.append(1e6)
                    init_state_plus_values.append(0.)

                for k in range(v - 1):
                    init_state_variances.append(1e6)
                    init_state_plus_values.append(0.)

                components[f"Lag-Seasonal.{v}"] = dict(
                    params=lag_season_params,
                    start_state_eqn_index=i,
                    end_state_eqn_index=i + v,
                    stochastic=self.stochastic_lag_seasonal[c],
                    stochastic_index=stochastic_index,
                    damped=self.damped_lag_seasonal[c],
                    damped_transition_index=damped_index,
                    damped_ar_coeff_col_index=damped_ar_coeff_col_index
                )
                i += v

            for k in self._gibbs_iter0_init_lag_season():
                gibbs_iter0_init_state.append(k)

            j += self.num_lag_season_state_eqs

        # DUMMY SEASONAL STATE
        if len(self.dummy_seasonal) > 0:
            i = j
            for c, v in enumerate(self.dummy_seasonal):
                dum_season_params = []

                if self.stochastic_dummy_seasonal[c]:
                    dum_season_params.append(f"Dummy-Seasonal.{v}.Var")

                    if dum_season_var_shape_prior is None:
                        shape_prior = default_shape_prior
                    else:
                        shape_prior = dum_season_var_shape_prior[c]
                    state_var_shape_post.append(shape_prior + 0.5 * n)

                    if dum_season_var_scale_prior is None:
                        scale_prior = default_root_scale ** 2
                    else:
                        scale_prior = dum_season_var_scale_prior[c]
                    state_var_scale_prior.append(scale_prior)

                    gibbs_iter0_state_error_var.append(scale_prior)

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

                components[f"Dummy-Seasonal.{v}"] = dict(
                    params=dum_season_params,
                    start_state_eqn_index=i,
                    end_state_eqn_index=i + (v - 1),
                    stochastic=self.stochastic_dummy_seasonal[c],
                    stochastic_index=stochastic_index,
                    damped=None,
                    damped_transition_index=None
                )
                i += v - 1

            for k in self._gibbs_iter0_init_dum_season():
                gibbs_iter0_init_state.append(k)

            j += self.num_dum_season_state_eqs

        # TRIGONOMETRIC SEASONAL STATE
        if len(self.trig_seasonal) > 0:
            i = j
            for c, v in enumerate(self.trig_seasonal):
                trig_season_params = []
                period, num_harmonics = v

                if period / num_harmonics == 2:
                    num_eqs = 2 * num_harmonics - 1
                else:
                    num_eqs = 2 * num_harmonics

                if self.stochastic_trig_seasonal[c]:
                    trig_season_params.append(f"Trig-Seasonal.{period}.{num_harmonics}.Var")

                    if trig_season_var_shape_prior is None:
                        shape_prior = default_shape_prior
                    else:
                        shape_prior = trig_season_var_shape_prior[c]
                    state_var_shape_post.append(shape_prior + 0.5 * n * num_eqs)

                    if trig_season_var_scale_prior is None:
                        scale_prior = default_root_scale ** 2 / num_eqs
                    else:
                        scale_prior = trig_season_var_scale_prior[c] / num_eqs
                    state_var_scale_prior.append(scale_prior)

                    for k in range(num_eqs):
                        gibbs_iter0_state_error_var.append(scale_prior)

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

                components[f"Trig-Seasonal"
                           f".{period}.{num_harmonics}"] = dict(
                    params=trig_season_params,
                    start_state_eqn_index=i,
                    end_state_eqn_index=i + num_eqs,
                    stochastic=self.stochastic_trig_seasonal[c],
                    stochastic_index=stochastic_index,
                    damped=None,
                    damped_transition_index=None
                )
                i += num_eqs

            for k in self._gibbs_iter0_init_trig_season():
                gibbs_iter0_init_state.append(k)

            j += self.num_trig_season_state_eqs

        # REGRESSION
        if self.has_predictors:
            components["Regression"] = dict(
                params=[f"Coeff.{i}" for i in self.predictors_names],
                start_state_eqn_index=j,
                end_state_eqn_index=j + 1,
                stochastic=False,
                stochastic_index=None,
                damped=None,
                damped_transition_index=None
            )

            y = self.response
            X = self.predictors
            num_obs, num_pred = X.shape
            Vt, StS = self._design_matrix_svd()
            XtX = Vt.T @ StS @ Vt

            # Get a record of all periodicities and harmonics
            season_specs = []
            if len(self.dummy_seasonal) > 0:
                for c, j in enumerate(self.dummy_seasonal):
                    season_specs.append(((j, int(j / 2)),
                                         self.stochastic_dummy_seasonal[c]))
            if len(self.trig_seasonal) > 0:
                for c, j in enumerate(self.trig_seasonal):
                    p, h = j
                    if h == 0:
                        h = int(j / 2)
                    season_specs.append(((p, h),
                                         self.stochastic_trig_seasonal[c]))
            if len(self.lag_seasonal) > 0:
                for c, j in enumerate(self.lag_seasonal):
                    season_specs.append(((j, int(j / 2)),
                                         self.stochastic_lag_seasonal[c]))

            # Convert seasonal specs to appropriate arguments
            # for statsmodels UnobservedComponents module.
            if len(season_specs) > 0:
                uc_season_spec_args = []
                uc_season_stoch_args = []
                for j in season_specs:
                    p, h = j[0]
                    stoch = j[1]
                    uc_season_spec_args.append({'period': p, 'harmonics': h})
                    uc_season_stoch_args.append(stoch)
            else:
                uc_season_spec_args = None
                uc_season_stoch_args = None

            # Get initial Gibbs regression coefficient values using
            # statsmodels UnobservedComponents module.
            try:
                uc_mod = UC(
                    endog=y,
                    exog=X,
                    level=self.level,
                    stochastic_level=self.stochastic_level,
                    trend=self.trend,
                    stochastic_trend=self.stochastic_trend,
                    freq_seasonal=uc_season_spec_args,
                    stochastic_freq_seasonal=uc_season_stoch_args,
                    irregular=True
                )
                uc_fit = uc_mod.fit(disp=False, method='powell', maxiter=10)
                uc_fit = uc_mod.fit(disp=False, start_params=uc_fit.params)
                params = pd.Series(uc_fit.params, index=uc_fit.param_names)
                gibbs_iter0_reg_coeff = (
                    np.array(params[[j for j in params.index if j.split('.')[0] == 'beta']])
                    .reshape(-1, 1)
                )
            except Exception as e:
                print(e.args[0])
                print("An attempt was made to establish initial Gibbs regression "
                      "coefficient values using statsmodels' UnobservedComponents "
                      "module, but failed. The prior for the mean regression "
                      "coefficients will be used instead."
                      )
                gibbs_iter0_reg_coeff = None

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
                reg_coeff_mean_prior = np.zeros((num_pred, 1))

            if reg_coeff_prec_prior is None:
                reg_coeff_prec_prior = (zellner_prior_obs / n
                                        * (0.5 * XtX
                                           + 0.5 * np.diag(np.diag(XtX))
                                           )
                                        )
                reg_coeff_cov_prior = ao.mat_inv(reg_coeff_prec_prior)
            else:
                reg_coeff_cov_prior = ao.mat_inv(reg_coeff_prec_prior)

            reg_ninvg_coeff_prec_post = Vt.T @ (StS + Vt @ reg_coeff_prec_prior @ Vt.T) @ Vt
            reg_ninvg_coeff_cov_post = Vt.T @ ao.mat_inv(StS + Vt @ reg_coeff_prec_prior @ Vt.T) @ Vt

        if q > 0:
            state_var_shape_post = np.vstack(state_var_shape_post)
            state_var_scale_prior = np.vstack(state_var_scale_prior)
            gibbs_iter0_state_error_covariance = np.diag(gibbs_iter0_state_error_var)

        gibbs_iter0_init_state = np.vstack(gibbs_iter0_init_state)
        init_state_plus_values = np.vstack(init_state_plus_values)
        init_state_covariance = np.diag(init_state_variances)

        # Get list of all parameters
        for c in components:
            self.parameters += components[c]['params']

        self.model_setup = ModelSetup(
            components,
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
            reg_ninvg_coeff_cov_post,
            reg_ninvg_coeff_prec_post,
            zellner_prior_obs,
            gibbs_iter0_reg_coeff
        )

        return self.model_setup

    def posterior_dict(self,
                       burn: int = 0
                       ) -> dict:
        """

        :param burn: non-negative integer. Represents how many of the first posterior samples to
        ignore for computing statistics like the mean and variance of a parameter. Default value is 0.
        :return:
        """

        if isinstance(burn, int) and burn >= 0:
            pass
        else:
            raise ValueError('burn must be a non-negative integer.')

        self._posterior_exists_check()
        posterior = self.posterior

        components = self.model_setup.components
        resp_err_var = posterior.response_error_variance[burn:]
        state_err_cov = posterior.state_error_covariance[burn:]
        params = self.parameters

        if self.has_predictors:
            reg_coeff = self.posterior.regression_coefficients[burn:, :, 0]

        post_dict = {}
        for p in params:
            if p == 'Irregular.Var':
                post_dict[p] = resp_err_var[:, 0, 0]

            elif p == 'Level.Var':
                c = components['Level']
                idx = c['stochastic_index']
                post_dict[p] = state_err_cov[:, idx, idx]

            elif p == 'Level.AR':
                post_dict[p] = self.posterior.damped_level_coefficient[burn:, 0, 0]

            elif p == 'Trend.Var':
                c = components['Trend']
                idx = c['stochastic_index']
                post_dict[p] = state_err_cov[:, idx, idx]

            elif p == 'Trend.AR':
                post_dict[p] = self.posterior.damped_trend_coefficient[burn:, 0, 0]

            elif 'Lag-Seasonal' in p and '.Var' in p:
                c = components[p.replace('.Var', '')]
                idx = c['stochastic_index']
                post_dict[p] = state_err_cov[:, idx, idx]

            elif 'Lag-Seasonal' in p and '.AR' in p:
                c = components[p.replace('.AR', '')]
                idx = c['damped_ar_coeff_col_index']
                post_dict[p] = self.posterior.damped_season_coefficients[burn:, idx, 0]

            elif 'Dummy-Seasonal' in p and '.Var' in p:
                c = components[p.replace('.Var', '')]
                idx = c['stochastic_index']
                post_dict[p] = state_err_cov[:, idx, idx]

            elif 'Trig-Seasonal' in p and '.Var' in p:
                c = components[p.replace('.Var', '')]
                idx = c['stochastic_index']
                post_dict[p] = state_err_cov[:, idx, idx]

            elif 'Coeff.' in p:
                post_dict[p] = reg_coeff[:, self.predictors_names.index(p.replace('Coeff.', ''))]

        return post_dict

    def _high_variance(self, burn: int = 0) -> dict:
        post_dict = self.posterior_dict(burn=burn)
        high_var = np.nanvar(self.response, ddof=1)

        if self.num_stoch_states > 0:
            hv = {}
            for k, v in post_dict.items():
                if ".Var" in k:
                    high_var_index = np.argwhere((v > high_var))
                    pct_high_var = high_var_index.size / v.size
                    hv[k] = dict(high_var_index=high_var_index,
                                 pct_high_var=pct_high_var)

            return hv

    def sample(self,
               num_samp: int,
               response_var_shape_prior: Union[int, float] = None,
               response_var_scale_prior: Union[int, float] = None,
               level_var_shape_prior: Union[int, float] = None,
               level_var_scale_prior: Union[int, float] = None,
               damped_level_coeff_mean_prior: Union[np.ndarray, list, tuple] = None,
               damped_level_coeff_prec_prior: Union[np.ndarray, list, tuple] = None,
               trend_var_shape_prior: Union[int, float] = None,
               trend_var_scale_prior: Union[int, float] = None,
               damped_trend_coeff_mean_prior: Union[np.ndarray, list, tuple] = None,
               damped_trend_coeff_prec_prior: Union[np.ndarray, list, tuple] = None,
               lag_season_var_shape_prior: tuple[Union[int, float], ...] = None,
               lag_season_var_scale_prior: tuple[Union[int, float], ...] = None,
               damped_lag_season_coeff_mean_prior: Union[np.ndarray, list, tuple] = None,
               damped_lag_season_coeff_prec_prior: Union[np.ndarray, list, tuple] = None,
               dum_season_var_shape_prior: tuple[Union[int, float], ...] = None,
               dum_season_var_scale_prior: tuple[Union[int, float], ...] = None,
               trig_season_var_shape_prior: tuple[Union[int, float], ...] = None,
               trig_season_var_scale_prior: tuple[Union[int, float], ...] = None,
               reg_coeff_mean_prior: Union[np.ndarray, list, tuple] = None,
               reg_coeff_prec_prior: Union[np.ndarray, list, tuple] = None,
               zellner_prior_obs: Union[int, float] = None,
               upper_var_limit: Union[int, float] = None,
               max_samp_iter_factor: Union[int, float] = None
               ) -> Posterior:

        """
        Posterior distributions for all parameters and states.

        :param num_samp: integer > 0. Specifies the number of posterior samples to draw.

        :param response_var_shape_prior: int, float > 0. Specifies the inverse-Gamma shape prior for the
        response error variance. Default is 0.01.

        :param response_var_scale_prior: int, float > 0. Specifies the inverse-Gamma scale prior for the
        response error variance. Default is (0.01 * std(response))^2.

        :param level_var_shape_prior: int, float > 0. Specifies the inverse-Gamma shape prior for the
        level state equation error variance. Default is 0.01.

        :param level_var_scale_prior: int, float > 0. Specifies the inverse-Gamma scale prior for the
        level state equation error variance. (5 * 0.01 * std(response))^2.

        :param damped_level_coeff_mean_prior: Numpy array, list, or tuple. Specifies the prior
        mean for the coefficient governing the level's AR(1) process without drift. Default is [[1.]].

        :param damped_level_coeff_prec_prior: Numpy array, list, or tuple. Specifies the prior
        precision matrix for the coefficient governing the level's an AR(1) process without drift.
        Default is [[1.]].

        :param trend_var_shape_prior: int, float > 0. Specifies the inverse-Gamma shape prior for the
        trend state equation error variance. Default is 0.01.

        :param trend_var_scale_prior: int, float > 0. Specifies the inverse-Gamma scale prior for the
        trend state equation error variance. Default is (0.25 * 0.01 * std(response))^2.

        :param damped_trend_coeff_mean_prior: Numpy array, list, or tuple. Specifies the prior
        mean for the coefficient governing the trend's AR(1) process without drift. Default is [[1.]].

        :param damped_trend_coeff_prec_prior: Numpy, list, or tuple. Specifies the prior
        precision matrix for the coefficient governing the trend's an AR(1) process without drift.
        Default is [[1.]].

        :param lag_season_var_shape_prior: tuple of int, float > 0 with s elements, where s is the number of
        stochastic periodicities. Specifies the inverse-Gamma shape priors for each periodicity in lag_seasonal.
        Default is 0.01 for each periodicity.

        :param lag_season_var_scale_prior: tuple of int, float > 0 with s elements, where s is the number of
        stochastic periodicities. Specifies the inverse-Gamma scale priors for each periodicity in lag_seasonal.
        Default is (10 * 0.01 * std(response))^2 for each periodicity.

        :param damped_lag_season_coeff_mean_prior: Numpy array, list, or tuple with s elements, where s is the
        number of stochastic periodicities with damping specified. Specifies the prior mean for the coefficient
        governing a lag_seasonal AR(1) process without drift. Default is [[1.]] for each damped, stochastic periodicity.

        :param damped_lag_season_coeff_prec_prior: Numpy array, list, or tuple with s elements, where s is the
        number of stochastic periodicities with damping specified. Specifies the prior precision matrix for the
        coefficient governing a lag_seasonal AR(1) process without drift. Default is [[1.]] for each damped periodicity.

        :param dum_season_var_shape_prior: tuple of int, float > 0 with s elements, where s is the number of
        stochastic periodicities. Specifies the inverse-Gamma shape priors for each periodicity in dummy_seasonal.
        Default is 0.01 for each periodicity.

        :param dum_season_var_scale_prior: tuple of int, float > 0 with s elements, where s is the number of
        stochastic periodicities. Specifies the inverse-Gamma scale priors for each periodicity in dummy_seasonal.
        Default is (10 * 0.01 * std(response))^2 for each periodicity.

        :param trig_season_var_shape_prior: tuple of int, float > 0 with s elements, where s is the number of
        stochastic periodicities. Specifies the inverse-Gamma shape priors for each periodicity in trig_seasonal.
        For example, if trig_seasonal = ((12, 3), (10, 2)) and stochastic_trig_seasonal = (True, False), only one
        shape prior needs to be specified, namely for periodicity 12.
        Default is 0.01 for each periodicity.

        :param trig_season_var_scale_prior: tuple of int, float > 0 with s elements, where s is the number of
        stochastic periodicities. Specifies the inverse-Gamma scale priors for each periodicity in trig_seasonal.
        For example, if trig_seasonal = ((12, 3), (10, 2)) and stochastic_trig_seasonal = (True, False), only two
        scale priors need to be specified, namely periodicity 12.
        Default is (10 * 0.01 * std(response))^2 / # of state equations for each periodicity. Note that whatever
        value is passed to trig_season_var_scale_prior is automatically scaled by the number of state
        equations implied by the periodicity and number of harmonics.

        :param reg_coeff_mean_prior: Numpy array, list, or tuple with k elements, where k is the number of predictors.
        If predictors are specified without a mean prior, a k-dimensional zero vector will be assumed.

        :param reg_coeff_prec_prior: Numpy array, list, or tuple with (k, k) elements, where k is the number of
        predictors. If predictors are specified without a precision prior, Zellner's g-prior will be enforced.
        Specifically, 1 / g * (w * dot(X.T, X) + (1 - w) * diag(dot(X.T, X))), where g = n / prior_obs, prior_obs
        is the number of prior observations given to the regression coefficient mean prior (i.e., it controls how
        much weight is given to the mean prior), n is the number of observations, X is the design matrix, and
        diag(dot(X.T, X)) is a diagonal matrix with the diagonal elements matching those of dot(X.T, X). The
        addition of the diagonal matrix to dot(X.T, X) is to guard against singularity (i.e., a design matrix
        that is not full rank). The weighting controlled by w is set to 0.5.

        :param zellner_prior_obs: int, float > 0. Relevant only if no regression precision matrix is provided.
        It controls how precise one believes their priors are for the regression coefficients, assuming no regression
        precision matrix is provided. Default value is 1e-6, which gives little weight to the regression coefficient
        mean prior. This should approximate maximum likelihood estimation.

        :param upper_var_limit: int of float > 0. This sets an acceptable upper bound on sampled variances (i.e.,
        response error variance and stochastic state error variances). By default, this value is set to the sample
        variance of the response variable.

        :param max_samp_iter_factor: int or float > 0. This factor is multiplied with num_samp to define an
        acceptable maximum number of sampling iterations before the sampling routine raises an exception.
        The maximum number of sampling iterations interacts with the acceptable upper bound on sampled variances.
        For example, suppose num_samp = 5000, max_samp_iter_factor = 3, and upper_var_limit = SampleVariance(response).
        If any of the model variances exceed upper_var_limit, a new draw will be made for the set of variances. It
        is possible, however, that the sampler will stay in a region of the parameter space that does not satisfy
        the upper bound on variances. Thus, after max_samp_iter = max_samp_iter_factor * num_samp = 15000 iterations,
        the sampler will raise an exception. Setting the upper bound on variances to a very high number, such as
        100 * SampleVariance(response), will likely result in the sampler not raising an exception, but caution should
        be taken if parametric inference matters. See the class attribute 'high_posterior_variance' for a summary
        of each of the model's stochastic variances in terms of "high" variance, where high is anything that exceeds
        the sample variance of the response. Default value is 2.

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

        if isinstance(num_samp, int) and num_samp > 0:
            pass
        else:
            raise ValueError(
                'num_samp must be a strictly positive integer.')

        # Response prior check
        if response_var_shape_prior is not None:
            if not isinstance(response_var_shape_prior, (int, float)):
                raise TypeError(
                    'response_var_shape_prior must be a strictly positive integer or float.')

            response_var_shape_prior = float(response_var_shape_prior)
            if np.isnan(response_var_shape_prior):
                raise ValueError('response_var_shape_prior cannot be NaN.')
            if np.isinf(response_var_shape_prior):
                raise ValueError('response_var_shape_prior cannot be Inf/-Inf.')
            if not response_var_shape_prior > 0:
                raise ValueError('response_var_shape_prior must be strictly positive.')

        if response_var_scale_prior is not None:
            if not isinstance(response_var_scale_prior, (int, float)):
                raise TypeError(
                    'response_var_scale_prior must be a strictly positive integer or float.')

            response_var_scale_prior = float(response_var_scale_prior)
            if np.isnan(response_var_scale_prior):
                raise ValueError('response_var_scale_prior cannot be NaN.')
            if np.isinf(response_var_scale_prior):
                raise ValueError('response_var_scale_prior cannot be Inf/-Inf.')
            if not response_var_scale_prior > 0:
                raise ValueError('response_var_scale_prior must be strictly positive.')

        # Level prior check
        if self.level and self.stochastic_level:
            if level_var_shape_prior is not None:
                if not isinstance(level_var_shape_prior, (int, float)):
                    raise TypeError(
                        'level_var_shape_prior must be a strictly positive integer or float.')

                level_var_shape_prior = float(level_var_shape_prior)
                if np.isnan(level_var_shape_prior):
                    raise ValueError('level_var_shape_prior cannot be NaN.')
                if np.isinf(level_var_shape_prior):
                    raise ValueError('level_var_shape_prior cannot be Inf/-Inf.')
                if not level_var_shape_prior > 0:
                    raise ValueError('level_var_shape_prior must be a strictly positive.')

            if level_var_scale_prior is not None:
                if not isinstance(level_var_scale_prior, (int, float)):
                    raise TypeError(
                        'level_var_scale_prior must be a strictly positive integer or float.')

                level_var_scale_prior = float(level_var_scale_prior)
                if np.isnan(level_var_scale_prior):
                    raise ValueError('level_var_scale_prior cannot be NaN.')
                if np.isinf(level_var_scale_prior):
                    raise ValueError('level_var_scale_prior cannot be Inf/-Inf.')
                if not level_var_scale_prior > 0:
                    raise ValueError('level_var_scale_prior must be strictly positive.')

            # Damped level prior check
            if self.damped_level:
                if damped_level_coeff_mean_prior is not None:
                    if not isinstance(damped_level_coeff_mean_prior, (np.ndarray, list, tuple)):
                        raise TypeError(
                            'damped_level_coeff_mean_prior must be a Numpy array, list, or tuple.')

                    if isinstance(damped_level_coeff_mean_prior, (list, tuple)):
                        damped_level_coeff_mean_prior = (np.asarray(damped_level_coeff_mean_prior,
                                                                    dtype=np.float64))
                    else:
                        damped_level_coeff_mean_prior = damped_level_coeff_mean_prior.astype(float)

                    if damped_level_coeff_mean_prior.ndim not in (1, 2):
                        raise ValueError('damped_level_coeff_mean_prior must have dimension 1 or 2.')
                    elif damped_level_coeff_mean_prior.ndim == 1:
                        damped_level_coeff_mean_prior = damped_level_coeff_mean_prior.reshape(1, 1)
                    else:
                        pass

                    if not damped_level_coeff_mean_prior.shape == (1, 1):
                        raise ValueError('damped_level_coeff_mean_prior must have shape (1, 1).')
                    if np.any(np.isnan(damped_level_coeff_mean_prior)):
                        raise ValueError('damped_level_coeff_mean_prior cannot have NaN values.')
                    if np.any(np.isinf(damped_level_coeff_mean_prior)):
                        raise ValueError('damped_level_coeff_mean_prior cannot have Inf/-Inf values.')
                    if abs(damped_level_coeff_mean_prior[0, 0]) > 1:
                        warnings.warn(
                            'The mean damped level coefficient is greater than 1 in absolute value, '
                            'which implies an explosive process. Note that an explosive process '
                            'can be stationary, but it implies that the future is needed to '
                            'predict the past.')

                if damped_level_coeff_prec_prior is not None:
                    if not isinstance(damped_level_coeff_prec_prior, (np.ndarray, list, tuple)):
                        raise TypeError(
                            'damped_level_coeff_prec_prior must be a Numpy array, list, or tuple.')

                    if isinstance(damped_level_coeff_prec_prior, (list, tuple)):
                        damped_level_coeff_prec_prior = (np.asarray(damped_level_coeff_prec_prior,
                                                                    dtype=np.float64))
                    else:
                        damped_level_coeff_prec_prior = damped_level_coeff_prec_prior.astype(float)

                    if damped_level_coeff_prec_prior.ndim not in (1, 2):
                        raise ValueError('damped_level_coeff_prec_prior must have dimension 1 or 2.')
                    elif damped_level_coeff_prec_prior.ndim == 1:
                        damped_level_coeff_prec_prior = damped_level_coeff_prec_prior.reshape(1, 1)
                    else:
                        pass

                    if not damped_level_coeff_prec_prior.shape == (1, 1):
                        raise ValueError('damped_level_coeff_prec_prior must have shape (1, 1).')
                    if np.any(np.isnan(damped_level_coeff_prec_prior)):
                        raise ValueError('damped_level_coeff_prec_prior cannot have NaN values.')
                    if np.any(np.isinf(damped_level_coeff_prec_prior)):
                        raise ValueError('damped_level_coeff_prec_prior cannot have Inf/-Inf values.')
                    # No need to do symmetric/positive definite checks since the matrix is 1x1

        # Trend prior check
        if self.trend and self.stochastic_trend:
            if trend_var_shape_prior is not None:
                if not isinstance(trend_var_shape_prior, (int, float)):
                    raise TypeError('trend_var_shape_prior must be a strictly positive integer or float.')

                trend_var_shape_prior = float(trend_var_shape_prior)
                if np.isnan(trend_var_shape_prior):
                    raise ValueError('trend_var_shape_prior cannot be NaN.')
                if np.isinf(trend_var_shape_prior):
                    raise ValueError('trend_var_shape_prior cannot be Inf/-Inf.')
                if not trend_var_shape_prior > 0:
                    raise ValueError('trend_var_shape_prior must be strictly positive.')

            if trend_var_scale_prior is not None:
                if not isinstance(trend_var_scale_prior, (int, float)):
                    raise TypeError('trend_var_scale_prior must be a strictly positive integer or float.')

                trend_var_scale_prior = float(trend_var_scale_prior)
                if np.isnan(trend_var_scale_prior):
                    raise ValueError('trend_var_scale_prior cannot be NaN.')
                if np.isinf(trend_var_scale_prior):
                    raise ValueError('trend_var_scale_prior cannot be Inf/-Inf.')
                if not trend_var_scale_prior > 0:
                    raise ValueError('trend_var_scale_prior must be a strictly positive.')

            # Damped trend prior check
            if self.damped_trend:
                if damped_trend_coeff_mean_prior is not None:
                    if not isinstance(damped_trend_coeff_mean_prior, (np.ndarray, list, tuple)):
                        raise TypeError('damped_trend_coeff_mean_prior must be a Numpy array, list, or tuple.')

                    if isinstance(damped_trend_coeff_mean_prior, (list, tuple)):
                        damped_trend_coeff_mean_prior = (np.asarray(damped_trend_coeff_mean_prior,
                                                                    dtype=np.float64))
                    else:
                        damped_trend_coeff_mean_prior = damped_trend_coeff_mean_prior.astype(float)

                    if damped_trend_coeff_mean_prior.ndim not in (1, 2):
                        raise ValueError('damped_trend_coeff_mean_prior must have dimension 1 or 2.')
                    elif damped_trend_coeff_mean_prior.ndim == 1:
                        damped_trend_coeff_mean_prior = damped_trend_coeff_mean_prior.reshape(1, 1)
                    else:
                        pass

                    if not damped_trend_coeff_mean_prior.shape == (1, 1):
                        raise ValueError('damped_trend_coeff_mean_prior must have shape (1, 1).')
                    if np.any(np.isnan(damped_trend_coeff_mean_prior)):
                        raise ValueError('damped_trend_coeff_mean_prior cannot have NaN values.')
                    if np.any(np.isinf(damped_trend_coeff_mean_prior)):
                        raise ValueError('damped_trend_coeff_mean_prior cannot have Inf/-Inf values.')
                    if abs(damped_trend_coeff_mean_prior[0, 0]) > 1:
                        warnings.warn(
                            'The mean damped trend coefficient is greater than 1 in absolute value, '
                            'which implies an explosive process. Note that an explosive process '
                            'can be stationary, but it implies that the future is needed to '
                            'predict the past.')

                if damped_trend_coeff_prec_prior is not None:
                    if not isinstance(damped_trend_coeff_prec_prior, (np.ndarray, list, tuple)):
                        raise TypeError(
                            'damped_trend_coeff_prec_prior must be a Numpy array, list, or tuple.')

                    if isinstance(damped_trend_coeff_prec_prior, (list, tuple)):
                        damped_trend_coeff_prec_prior = (np.asarray(damped_trend_coeff_prec_prior,
                                                                    dtype=np.float64))
                    else:
                        damped_trend_coeff_prec_prior = damped_trend_coeff_prec_prior.astype(float)

                    if damped_trend_coeff_prec_prior.ndim not in (1, 2):
                        raise ValueError('damped_trend_coeff_prec_prior must have dimension 1 or 2.')
                    elif damped_trend_coeff_prec_prior.ndim == 1:
                        damped_trend_coeff_prec_prior = damped_trend_coeff_prec_prior.reshape(1, 1)
                    else:
                        pass

                    if not damped_trend_coeff_prec_prior.shape == (1, 1):
                        raise ValueError('damped_trend_coeff_prec_prior must have shape (1, 1).')
                    if np.any(np.isnan(damped_trend_coeff_prec_prior)):
                        raise ValueError('damped_trend_coeff_prec_prior cannot have NaN values.')
                    if np.any(np.isinf(damped_trend_coeff_prec_prior)):
                        raise ValueError('damped_trend_coeff_prec_prior cannot have Inf/-Inf values.')
                    # No need to do symmetric/positive definite checks since the matrix is 1x1

        # Lag seasonal prior check
        if len(self.lag_seasonal) > 0 and sum(self.stochastic_lag_seasonal) > 0:
            if lag_season_var_shape_prior is not None:
                if not isinstance(lag_season_var_shape_prior, tuple):
                    raise TypeError(
                        'lag_season_var_shape_prior must be a tuple '
                        'to accommodate potentially multiple seasonality.')
                if len(lag_season_var_shape_prior) != sum(self.stochastic_lag_seasonal):
                    raise ValueError(
                        'The number of elements in lag_season_var_shape_prior must match the '
                        'number of stochastic periodicities in lag_seasonal. That is, for each '
                        'stochastic periodicity in lag_seasonal, there must be a corresponding '
                        'shape prior.')
                if not all(isinstance(i, (int, float)) for i in lag_season_var_shape_prior):
                    raise TypeError(
                        'All values in lag_season_var_shape_prior must be of type integer or float.')

                lag_season_var_shape_prior = tuple(float(i) for i in lag_season_var_shape_prior)
                if any(np.isnan(i) for i in lag_season_var_shape_prior):
                    raise ValueError('No values in lag_season_var_shape_prior can be NaN.')
                if any(np.isinf(i) for i in lag_season_var_shape_prior):
                    raise ValueError('No values in lag_season_var_shape_prior can be Inf/-Inf.')
                if not all(i > 0 for i in lag_season_var_shape_prior):
                    raise ValueError('All values in lag_season_var_shape_prior must be strictly positive.')

            if lag_season_var_scale_prior is not None:
                if not isinstance(lag_season_var_scale_prior, tuple):
                    raise TypeError(
                        'lag_season_var_scale_prior must be a tuple '
                        'to accommodate potentially multiple seasonality.')
                if len(lag_season_var_scale_prior) != sum(self.stochastic_lag_seasonal):
                    raise ValueError(
                        'The number of elements in lag_season_var_scale_prior must match the '
                        'number of stochastic periodicities in lag_seasonal. That is, for each '
                        'stochastic periodicity in lag_seasonal, there must be a corresponding '
                        'scale prior.')
                if not all(isinstance(i, (int, float)) for i in lag_season_var_scale_prior):
                    raise TypeError(
                        'All values in lag_season_var_scale_prior must be of type integer or float.')

                lag_season_var_scale_prior = tuple(float(i) for i in lag_season_var_scale_prior)
                if any(np.isnan(i) for i in lag_season_var_scale_prior):
                    raise ValueError('No values in lag_season_var_scale_prior can be NaN.')
                if any(np.isinf(i) for i in lag_season_var_scale_prior):
                    raise ValueError('No values in lag_season_var_scale_prior can be Inf/-Inf.')
                if not all(i > 0 for i in lag_season_var_scale_prior):
                    raise ValueError(
                        'All values in lag_season_var_scale_prior must be strictly positive.')

            # Damped lag seasonal prior check
            if self.num_damped_lag_season > 0:
                if damped_lag_season_coeff_mean_prior is not None:
                    if not isinstance(damped_lag_season_coeff_mean_prior, (np.ndarray, list, tuple)):
                        raise TypeError(
                            'damped_lag_season_coeff_mean_prior must be a Numpy array, list, or tuple.')

                    if isinstance(damped_lag_season_coeff_mean_prior, (list, tuple)):
                        damped_lag_season_coeff_mean_prior = (np.asarray(damped_lag_season_coeff_mean_prior,
                                                                         dtype=np.float64))
                    else:
                        damped_lag_season_coeff_mean_prior = damped_lag_season_coeff_mean_prior.astype(float)

                    if damped_lag_season_coeff_mean_prior.ndim not in (1, 2):
                        raise ValueError('damped_lag_season_coeff_mean_prior must have dimension 1 or 2.')
                    elif damped_lag_season_coeff_mean_prior.ndim == 1:
                        damped_lag_season_coeff_mean_prior = (damped_lag_season_coeff_mean_prior
                                                              .reshape(self.num_damped_lag_season, 1))
                    else:
                        pass

                    if not damped_lag_season_coeff_mean_prior.shape == (self.num_damped_lag_season, 1):
                        raise ValueError(
                            f'damped_lag_season_coeff_mean_prior must have shape '
                            f'({self.num_damped_lag_season}, 1), where the row count '
                            f'corresponds to the number of lags with damping.')
                    if np.any(np.isnan(damped_lag_season_coeff_mean_prior)):
                        raise ValueError('damped_trend_coeff_mean_prior cannot have NaN values.')
                    if np.any(np.isinf(damped_lag_season_coeff_mean_prior)):
                        raise ValueError('damped_lag_season_coeff_mean_prior cannot have Inf/-Inf values.')
                    for j in range(self.num_damped_lag_season):
                        if abs(damped_lag_season_coeff_mean_prior[j, 0]) > 1:
                            warnings.warn(
                                f'The mean damped coefficient for seasonal lag {j} is greater than 1 in '
                                f'absolute value, which implies an explosive process. Note that an '
                                f'explosive process can be stationary, but it implies that the future '
                                f'is needed to predict the past.')

                if damped_lag_season_coeff_prec_prior is not None:
                    if not isinstance(damped_lag_season_coeff_prec_prior, (np.ndarray, list, tuple)):
                        raise TypeError(
                            'damped_lag_season_coeff_prec_prior must be a Numpy array, list, or tuple.')

                    if isinstance(damped_lag_season_coeff_prec_prior, (list, tuple)):
                        damped_lag_season_coeff_prec_prior = (np.asarray(damped_lag_season_coeff_prec_prior,
                                                                         dtype=np.float64))
                    else:
                        damped_lag_season_coeff_prec_prior = damped_lag_season_coeff_prec_prior.astype(float)

                    if damped_lag_season_coeff_prec_prior.ndim not in (1, 2):
                        raise ValueError('damped_lag_season_coeff_prec_prior must have dimension 1 or 2.')
                    elif damped_lag_season_coeff_prec_prior.ndim == 1:
                        damped_lag_season_coeff_prec_prior = (damped_lag_season_coeff_prec_prior
                                                              .reshape(self.num_damped_lag_season, 1))
                    else:
                        pass

                    if not damped_lag_season_coeff_prec_prior.shape == (self.num_damped_lag_season, 1):
                        raise ValueError(
                            f'damped_lag_season_coeff_prec_prior must have shape '
                            f'({self.num_damped_lag_season}, 1), where the row count '
                            f'corresponds to the number of lags with damping.')
                    if np.any(np.isnan(damped_lag_season_coeff_prec_prior)):
                        raise ValueError('damped_lag_season_coeff_prec_prior cannot have NaN values.')
                    if np.any(np.isinf(damped_lag_season_coeff_prec_prior)):
                        raise ValueError('damped_lag_season_coeff_prec_prior cannot have Inf/-Inf values.')
                    # No need to do symmetric/positive definite checks since the matrix is 1x1 for each periodicity

        # Dummy seasonal prior check
        if len(self.dummy_seasonal) > 0 and sum(self.stochastic_dummy_seasonal) > 0:
            if dum_season_var_shape_prior is not None:
                if not isinstance(dum_season_var_shape_prior, tuple):
                    raise TypeError(
                        'dum_seasonal_var_shape_prior must be a tuple '
                        'to accommodate potentially multiple seasonality.')
                if len(dum_season_var_shape_prior) != sum(self.stochastic_dummy_seasonal):
                    raise ValueError(
                        'The number of elements in dum_season_var_shape_prior must match the '
                        'number of stochastic periodicities in dummy_seasonal. That is, for each '
                        'stochastic periodicity in dummy_seasonal, there must be a corresponding '
                        'shape prior.')
                if not all(isinstance(i, (int, float)) for i in dum_season_var_shape_prior):
                    raise TypeError(
                        'All values in dum_season_var_shape_prior must be of type integer or float.')

                dum_season_var_shape_prior = tuple(float(i) for i in dum_season_var_shape_prior)
                if any(np.isnan(i) for i in dum_season_var_shape_prior):
                    raise ValueError('No values in dum_season_var_shape_prior can be NaN.')
                if any(np.isinf(i) for i in dum_season_var_shape_prior):
                    raise ValueError('No values in dum_season_var_shape_prior can be Inf/-Inf.')
                if not all(i > 0 for i in dum_season_var_shape_prior):
                    raise ValueError('All values in dum_season_var_shape_prior must be strictly positive.')

            if dum_season_var_scale_prior is not None:
                if not isinstance(dum_season_var_scale_prior, tuple):
                    raise TypeError(
                        'dum_seasonal_var_scale_prior must be a tuple '
                        'to accommodate potentially multiple seasonality.')
                if len(dum_season_var_scale_prior) != sum(self.stochastic_dummy_seasonal):
                    raise ValueError(
                        'The number of elements in dum_season_var_scale_prior must match the '
                        'number of stochastic periodicities in dummy_seasonal. That is, for each '
                        'stochastic periodicity in dummy_seasonal, there must be a corresponding '
                        'scale prior.')
                if not all(isinstance(i, (int, float)) for i in dum_season_var_scale_prior):
                    raise TypeError(
                        'All values in dum_season_var_scale_prior must be of type integer or float.')

                dum_season_var_scale_prior = tuple(float(i) for i in dum_season_var_scale_prior)
                if any(np.isnan(i) for i in dum_season_var_scale_prior):
                    raise ValueError('No values in dum_season_var_scale_prior can be NaN.')
                if any(np.isinf(i) for i in dum_season_var_scale_prior):
                    raise ValueError('No values in dum_season_var_scale_prior can be Inf/-Inf.')
                if not all(i > 0 for i in dum_season_var_scale_prior):
                    raise ValueError(
                        'All values in dum_season_var_scale_prior must be strictly positive.')

        # Trigonometric seasonal prior check
        if len(self.trig_seasonal) > 0 and sum(self.stochastic_trig_seasonal) > 0:
            if trig_season_var_shape_prior is not None:
                if not isinstance(trig_season_var_shape_prior, tuple):
                    raise TypeError(
                        'trig_seasonal_var_shape_prior must be a tuple '
                        'to accommodate potentially multiple seasonality.')
                if len(trig_season_var_shape_prior) != sum(self.stochastic_trig_seasonal):
                    raise ValueError(
                        'The number of elements in trig_season_var_shape_prior must match the '
                        'number of stochastic periodicities in trig_seasonal. That is, for each '
                        'stochastic periodicity in trig_seasonal, there must be a corresponding '
                        'shape prior.')
                if not all(isinstance(i, (int, float)) for i in trig_season_var_shape_prior):
                    raise TypeError(
                        'All values in trig_season_var_shape_prior must be of type integer or float.')

                trig_season_var_shape_prior = tuple(float(i) for i in trig_season_var_shape_prior)
                if any(np.isnan(i) for i in trig_season_var_shape_prior):
                    raise ValueError('No values in trig_season_var_shape_prior can be NaN.')
                if any(np.isinf(i) for i in trig_season_var_shape_prior):
                    raise ValueError('No values in trig_season_var_shape_prior can be Inf/-Inf.')
                if not all(i > 0 for i in trig_season_var_shape_prior):
                    raise ValueError('All values in trig_season_var_shape_prior must be strictly positive.')

            if trig_season_var_scale_prior is not None:
                if not isinstance(trig_season_var_scale_prior, tuple):
                    raise TypeError(
                        'trig_seasonal_var_scale_prior must be a tuple '
                        'to accommodate potentially multiple seasonality.')
                if len(trig_season_var_scale_prior) != sum(self.stochastic_trig_seasonal):
                    raise ValueError(
                        'The number of elements in trig_season_var_scale_prior must match the '
                        'number of stochastic periodicities in trig_seasonal. That is, for each '
                        'stochastic periodicity in trig_seasonal, there must be a corresponding '
                        'scale prior.')
                if not all(isinstance(i, (int, float)) for i in trig_season_var_scale_prior):
                    raise TypeError(
                        'All values in trig_season_var_scale_prior must be of type integer or float.')

                trig_season_var_scale_prior = tuple(float(i) for i in trig_season_var_scale_prior)
                if any(np.isinf(i) for i in trig_season_var_scale_prior):
                    raise ValueError('No values in trig_season_var_scale_prior can be NaN.')
                if any(np.isinf(i) for i in trig_season_var_scale_prior):
                    raise ValueError('No values in trig_season_var_scale_prior can be Inf/-Inf.')
                if not all(i > 0 for i in trig_season_var_scale_prior):
                    raise ValueError('All values in trig_season_var_scale_prior must be strictly positive.')

        # Predictors prior check
        if self.has_predictors:
            if reg_coeff_mean_prior is not None:
                if not isinstance(reg_coeff_mean_prior, (np.ndarray, list, tuple)):
                    raise TypeError('reg_coeff_mean_prior must be of type Numpy ndarray, list, or tuple.')

                if isinstance(reg_coeff_mean_prior, (list, tuple)):
                    reg_coeff_mean_prior = (np.asarray(reg_coeff_mean_prior,
                                                       dtype=np.float64))
                else:
                    reg_coeff_mean_prior = reg_coeff_mean_prior.astype(float)

                if reg_coeff_mean_prior.ndim not in (1, 2):
                    raise ValueError('reg_coeff_mean_prior must have dimension 1 or 2.')
                elif reg_coeff_mean_prior.ndim == 1:
                    reg_coeff_mean_prior = (reg_coeff_mean_prior.reshape(self.num_predictors, 1))
                else:
                    pass

                if not reg_coeff_mean_prior.shape == (self.num_predictors, 1):
                    raise ValueError(f'reg_coeff_mean_prior must have shape ({self.num_predictors}, 1).')
                if np.any(np.isnan(reg_coeff_mean_prior)):
                    raise ValueError('reg_coeff_mean_prior cannot have NaN values.')
                if np.any(np.isinf(reg_coeff_mean_prior)):
                    raise ValueError('reg_coeff_mean_prior cannot have Inf and/or -Inf values.')

            if reg_coeff_prec_prior is not None:
                if not isinstance(reg_coeff_prec_prior, (np.ndarray, list, tuple)):
                    raise TypeError('reg_coeff_prec_prior must be of type Numpy ndarray, list, or tuple.')

                if isinstance(reg_coeff_prec_prior, (list, tuple)):
                    reg_coeff_prec_prior = np.asarray(reg_coeff_prec_prior, dtype=np.float64)
                else:
                    reg_coeff_prec_prior = reg_coeff_prec_prior.astype(float)

                if reg_coeff_prec_prior.ndim != 2:
                    raise ValueError('reg_coeff_prec_prior must have dimension 2.')
                else:
                    pass

                if not reg_coeff_prec_prior.shape == (self.num_predictors, self.num_predictors):
                    raise ValueError(f'reg_coeff_prec_prior must have shape ({self.num_predictors}, '
                                     f'{self.num_predictors}).')
                if not ao.is_positive_definite(reg_coeff_prec_prior):
                    raise ValueError('reg_coeff_prec_prior must be a positive definite matrix.')
                if not ao.is_symmetric(reg_coeff_prec_prior):
                    raise ValueError('reg_coeff_prec_prior must be a symmetric matrix.')

            if zellner_prior_obs is not None:
                if isinstance(zellner_prior_obs, (int, float)) and zellner_prior_obs > 0:
                    zellner_prior_obs = float(zellner_prior_obs)
                else:
                    raise ValueError('zellner_prior_obs must be strictly positive.')

        # Set upper limits on variance draws and number of sampling iterations
        var_y = np.nanvar(self.response, ddof=1)
        if upper_var_limit is None:
            upper_var_limit = var_y
        else:
            if isinstance(upper_var_limit, (int, float)) and upper_var_limit > 0:
                pass
            else:
                raise ValueError('upper_var_limit must be strictly positive.')

        if max_samp_iter_factor is None:
            max_samp_iter = 2 * num_samp
        else:
            if isinstance(max_samp_iter_factor, (int, float)) and max_samp_iter_factor > 1:
                max_samp_iter = int(max_samp_iter_factor * num_samp)
            else:
                raise ValueError('max_samp_iter_factor must be greater than 1.')

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
        model = self._model_setup(
            response_var_shape_prior,
            response_var_scale_prior,
            level_var_shape_prior,
            level_var_scale_prior,
            damped_level_coeff_mean_prior,
            damped_level_coeff_prec_prior,
            trend_var_shape_prior,
            trend_var_scale_prior,
            damped_trend_coeff_mean_prior,
            damped_trend_coeff_prec_prior,
            lag_season_var_shape_prior,
            lag_season_var_scale_prior,
            damped_lag_season_coeff_mean_prior,
            damped_lag_season_coeff_prec_prior,
            dum_season_var_shape_prior,
            dum_season_var_scale_prior,
            trig_season_var_shape_prior,
            trig_season_var_scale_prior,
            reg_coeff_mean_prior,
            reg_coeff_prec_prior,
            zellner_prior_obs
        )

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
            ar_level_tran_idx = components['Level']['damped_transition_index']
        else:
            damped_level_coefficient = np.array([[[]]])

        # Initialize damped trend coefficient output, if applicable
        if self.trend and self.damped_trend:
            damped_trend_coefficient = np.empty((num_samp, 1, 1))
            trend_stoch_idx = [components['Trend']['stochastic_index']]
            trend_state_eqn_idx = [components['Trend']['start_state_eqn_index']]
            ar_trend_tran_idx = components['Trend']['damped_transition_index']
        else:
            damped_trend_coefficient = np.array([[[]]])

        # Initialize damped lag_seasonal coefficient output, if applicable
        if len(self.lag_seasonal) > 0 and self.num_damped_lag_season > 0:
            damped_season_coefficients = np.empty((num_samp, self.num_damped_lag_season, 1))
            season_stoch_idx = []
            season_state_eqn_idx = []
            ar_season_tran_idx = ()
            for j in self.lag_seasonal:
                season_stoch_idx.append(components[f'Lag-Seasonal.{j}']['stochastic_index'])
                season_state_eqn_idx.append(components[f'Lag-Seasonal.{j}']['start_state_eqn_index'])
                if components[f'Lag-Seasonal.{j}']['damped']:
                    ar_season_tran_idx += (components[f'Lag-Seasonal.{j}']['damped_transition_index'],)
                else:
                    pass
        else:
            damped_season_coefficients = np.array([[[]]])

        # Helper matrices
        q_eye = np.eye(q)
        n_ones = np.ones((n, 1))

        if self.has_predictors:
            Vt, _ = self._design_matrix_svd()
            reg_coeff_mean_prior = model.reg_coeff_mean_prior
            reg_coeff_prec_prior = model.reg_coeff_prec_prior
            reg_ninvg_coeff_cov_post = model.reg_ninvg_coeff_cov_post
            reg_ninvg_coeff_prec_post = model.reg_ninvg_coeff_prec_post
            gibbs_iter0_reg_coeff = model.gibbs_iter0_reg_coeff

            if gibbs_iter0_reg_coeff is None:
                gibbs_iter0_reg_coeff = reg_coeff_mean_prior

            regression_coefficients = np.empty((num_samp, self.num_predictors, 1))

            # Compute the Normal-Inverse-Gamma posterior covariance matrix and
            # precision-weighted mean prior ahead of time. This is to save
            # on computational expense in the sampling for-loop.
            W = Vt @ reg_ninvg_coeff_cov_post @ Vt.T
            c = reg_coeff_prec_prior @ reg_coeff_mean_prior
        else:
            regression_coefficients = np.array([[[]]])

        # Run Gibbs sampler
        s = 0
        num_iter = 0
        while s < num_samp:
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
                    init_state_plus_values[level_state_eqn_idx[0]] = (
                        dist.vec_norm(0., np.sqrt(damped_level_var))
                    )
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
                    init_state_plus_values[trend_state_eqn_idx[0]] = (
                        dist.vec_norm(0., np.sqrt(damped_trend_var))
                    )
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
                        init_state_plus_values[season_state_eqn_idx[j]] = (
                            dist.vec_norm(0., np.sqrt(damped_season_var))
                        )
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
                      init_state_covariance=init_state_covariance
                      )

            _filtered_state = y_kf.filtered_state
            _state_covariance = y_kf.state_covariance
            _filtered_prediction = y_kf.one_step_ahead_prediction
            _response_variance = y_kf.response_variance

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
                     init_state_covariance=init_state_covariance
                     )

            # Smoothed disturbances and state
            _smoothed_errors = dk.simulated_smoothed_errors
            _smoothed_state = dk.simulated_smoothed_state
            _smoothed_prediction = dk.simulated_smoothed_prediction

            if self.response_has_nan:
                y = self.response_replace_nan + self.response_nan_indicator * _smoothed_prediction

            if q > 0:
                state_resid = _smoothed_errors[:, 1:, 0]
                state_sse = dot(state_resid.T ** 2, n_ones)
                state_var_scale_post = state_var_scale_prior + 0.5 * dot(H, state_sse)
                state_err_var_post = dist.vec_ig(state_var_shape_post, state_var_scale_post)

                if np.all(state_err_var_post < upper_var_limit):
                    filtered_state[s] = _filtered_state
                    state_covariance[s] = _state_covariance
                    filtered_prediction[s] = _filtered_prediction
                    response_variance[s] = _response_variance
                    smoothed_errors[s] = _smoothed_errors
                    smoothed_state[s] = _smoothed_state
                    smoothed_prediction[s] = _smoothed_prediction
                    state_error_covariance[s] = q_eye * dot(H.T, state_err_var_post)

                    # Get new draw for the level's AR(1) coefficient, if applicable
                    if self.level and self.damped_level:
                        ar_args = dict(smoothed_state=smoothed_state[s][:n],
                                       lag=(1,),
                                       mean_prior=damped_level_coeff_mean_prior,
                                       precision_prior=damped_level_coeff_prec_prior,
                                       state_err_var_post=state_err_var_post,
                                       state_eqn_index=level_state_eqn_idx,
                                       state_err_var_post_index=level_stoch_idx)
                        damped_level_coefficient[s] = self._ar_state_post_upd(**ar_args)

                    # Get new draw for the trend's AR(1) coefficient, if applicable
                    if self.trend and self.damped_trend:
                        ar_args = dict(smoothed_state=smoothed_state[s][:n],
                                       lag=(1,),
                                       mean_prior=damped_trend_coeff_mean_prior,
                                       precision_prior=damped_trend_coeff_prec_prior,
                                       state_err_var_post=state_err_var_post,
                                       state_eqn_index=trend_state_eqn_idx,
                                       state_err_var_post_index=trend_stoch_idx)
                        damped_trend_coefficient[s] = self._ar_state_post_upd(**ar_args)

                    # Get new draw for lag_seasonal AR(1) coefficients, if applicable
                    if len(self.lag_seasonal) > 0 and self.num_damped_lag_season > 0:
                        ar_args = dict(smoothed_state=smoothed_state[s][:n],
                                       lag=self.lag_season_damped_lags,
                                       mean_prior=damped_lag_season_coeff_mean_prior,
                                       precision_prior=damped_lag_season_coeff_prec_prior,
                                       state_err_var_post=state_err_var_post,
                                       state_eqn_index=season_state_eqn_idx,
                                       state_err_var_post_index=season_stoch_idx)
                        damped_season_coefficients[s] = self._ar_state_post_upd(**ar_args)

                    # Get new draw for regression coefficients
                    if self.has_predictors:
                        smooth_time_prediction = smoothed_prediction[s] - Z[:, :, -1]
                        y_tilde = y - smooth_time_prediction  # y with smooth time prediction subtracted out
                        reg_coeff_mean_post = reg_ninvg_coeff_cov_post @ (X.T @ y_tilde + c)
                        response_var_scale_post = (
                            response_var_scale_prior
                            + 0.5 * (
                            y_tilde.T @ y_tilde
                            + reg_coeff_mean_prior.T @ reg_coeff_prec_prior @ reg_coeff_mean_prior
                            - reg_coeff_mean_post.T @ reg_ninvg_coeff_prec_post @ reg_coeff_mean_post
                            )
                        )
                        response_error_variance[s] = dist.vec_ig(
                            response_var_shape_post,
                            response_var_scale_post
                        )
                        cov_post = response_error_variance[s][0, 0] * W
                        regression_coefficients[s] = (
                                Vt.T @ np.random
                                .multivariate_normal(mean=(Vt @ reg_coeff_mean_post).flatten(),
                                                     cov=cov_post).reshape(-1, 1)
                        )
                    else:
                        # Get new draw for observation variance
                        smooth_one_step_ahead_prediction_resid = _smoothed_errors[:, 0]
                        response_var_scale_post = (
                                response_var_scale_prior
                                + 0.5 * dot(
                            smooth_one_step_ahead_prediction_resid.T,
                            smooth_one_step_ahead_prediction_resid)
                        )
                        response_error_variance[s] = dist.vec_ig(
                            response_var_shape_post,
                            response_var_scale_post
                        )

                    s += 1
            else:
                filtered_state[s] = _filtered_state
                state_covariance[s] = _state_covariance
                filtered_prediction[s] = _filtered_prediction
                response_variance[s] = _response_variance
                smoothed_errors[s] = _smoothed_errors
                smoothed_state[s] = _smoothed_state
                smoothed_prediction[s] = _smoothed_prediction

                if self.has_predictors:
                    smooth_time_prediction = smoothed_prediction[s] - Z[:, :, -1]
                    y_tilde = y - smooth_time_prediction  # y with smooth time prediction subtracted out
                    reg_coeff_mean_post = reg_ninvg_coeff_cov_post @ (X.T @ y_tilde + c)
                    response_var_scale_post = (
                            response_var_scale_prior
                            + 0.5 * (
                            y_tilde.T @ y_tilde
                            + reg_coeff_mean_prior.T @ reg_coeff_prec_prior @ reg_coeff_mean_prior
                            - reg_coeff_mean_post.T @ reg_ninvg_coeff_prec_post @ reg_coeff_mean_post
                            )
                    )
                    response_error_variance[s] = dist.vec_ig(
                        response_var_shape_post,
                        response_var_scale_post
                    )
                    cov_post = response_error_variance[s][0, 0] * W
                    regression_coefficients[s] = (
                            Vt.T @ np.random
                            .multivariate_normal(mean=(Vt @ reg_coeff_mean_post).flatten(),
                                                 cov=cov_post).reshape(-1, 1)
                    )
                else:
                    # Get new draw for observation variance
                    smooth_one_step_ahead_prediction_resid = _smoothed_errors[:, 0]
                    response_var_scale_post = (
                            response_var_scale_prior
                            + 0.5 * dot(
                        smooth_one_step_ahead_prediction_resid.T,
                        smooth_one_step_ahead_prediction_resid)
                    )
                    response_error_variance[s] = dist.vec_ig(
                        response_var_shape_post,
                        response_var_scale_post
                    )

                s += 1

            num_iter += 1
            if num_iter > max_samp_iter:
                raise MaxIterSamplingError(upper_var_limit, max_samp_iter)

        self.num_sampling_iterations = num_iter
        self.posterior = Posterior(
            num_samp,
            smoothed_state,
            smoothed_errors,
            smoothed_prediction,
            filtered_state,
            filtered_prediction,
            response_variance,
            state_covariance,
            response_error_variance,
            state_error_covariance,
            damped_level_coefficient,
            damped_trend_coefficient,
            damped_season_coefficients,
            regression_coefficients
        )
        self.high_posterior_variance = self._high_variance()

        return self.posterior

    def forecast(self,
                 num_periods: int,
                 burn: int = 0,
                 future_predictors: Union[np.ndarray, list, tuple, pd.Series, pd.DataFrame] = None
                 ) -> Forecast:

        """
        Posterior forecast distribution for the response and states.

        :param num_periods: positive integer. Defines how many future periods to forecast.

        :param burn: non-negative integer. Represents how many of the first posterior samples to
        ignore for computing statistics like the mean and variance of a parameter. Default value is 0.

        :param future_predictors: ndarray of dimension (k, 1), where k is the number of predictors.
        Data type is float64. Default is None.

        :return: ndarray of dimension (S, h, 1), where S is the number of posterior samples and
        h is the number of future periods to be forecast. This array represents the posterior
        forecast distribution for the response.
        """

        self._posterior_exists_check()
        posterior = self.posterior
        components = self.model_setup.components

        if isinstance(num_periods, int) and num_periods > 0:
            pass
        else:
            raise ValueError('num_periods must be a strictly positive integer.')

        if isinstance(burn, int) and burn >= 0:
            pass
        else:
            raise ValueError('burn must be a non-negative integer.')

        if isinstance(self.historical_time_index, pd.DatetimeIndex):
            # noinspection PyUnresolvedReferences
            freq = self.historical_time_index.freq
            last_historical_date = self.historical_time_index[-1]
            first_future_date = last_historical_date + 1 * freq
            last_future_date = last_historical_date + num_periods * freq
            self.future_time_index = pd.date_range(first_future_date, last_future_date, freq=freq)
        else:
            self.future_time_index = np.arange(self.num_obs, self.num_obs + num_periods)

        # Check and prepare future predictor data
        if self.has_predictors and future_predictors is not None:
            if not isinstance(future_predictors, (np.ndarray, list, tuple, pd.Series, pd.DataFrame)):
                raise TypeError(
                    "The future_predictors array must be a NumPy array, list, tuple, Pandas Series, "
                    "or Pandas DataFrame."
                )
            else:
                if isinstance(future_predictors, (list, tuple)):
                    fut_pred = np.asarray(future_predictors, dtype=np.float64)
                else:
                    fut_pred = future_predictors.copy()

                # -- data types match across predictors and future_predictors
                if not isinstance(fut_pred, self.predictors_type):
                    raise TypeError(
                        'Object types for predictors and future_predictors must match.'
                    )

                if not isinstance(self.future_time_index, type(self.historical_time_index)):
                    raise TypeError(
                        'The future_predictors and predictors indexes must be of the same type.'
                    )

                else:
                    # -- if Pandas type, grab index and column names
                    if isinstance(fut_pred, (pd.Series, pd.DataFrame)):
                        if not (fut_pred.index[:num_periods] == self.future_time_index).all():
                            raise ValueError(
                                'The future_predictors index must match the future time index '
                                'implied by the last observed date for the response and the '
                                'number of desired forecast periods. Check the class attribute '
                                'future_time_index to verify that it is correct.'
                            )

                        if isinstance(fut_pred, pd.Series):
                            future_predictors_names = [fut_pred.name]
                        else:
                            future_predictors_names = fut_pred.columns.values.tolist()

                        if len(future_predictors_names) != self.num_predictors:
                            raise ValueError(
                                f'The number of predictors used for historical estimation {self.num_predictors} '
                                f'does not match the number of predictors specified for forecasting '
                                f'{len(future_predictors_names)}. The same set of predictors must be used.'
                            )
                        else:
                            if not all(self.predictors_names[i] == future_predictors_names[i]
                                       for i in range(self.num_predictors)):
                                raise ValueError(
                                    'The order and names of the columns in predictors must match '
                                    'the order and names in future_predictors.'
                                )

                        fut_pred = fut_pred.to_numpy()

                # -- dimensions
                if fut_pred.ndim not in (1, 2):
                    raise ValueError('The future_predictors array must have dimension 1 or 2.')
                elif fut_pred.ndim == 1:
                    fut_pred = fut_pred.reshape(-1, 1)
                else:
                    pass

                if np.isnan(fut_pred).any():
                    raise ValueError('The future_predictors array cannot have null values.')
                if np.isinf(fut_pred).any():
                    raise ValueError('The future_predictors array cannot have Inf and/or -Inf values.')

                # Final sanity checks
                if self.num_predictors != fut_pred.shape[1]:
                    raise ValueError(
                        f'The number of predictors used for historical estimation {self.num_predictors} '
                        f'does not match the number of predictors specified for forecasting '
                        f'{fut_pred.shape[1]}. The same set of predictors must be used.'
                    )

                if num_periods > fut_pred.shape[0]:
                    raise ValueError(
                        f'The number of requested forecast periods {num_periods} exceeds the '
                        f'number of observations provided in future_predictors {fut_pred.shape[0]}. '
                        f'The former must be no larger than the latter.'
                    )
                else:
                    if num_periods < fut_pred.shape[0]:
                        fut_pred = fut_pred[:num_periods, :]
                        warnings.warn(
                            f'The number of requested forecast periods {num_periods} is less than the '
                            f'number of observations provided in future_predictors {fut_pred.shape[0]}. '
                            f'Only the first {num_periods} observations will be used '
                            f'in future_predictors.'
                        )

        elif self.has_predictors and future_predictors is None:
            raise ValueError(
                "The instantiated model has predictors, but forecast() was provided none. "
                "Future predictors must be passed to forecast() if the fitted model includes "
                "predictors."
            )
        elif not self.has_predictors and future_predictors is not None:
            fut_pred = np.array([[]])
            warnings.warn(
                "The instantiated model has no predictors, but forecast() was provided some. "
                "The future predictors passed to forecast() will be ignored."
            )
        else:
            fut_pred = np.array([[]])

        Z = self.observation_matrix(num_rows=num_periods)
        T = self.state_transition_matrix
        C = self.state_intercept_matrix
        R = self.state_error_transformation_matrix
        damped_level_transition_index = ()
        damped_trend_transition_index = ()
        damped_season_transition_index = ()

        if self.level and self.damped_level:
            damped_level_transition_index = components['Level']['damped_transition_index']

        if self.trend and self.damped_trend:
            damped_trend_transition_index = components['Trend']['damped_transition_index']

        if len(self.lag_seasonal) > 0 and self.num_damped_lag_season > 0:
            for c in components:
                if 'Lag-Seasonal' in c and components[c]['damped']:
                    damped_season_transition_index += (components[c]['damped_transition_index'],)

        response_forecast, state_forecast = _forecast(
            posterior=posterior,
            num_periods=num_periods,
            state_observation_matrix=Z,
            state_transition_matrix=T,
            state_intercept_matrix=C,
            state_error_transformation_matrix=R,
            future_predictors=fut_pred,
            burn=burn,
            damped_level_transition_index=damped_level_transition_index,
            damped_trend_transition_index=damped_trend_transition_index,
            damped_season_transition_index=damped_season_transition_index
        )

        return Forecast(response_forecast, state_forecast)

    def post_pred_dist(self,
                       predictors: Union[np.ndarray, list, tuple, pd.Series, pd.DataFrame] = None,
                       burn: int = 0,
                       smoothed: bool = False,
                       num_first_obs_ignore: int = None,
                       random_sample_size_prop: float = 1.
                       ) -> np.ndarray:
        """
        Generates the posterior predictive density of the response variable. Either the filtered
        or the smoothed state can be used. The observation equation for the response is

        y(t) = Z(t).a(t) + response_error(t),

        where Z(t) is the known observation matrix,

            a(t) = E[a(t) | t-1] if filtered,
            a(t) = E[a(t) | t = 1,...,n] if smoothed

        and response_error(t) is the error at time t, respectively.

        Filtered Case:
        -------------
        Note: The algorithm for computing the posterior distribution of the parameters is conditional
        on the whole response vector, i.e., on all time periods in the sample t=1,...,n.
        That is, the posterior distribution of the smoothed state vector a(t|y(1), y(2), ..., y(n))
        is given, but the posterior distribution of the filtered state vector a(t|y(t-1)) is not.
        Thus, variance of response_error(t), for any t, is not enough to capture all the variance
        in y(t) | y(t-1); the systematic variance of Z(t).a(t|t-1) also has to be accounted for.
        See below.

        The posterior predictive distribution for the response at any time t, conditional on t-1,
        is obtained by sampling from

        y(t) | t-1, Z(t), ResponseErrVar ~ N(Z(t).a(t|t-1) | a(t|t-1), Z(t).P(t|t-1).Z(t)' + ResponseErrVar),

        where P(t|t-1) = Var[a(t) | t-1] is the state covariance matrix captured by the Kalman filter, and
        ResponseErrVar = Var[error(t) | t-j] = Var[error(t) | t-k] for all j, k, j !=k,
        is the homoskedastic variance of the response error.

        Since y(t) | t-1 is conditionally independent of y(t-1) for all t, the posterior
        predictive distribution of y(t), t=1,...,n, can be sampled by independently
        sampling for each y(t).

        y(1) | t=0, Z(1), ResponseErrVar ~ N(Z(1).a(1|t=0) | a(1|t=0), Z(1).P(1|t=0).Z(1)' + ResponseErrVar)
        y(2) | t=1, Z(2), ResponseErrVar ~ N(Z(2).a(2|t=1) | a(2|t=1), Z(2).P(2|t=1).Z(2)' + ResponseErrVar)
        ...
        y(n) | t=n-1, Z(n), ResponseErrVar ~ N(Z(n).a(n|t=n-1) | a(n|t=n-1), Z(n).P(n|t=n-1).Z(T)' + ResponseErrVar)

        Smoothed Case:
        -------------

        The posterior predictive distribution for the response at any time t, conditional on t=1,...,n,
        is obtained by sampling from

        y(t) | t=1,...,n, Z(t), a(t|t=1,...,n), ResponseErrVar ~ N(Z(t).a(t|t=1,...,n), ResponseErrVar),

        Since y(t) | t=1,...,n is conditionally independent of y(t-1) for all t, the posterior
        predictive distribution of y(t), t=1,...,n, can be sampled by independently sampling for each y(t).

        y(1) | t=1,...,n, Z(1), a(1|t=1,...,n), ResponseErrVar ~ N(Z(1).a(1|t=1,...,n), ResponseErrVar)
        y(1) | t=1,...,n, Z(2), a(2|t=1,...,n), ResponseErrVar ~ N(Z(2).a(2|t=1,...,n), ResponseErrVar)
        ...
        y(n) | t=1,...,n, Z(n), a(n|t=1,...,n), ResponseErrVar ~ N(Z(n).a(n|t=1,...,n), ResponseErrVar)

        Note that conditioning on Z(t) is necessary if a regression component is specified because Z(t,s)
        includes X(t).dot(regression_coefficients(s)) by construction. Otherwise, conditioning on Z(t) is
        not necessary since Z(t,s) = Z for all t, s.

        :param predictors: Numpy array, list, tuple, Pandas Series, or Pandas DataFrame, float64.
        Array that represents a set of predictors different from the ones used for model estimation.
        The dimension must match the dimension of the predictors used for model estimation.

        :param burn: non-negative integer. Represents how many of the first posterior samples to
        ignore for computing statistics like the mean and variance of a parameter. Default value is 0.

        :param smoothed: boolean. If True, the smoothed Kalman values will be plotted (as opposed
        to the filtered Kalman values).

        :param num_first_obs_ignore: non-negative integer. Represents how many of the first observations
        of a response variable to ignore for computing the posterior predictive distribution of the
        response. The number of observations to ignore depends on the state specification of the unobserved
        components model. Some observations are ignored because diffuse initialization is used for the
        state vector in the Kalman filter. Default value is 0.

        :param random_sample_size_prop: float in interval (0, 1]. Represents the proportion of the
        posterior samples to take for constructing the posterior predictive distribution. Sampling is
        done without replacement. Default value is 1.

        :return:

        ndarray of dimension (S, n), where S is the number of posterior samples and
        n is the number of observations used for constructing the posterior predictive distribution.
        Note that S = random_sample_size_prop * S_total, where S_total is total number of
        posterior samples, and n = n_tot - num_first_obs_ignore, where n_tot is the total number
        of observations for the response.
        """

        self._posterior_exists_check()
        posterior = self.posterior

        if burn is None:
            burn = 0
        if smoothed is None:
            smoothed = False
        if num_first_obs_ignore is None:
            num_first_obs_ignore = self.num_first_obs_ignore
        if random_sample_size_prop is None:
            random_sample_size_prop = 1.

        if predictors is not None:
            if self.has_predictors:
                X = self.predictors[num_first_obs_ignore:, :]
                X_new = np.array(predictors)

                if X_new.ndim == 1:
                    X_new = X_new.reshape(-1, 1)
                X_new = X_new[num_first_obs_ignore:, :]

                if not np.all(X.shape == X_new.shape):
                    raise AssertionError("\n"
                                         "The dimension of predictors must match the dimension of the \n"
                                         "predictors array used for model estimation, i.e., the predictors \n"
                                         "array used for class instantiation.")
                else:
                    reg_coeff = posterior.regression_coefficients[burn:, :, 0].T
                    reg_comp = X @ reg_coeff
                    reg_comp_new = X_new @ reg_coeff
            else:
                raise AssertionError("Predictors were not used during model estimation, so \n"
                                     "they are not applicable for computing the posterior \n"
                                     "predictive distribution of the response.")

        if smoothed:
            response_mean = posterior.smoothed_prediction[burn:, num_first_obs_ignore:, 0]
            response_variance = posterior.response_error_variance[burn:]
        else:
            response_mean = posterior.filtered_prediction[burn:, num_first_obs_ignore:, 0]
            response_variance = posterior.response_variance[burn:, num_first_obs_ignore:, 0]

        if predictors is not None:
            response_mean = response_mean.copy() + (reg_comp_new - reg_comp).T

        num_posterior_samp = response_mean.shape[0]
        n = response_mean.shape[1]

        if int(random_sample_size_prop * num_posterior_samp) > posterior.num_samp:
            raise ValueError('random_sample_size_prop must be between 0 and 1.')

        if int(random_sample_size_prop * num_posterior_samp) < 1:
            raise ValueError('random_sample_size_prop implies a sample with less than 1 observation. '
                             'Provide a random_sample_size_prop such that the number of samples '
                             'is at least 1 but no larger than the number of posterior samples.')

        if random_sample_size_prop == 1:
            num_samp = num_posterior_samp
            S = list(np.arange(num_posterior_samp))
        else:
            num_samp = int(random_sample_size_prop * num_posterior_samp)
            S = list(np.random.choice(num_posterior_samp, num_samp, replace=False))

        ppd = np.empty((num_samp, n), dtype=np.float64)
        i = 0
        for s in S:
            ppd[i] = dist.vec_norm(response_mean[s],
                                   np.sqrt(response_variance[s][:, 0]))
            i += 1

        return ppd

    def plot_post_pred_dist(self,
                            predictors: Union[np.ndarray, list, tuple, pd.Series, pd.DataFrame] = None,
                            burn: int = 0,
                            smoothed: bool = False,
                            cred_int_level: float = 0.05,
                            num_first_obs_ignore: int = None,
                            random_sample_size_prop: float = 1.,
                            ax: plt.axis = None,
                            ppd: np.ndarray = None
                            ) -> plt.figure:
        """

        :param predictors: Numpy array, list, tuple, Pandas Series, or Pandas DataFrame, float64.
        Array that represents a set of predictors different from the ones used for model estimation.
        The dimension must match the dimension of the predictors used for model estimation.

        :param burn: non-negative integer. Represents how many of the first posterior samples to
        ignore for computing statistics like the mean and variance of a parameter. Default value is 0.

        :param smoothed: boolean. If True, the smoothed Kalman values will be plotted (as opposed
        to the filtered Kalman values).

        :param cred_int_level: float in (0, 1). Defines the width of the predictive distribution.
        E.g., a value of 0.05 will represent all values of the predicted variable between the
        2.5% and 97.5% quantiles.

        :param num_first_obs_ignore: non-negative integer. Represents how many of the first observations
        of a response variable to ignore for computing the posterior predictive distribution of the
        response. The number of observations to ignore depends on the state specification of the unobserved
        components model. Some observations are ignored because diffuse initialization is used for the
        state vector in the Kalman filter. Default value is 0.

        :param random_sample_size_prop: float in interval (0, 1]. Represents the proportion of the
        posterior samples to take for constructing the posterior predictive distribution. Sampling is
        done without replacement. Default value is 1.

        :param ax: Matplotlib pyplot axis object. Optional. Default is None.

        :param ppd: 2-d NumPy array: Posterior predictive distribution that's already been computed.

        :return: Matplotlib pyplot figure
        """

        self._posterior_exists_check()
        cred_int_lb = 0.5 * cred_int_level
        cred_int_ub = 1. - 0.5 * cred_int_level

        if num_first_obs_ignore is None:
            num_first_obs_ignore = self.num_first_obs_ignore

        y = self.response[num_first_obs_ignore:]
        if ppd is None:
            ppd = self.post_pred_dist(
                predictors=predictors,
                burn=burn,
                smoothed=smoothed,
                num_first_obs_ignore=num_first_obs_ignore,
                random_sample_size_prop=random_sample_size_prop
            )

        historical_time_index = self.historical_time_index[num_first_obs_ignore:]

        if smoothed:
            kalman_type = "Smoothed"
        else:
            kalman_type = "Filtered"

        if ax is None:
            fig, ax = plt.subplots(1)

        if predictors is None:
            ax.plot(historical_time_index, y)

        ax.plot(historical_time_index, np.mean(ppd, axis=0))
        lb = np.quantile(ppd, cred_int_lb, axis=0)
        ub = np.quantile(ppd, cred_int_ub, axis=0)
        ax.fill_between(historical_time_index, lb, ub, alpha=0.2)

        if predictors is None:
            ax.title.set_text(f"Predicted vs. observed response - {kalman_type}")
            ax.legend(("Observed", f"{kalman_type} prediction",
                       f"{100 * (1 - cred_int_level)}% credible interval"),
                      loc="upper left")
        else:
            ax.title.set_text(f"Predicted response - {kalman_type}")
            ax.legend((f"{kalman_type} prediction",
                       f"{100 * (1 - cred_int_level)}% credible interval"),
                      loc="upper left")

        return

    def waic(self,
             burn: int = None
             ) -> WAIC:
        self._posterior_exists_check()

        if burn is None:
            burn = 0

        posterior = self.posterior
        num_first_obs_ignore = self.num_first_obs_ignore
        response = self.response[num_first_obs_ignore:]
        post_resp_mean = posterior.filtered_prediction[burn:, num_first_obs_ignore:, 0]
        post_resp_var = posterior.response_variance[burn:, num_first_obs_ignore:, 0, 0]

        return watanabe_akaike(
            response=response.T,
            post_resp_mean=post_resp_mean,
            post_err_var=post_resp_var
        )

    def components(self,
                   burn: int = 0,
                   random_sample_size_prop: float = 1.,
                   smoothed: bool = True,
                   num_first_obs_ignore: int = None,
                   ) -> dict:

        """
        Plots the in-sample posterior components and posterior predictive distribution for the response.

        :param burn: non-negative integer. Represents how many of the first posterior samples to
        ignore for computing statistics like the mean and variance of a parameter. Default value is 0.

        :param random_sample_size_prop: float in interval (0, 1]. Represents the proportion of the
        posterior samples to take for constructing the posterior predictive distribution. Sampling is
        done without replacement. Default value is 1.

        :param smoothed: boolean. If True, the smoothed Kalman values will be plotted (as opposed
        to the filtered Kalman values).

        :param num_first_obs_ignore: non-negative integer. Represents how many of the first observations
        of a response variable to ignore for computing the posterior predictive distribution of the
        response. The number of observations to ignore depends on the state specification of the unobserved
        components model. Some observations are ignored because diffuse initialization is used for the
        state vector in the Kalman filter. Default value is 0.

        :return: tuple

            [1] dictionary: Captures the posterior for each time series component specified
            [2] 2-d NumPy array: Posterior predictive distribution used for generating the posterior
                                 of the irregular component
            [3] dictionary: Captures the arguments passed to post_pred_dist() for creating the posterior
                            predictive distribution
        """

        self._posterior_exists_check()
        posterior = self.posterior

        if isinstance(burn, int) and burn >= 0:
            pass
        else:
            raise ValueError('burn must be a non-negative integer.')

        if isinstance(random_sample_size_prop, float) and (0 < random_sample_size_prop <= 1):
            pass
        else:
            raise ValueError('random_sample_size_prop must be a value in the interval (0, 1].')

        if not isinstance(smoothed, bool):
            raise TypeError('smoothed must be of type bool.')

        if num_first_obs_ignore is None:
            num_first_obs_ignore = self.num_first_obs_ignore

        if self.has_predictors:
            X = self.predictors[num_first_obs_ignore:, :]
            reg_coeff = posterior.regression_coefficients[burn:, :, 0].T

        y = self.response[num_first_obs_ignore:, 0]
        n = self.num_obs
        Z = self.observation_matrix(num_rows=n - num_first_obs_ignore)
        model = self.model_setup
        components = model.components
        self._components_ppd = self.post_pred_dist(
            burn=burn,
            smoothed=smoothed,
            num_first_obs_ignore=num_first_obs_ignore,
            random_sample_size_prop=random_sample_size_prop
        )

        if smoothed:
            state = posterior.smoothed_state[burn:, num_first_obs_ignore:n, :, :]
        else:
            state = _simulate_posterior_predictive_filtered_state(
                posterior=posterior,
                burn=burn,
                num_first_obs_ignore=num_first_obs_ignore,
                random_sample_size_prop=random_sample_size_prop,
                has_predictors=self.has_predictors
            )

        comps = {}
        for i, c in enumerate(components):
            if c == 'Irregular':
                resid = y[np.newaxis] - self._components_ppd
                comps[c] = resid

            if c == 'Trend':
                state_eqn_index = components['Trend']['start_state_eqn_index']
                time_component = state[:, :, state_eqn_index, 0]
                comps[c] = time_component

            if c not in ('Irregular', 'Regression', 'Trend'):
                v = components[c]
                start_index, end_index = v['start_state_eqn_index'], v['end_state_eqn_index']
                A = Z[:, 0, start_index:end_index]
                B = state[:, :, start_index:end_index, 0]
                time_component = (A[np.newaxis] * B).sum(axis=2)
                comps[c] = time_component

            if c == 'Regression':
                reg_component = X.dot(reg_coeff)
                comps[c] = reg_component.T

        return comps

    def plot_components(self,
                        burn: int = 0,
                        cred_int_level: float = 0.05,
                        random_sample_size_prop: float = 1.,
                        smoothed: bool = True,
                        num_first_obs_ignore: int = None,
                        fig_size: tuple[Union[int, float], Union[int, float]] = (15, 9),
                        dpi: Union[int, float] = 125,
                        pad: Union[int, float] = 2.5
                        ) -> plt.figure:

        """
        Plots the in-sample posterior components and posterior predictive distribution for the response.

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

        :param num_first_obs_ignore: non-negative integer. Represents how many of the first observations
        of a response variable to ignore for computing the posterior predictive distribution of the
        response. The number of observations to ignore depends on the state specification of the unobserved
        components model. Some observations are ignored because diffuse initialization is used for the
        state vector in the Kalman filter. Default value is 0.

        :param fig_size: tuple of floats or ints that define the size of the figure in inches (height, width)

        :param dpi: int or float that defines the dots per inch of the figure

        :param pad: int or float that governs how much padding to add to the height and width of the subplots

        :return: Return a matplotlib.pyplot figure that plots the posterior predictive distribution
        of the response and the posterior of each state component.
        """

        if isinstance(cred_int_level, float) and (0 < cred_int_level < 1):
            pass
        else:
            raise ValueError('cred_int_level must be a value in the interval (0, 1).')

        if num_first_obs_ignore is None:
            num_first_obs_ignore = self.num_first_obs_ignore

        historical_time_index = self.historical_time_index[num_first_obs_ignore:]

        comps = self.components(
            burn=burn,
            random_sample_size_prop=random_sample_size_prop,
            smoothed=smoothed,
            num_first_obs_ignore=num_first_obs_ignore
        )
        cred_int_lb = 0.5 * cred_int_level
        cred_int_ub = 1. - 0.5 * cred_int_level

        fig, ax = plt.subplots(1 + len(comps))
        fig.set_size_inches(fig_size)
        fig.set_dpi(dpi)
        self.plot_post_pred_dist(
            ppd=self._components_ppd,
            burn=burn,
            random_sample_size_prop=random_sample_size_prop,
            smoothed=smoothed,
            num_first_obs_ignore=num_first_obs_ignore,
            cred_int_level=cred_int_level,
            ax=ax[0],
        )
        ax[0].get_legend().remove()
        yticks = ax[0].get_yticks()
        ax[0].set_yticks(yticks[::max(1, len(yticks) // 5)])

        for i, c in enumerate(comps):
            x = comps[c]
            ax[i + 1].plot(historical_time_index, np.mean(x, axis=0))
            lb = np.quantile(x, cred_int_lb, axis=0)
            ub = np.quantile(x, cred_int_ub, axis=0)
            ax[i + 1].fill_between(historical_time_index, lb, ub, alpha=0.2)
            ax[i + 1].title.set_text(c)
            yticks = ax[i + 1].get_yticks()
            ax[i + 1].set_yticks(yticks[::max(1, len(yticks) // 5)])

        fig.tight_layout(pad=pad)

        return

    def plot_trace(self,
                   parameters: list = None,
                   burn: int = 0,
                   fig_size: tuple[Union[int, float], Union[int, float]] = (15, 9),
                   dpi: Union[int, float] = 125,
                   pad: Union[int, float] = 2.5
                   ) -> plt.figure:
        """

        :param parameters: list of parameter names. A histogram and sampling trace will be plotted
        for each parameter in the list. Default is all parameters in the model, which can be accessed
        via the class attribute 'parameters'.

        :param burn: non-negative integer. Represents how many of the first posterior samples to
        ignore for computing statistics like the mean and variance of a parameter. Default value is 0.

        :param fig_size: tuple of floats or ints that define the size of the figure in inches (height, width)

        :param dpi: int or float that defines the dots per inch of the figure

        :param pad: int or float that governs how much padding to add to the height and width of the subplots

        :return:
        """

        if parameters is None:
            parameters = self.parameters

        if not isinstance(parameters, list):
            raise ValueError("parameters must be a list.")
        else:
            if not all(i in self.parameters for i in parameters):
                raise ValueError("Not all parameter names are valid. See the class attribute "
                                 "'parameters' for the full list of valid parameter names.")

        if isinstance(burn, int) and burn >= 0:
            pass
        else:
            raise ValueError('burn must be a non-negative integer.')

        self._posterior_exists_check()
        post_dict = self.posterior_dict(burn=burn)
        num_params = len(parameters)

        fig, ax = plt.subplots(num_params, 2)
        fig.set_size_inches(fig_size)
        fig.set_dpi(dpi)
        row = 0
        for p in parameters:
            histplot(post_dict[p], ax=ax[row, 0], kde=True)
            lineplot(post_dict[p], ax=ax[row, 1])
            ax[row, 0].title.set_text(p)
            yticks = ax[row, 0].get_yticks()
            ax[row, 0].set_yticks(yticks[::max(1, len(yticks) // 5)])
            yticks = ax[row, 1].get_yticks()
            ax[row, 1].set_yticks(yticks[::max(1, len(yticks) // 5)])
            xticks = ax[row, 0].get_xticks()
            ax[row, 0].set_xticks(xticks[::max(1, len(xticks) // 5)])
            ax[row, 0].set_xlim(min(post_dict[p]), max(post_dict[p]))
            xticks = ax[row, 1].get_xticks()
            ax[row, 1].set_xticks(xticks[::max(1, len(xticks) // 5)])
            ax[row, 1].set_xlim(0, len(post_dict[p]))
            row += 1

        fig.tight_layout(pad=pad)

        return

    def summary(self,
                burn: int = 0,
                cred_int_level: float = 0.05
                ) -> dict:

        """
        Summary of the posterior distribution for each parameter in the model.

        :param burn: non-negative integer. Represents how many of the first posterior samples to
        ignore for computing statistics like the mean and variance of a parameter. Default value is 0.

        :param cred_int_level: float in (0, 1). Defines the width of the predictive distribution.
        E.g., a value of 0.05 will represent all values of the predicted variable between the
        2.5% and 97.5% quantiles.

        :return: A dictionary that summarizes the statistics of each of the model's parameters.
        Statistics include the mean, standard deviation, and lower and upper quantiles defined by
        cred_int_level.
        """

        self._posterior_exists_check()
        num_post_samp_after_burn = self.posterior.num_samp - burn
        post_dict = self.posterior_dict(burn=burn)

        if isinstance(burn, int) and burn >= 0:
            pass
        else:
            raise ValueError('burn must be a non-negative integer.')

        if isinstance(cred_int_level, float) and (0 < cred_int_level < 1):
            lb = 0.5 * cred_int_level
            ub = 1. - lb
        else:
            raise ValueError('cred_int_level must be a value in the interval (0, 1).')

        smy = {'Number of posterior samples (after burn)': num_post_samp_after_burn}

        for k, v in post_dict.items():
            smy[f"Posterior.Mean[{k}]"] = np.mean(v)
            smy[f"Posterior.StdDev[{k}]"] = np.std(v)
            smy[f"Posterior.CredInt.LB[{k}]"] = np.quantile(v, lb)
            smy[f"Posterior.CredInt.UB[{k}]"] = np.quantile(v, ub)

            if 'Coeff.' in k:
                smy[f"Posterior.ProbPositive[{k}]"] = np.sum((v > 0) * 1) / num_post_samp_after_burn

        return smy
