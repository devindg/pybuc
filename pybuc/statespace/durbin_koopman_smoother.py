import numpy as np
from typing import NamedTuple
from numba import njit
from ..statespace.kalman_filter import kalman_filter as kf
from ..vectorized import distributions as dist
from ..utils import array_operations as ao


class DKSS(NamedTuple):
    simulated_smoothed_errors: np.ndarray
    simulated_smoothed_state: np.ndarray
    simulated_smoothed_prediction: np.ndarray


class SLSS(NamedTuple):
    simulated_response: np.ndarray
    simulated_state: np.ndarray
    simulated_errors: np.ndarray


@njit(cache=True)
def simulate_fake_linear_state_space(observation_matrix,
                                     state_transition_matrix,
                                     state_intercept_matrix,
                                     state_error_transformation_matrix,
                                     init_state_plus,
                                     response_error_variance_matrix,
                                     state_error_covariance_matrix):
    """

    :param observation_matrix: ndarray of dimension (n, 1, m), where m is the
    number of state equations. Data type must be float64. Matrix that maps the
    state to the response.

    :param state_transition_matrix: ndarray of dimension (m, m). Data type must
    be float64. Matrix that maps previous state values to the current state.

    :param state_intercept_matrix: ndarray of dimension (m, 1). Data type
    must be float64. An intercept that may be added to the transition state equations.
    Generally, the intercept is a matrix of zeros and thus has no impact.

    :param state_error_transformation_matrix: ndarray of dimension (m, q), where
    q is the number of state equations that evolve stochastically. Data type must
    be float64. Matrix that maps state equations to stochastic or non-stochastic form.

    :param init_state_plus: ndarray of dimension (m, 1). Data type must be float64.
    The initial starting values for each of the m state equations.

    :param response_error_variance_matrix: ndarray of dimension (1, 1). Data type
    must be float64. A singleton-matrix that captures the error variance of the
    response.

    :param state_error_covariance_matrix: ndarray of dimension (q, q). Data type
    must be float64. A diagonal matrix that captures the error variance of each
    stochastic state equation. It is possible for this matrix to be empty, in
    which case no state equation evolves stochastically.

    :return: Named tuple with the following:

    1. simulated_response: ndarray of dimension (n, 1). Simulated 0-mean response.
    2. simulated_state: ndarray of dimension (n + 1, m, 1). Simulated 0-mean state.
    3. simulated_errors: ndarray of dimension (n, 1 + q, 1). Simulated error with
    0 mean and variances matching the error variances estimated from the model.
    """
    # Get state and observation transformation matrices
    T = state_transition_matrix
    C = state_intercept_matrix
    Z = observation_matrix
    R = state_error_transformation_matrix

    # Establish number of state variables (m), state parameters (q), observations (n)
    m = Z.shape[2]
    q = R.shape[1]
    n = Z.shape[0]

    if q > 0:
        error_variances = np.concatenate((response_error_variance_matrix,
                                          ao.diag_2d(state_error_covariance_matrix)))
    else:
        error_variances = response_error_variance_matrix

    errors = np.empty((n, 1 + q, 1), dtype=np.float64)
    for t in range(n):
        errors[t] = dist.vec_norm(np.zeros((1 + q, 1)), np.sqrt(error_variances))

    alpha = np.empty((n + 1, m, 1), dtype=np.float64)
    alpha[0] = init_state_plus
    y = np.empty((n, 1), dtype=np.float64)
    for t in range(n):
        y[t] = Z[t].dot(alpha[t]) + errors[t, 0, :]
        if q > 0:
            alpha[t + 1] = C + T.dot(alpha[t]) + R.dot(errors[t, 1:])
        else:
            alpha[t + 1] = C + T.dot(alpha[t])

    return SLSS(y, alpha, errors)


@njit(cache=True)
def dk_smoother(y: np.ndarray,
                observation_matrix: np.ndarray,
                state_transition_matrix: np.ndarray,
                state_intercept_matrix: np.ndarray,
                state_error_transformation_matrix: np.ndarray,
                response_error_variance_matrix: np.ndarray,
                state_error_covariance_matrix: np.ndarray,
                init_state: np.ndarray,
                init_state_plus: np.ndarray,
                init_state_covariance: np.ndarray):
    """

    :param y: ndarray of dimension (n, 1), where n is the number of observations.
    Data type must be float64. Response variable for observation equation.

    :param observation_matrix: ndarray of dimension (n, 1, m), where m is the
    number of state equations. Data type must be float64. Matrix that maps the
    state to the response.

    :param state_transition_matrix: ndarray of dimension (m, m). Data type must
    be float64. Matrix that maps previous state values to the current state.

    :param state_intercept_matrix: ndarray of dimension (m, 1). Data type
    must be float64. An intercept that may be added to the transition state equations.
    Generally, the intercept is a matrix of zeros and thus has no impact.

    :param state_error_transformation_matrix: ndarray of dimension (m, q), where
    q is the number of state equations that evolve stochastically. Data type must
    be float64. Matrix that maps state equations to stochastic or non-stochastic form.

    :param response_error_variance_matrix: ndarray of dimension (1, 1). Data type
    must be float64. A singleton-matrix that captures the error variance of the
    response.

    :param state_error_covariance_matrix: ndarray of dimension (q, q). Data type
    must be float64. A diagonal matrix that captures the error variance of each
    stochastic state equation. It is possible for this matrix to be empty, in
    which case no state equation evolves stochastically.

    :param init_state: ndarray of dimension (m, 1). Data type must be float64.
    The initial starting values for each of the m state equations.

    :param init_state_plus: ndarray of dimension (m, 1). Data type must be float64.
    The initial starting values for each of the synthetic m state equations.

    :param init_state_covariance: ndarray of dimension (m, m). Data type must be float64.
    The initial state covariance matrix for the m state equations.


    :return: Named tuple with the following:

    1. simulated_smoothed_errors: np.ndarray of dimension (n, 1 + q, 1). Simulated smoothed
    errors from Durbin-Koopman algorithm.
    2. simulated_smoothed_state: np.ndarray of dimension (n + 1, m, 1). Simulated smoothed
    state from Durbin-Koopman algorithm.
    3. simulated_smoothed_prediction: np.ndarray of dimension (n, 1). Simulated smoothed
    response prediction from Durbin-Koopman algorithm.
    """
    # Get state and observation transformation matrices
    T = state_transition_matrix
    C = state_intercept_matrix
    Z = observation_matrix
    R = state_error_transformation_matrix

    # Store number of state variables (m), stochastic state equations (q), and observations (n)
    m = T.shape[0]
    q = R.shape[1]
    n = y.shape[0]

    # Data checks
    if not y.shape == (n, 1):
        raise ValueError('The response vector must have shape (n, 1), where n is the number of '
                         'observations.')

    if not T.shape == (m, m):
        raise ValueError('The state transition matrix must have shape (m, m), where m denotes '
                         'the number of state equations.')

    if not Z.shape == (n, 1, m):
        raise ValueError('The observation matrix must have shape (n, 1, m), where n denotes the '
                         'number of observations and m is the number of stochastic state equations.')

    if not R.shape == (m, q):
        raise ValueError('The state error transformation matrix must have shape (m, q), where m '
                         'denotes the number of state equations and q is the number of stochastic '
                         'state equations.')

    if not C.shape == (m, 1):
        raise ValueError('The state intercept vector must have shape (m, 1), where m denotes '
                         'the number of state equations.')

    if not response_error_variance_matrix.shape == (1, 1):
        raise ValueError('The response error variance matrix must have shape (1, 1).')

    if not state_error_covariance_matrix.shape == (q, q):
        raise ValueError('The state error covariance matrix must have shape (q, q), where '
                         'q denotes the number of stochastic state equations.')

    if not np.all(response_error_variance_matrix > 0):
        raise ValueError('The response error variance must be a strictly positive number.')

    if not np.all(np.diag(state_error_covariance_matrix) >= 0):
        raise ValueError('All values along the diagonal of the state error covariance matrix '
                         'must be non-negative.')

    if not ao.is_symmetric(state_error_covariance_matrix):
        raise ValueError('The state error covariance matrix must be symmetric.')

    if not init_state.shape == (m, 1):
        raise ValueError('The initial state value vector must have shape (m, 1), where m denotes '
                         'the number of state equations.')

    if not init_state_covariance.shape == (m, m):
        raise ValueError('The initial state covariance matrix must have shape (m, m), where m denotes '
                         'the number of state equations.')

    C_plus = C - C
    sim_fake_lss = simulate_fake_linear_state_space(observation_matrix=Z,
                                                    state_transition_matrix=T,
                                                    state_intercept_matrix=C_plus,
                                                    state_error_transformation_matrix=R,
                                                    init_state_plus=init_state_plus,
                                                    response_error_variance_matrix=response_error_variance_matrix,
                                                    state_error_covariance_matrix=state_error_covariance_matrix)

    y_plus = sim_fake_lss.simulated_response
    alpha_plus = sim_fake_lss.simulated_state
    w_plus = sim_fake_lss.simulated_errors

    # Run y* = y - y+ through Kalman filter.
    y_star = y - y_plus
    y_star_kf = kf(y=y_star,
                   observation_matrix=Z,
                   state_transition_matrix=T,
                   state_intercept_matrix=C,
                   state_error_transformation_matrix=R,
                   response_error_variance_matrix=response_error_variance_matrix,
                   state_error_covariance_matrix=state_error_covariance_matrix,
                   init_state=init_state - init_state_plus,
                   init_state_covariance=init_state_covariance)

    v = y_star_kf.one_step_ahead_prediction_resid
    F_inv = y_star_kf.inverse_response_variance
    K = y_star_kf.kalman_gain
    L = y_star_kf.L

    # Backward-pass filter for observation and state residuals
    r = np.empty((n, m, 1), dtype=np.float64)
    r[-1] = np.zeros((m, 1), dtype=np.float64)
    for t in range(n - 1, 0, -1):
        r[t - 1] = Z[t].T.dot(F_inv[t]).dot(v[t]) + L[t].T.dot(r[t])

    # Initial backward-pass residual for computing the smoothed state
    r_init = Z[0].T.dot(F_inv[0]).dot(v[0]) + L[0].T.dot(r[0])

    # Compute smoothed observation residuals, state residuals, and state vector
    w_hat = np.empty((n, 1 + q, 1), dtype=np.float64)
    alpha_hat = np.empty((n + 1, m, 1), dtype=np.float64)

    alpha_hat[0] = init_state - init_state_plus + init_state_covariance.dot(r_init)
    for t in range(n):
        eps_hat = response_error_variance_matrix.dot(F_inv[t].dot(v[t]) - K[t].T.dot(r[t]))
        if q > 0:
            eta_hat = state_error_covariance_matrix.dot(R.T).dot(r[t])
            w_hat[t] = np.concatenate((eps_hat, eta_hat))
            alpha_hat[t + 1] = C + T.dot(alpha_hat[t]) + R.dot(eta_hat)
        else:
            w_hat[t] = eps_hat
            alpha_hat[t + 1] = C + T.dot(alpha_hat[t])

    # Compute simulation smoothed errors (E[error | y]), state (E[state | y]), and prediction
    smoothed_errors = w_hat + w_plus
    smoothed_state = alpha_hat + alpha_plus
    smoothed_prediction = (Z[:, 0, :] * smoothed_state[:n, :, 0]).dot(np.ones((m, 1)))

    return DKSS(smoothed_errors, smoothed_state, smoothed_prediction)
