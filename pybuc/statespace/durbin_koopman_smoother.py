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
                                     init_state,
                                     response_error_variance,
                                     state_error_variance_matrix):
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
        error_variances = np.concatenate((response_error_variance, ao.diag_2d(state_error_variance_matrix)))
    else:
        error_variances = response_error_variance

    errors = np.empty((n, 1 + q, 1), dtype=np.float64)
    for t in range(n):
        errors[t] = dist.vec_norm(np.zeros((1 + q, 1)), np.sqrt(error_variances))

    alpha = np.empty((n + 1, m, 1), dtype=np.float64)
    alpha[0] = init_state
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
                state_error_variance_matrix: np.ndarray,
                init_state_values: np.ndarray,
                init_state_plus_values: np.ndarray,
                init_state_covariance: np.ndarray,
                has_predictors: bool = False):
    # Get state and observation transformation matrices
    T = state_transition_matrix
    C = state_intercept_matrix
    Z = observation_matrix
    R = state_error_transformation_matrix

    # Store number of state variables (m), state parameters (u), observations (n)
    m = Z.shape[2]
    q = R.shape[1]
    n = y.shape[0]

    init_state_cov = init_state_covariance.copy()
    if has_predictors:
        init_state_cov[-1, -1] = 1e6
    else:
        init_state_cov = init_state_covariance

    C_plus = C - C
    sim_fake_lss = simulate_fake_linear_state_space(Z,
                                                    T,
                                                    C_plus,
                                                    R,
                                                    init_state_plus_values,
                                                    response_error_variance_matrix,
                                                    state_error_variance_matrix)

    y_plus = sim_fake_lss.simulated_response
    alpha_plus = sim_fake_lss.simulated_state
    w_plus = sim_fake_lss.simulated_errors

    # Run y* = y - y+ through Kalman filter.
    y_star = y - y_plus
    y_star_kf = kf(y_star,
                   Z,
                   T,
                   C,
                   R,
                   response_error_variance_matrix,
                   state_error_variance_matrix,
                   init_state=init_state_values - init_state_plus_values,
                   init_state_covariance=init_state_cov)

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

    # Compute smoothed observation and state residuals and state vector
    w_hat = np.empty((n, 1 + q, 1), dtype=np.float64)
    alpha_hat = np.empty((n + 1, m, 1), dtype=np.float64)
    alpha_hat[0] = init_state_values - init_state_plus_values + init_state_covariance.dot(r_init)
    for t in range(n):
        eps_hat = response_error_variance_matrix.dot(F_inv[t].dot(v[t]) - K[t].T.dot(r[t]))
        if q > 0:
            eta_hat = state_error_variance_matrix.dot(R.T).dot(r[t])
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
