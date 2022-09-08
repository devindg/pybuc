import numpy as np
from typing import NamedTuple
from numba import njit
from ..utils import array_operations as ao


class KF(NamedTuple):
    one_step_ahead_prediction_resid: np.ndarray
    kalman_gain: np.ndarray
    filtered_state: np.ndarray
    state_covariance: np.ndarray
    response_variance: np.ndarray
    inverse_response_variance: np.ndarray
    L: np.ndarray


# @njit
# def var_mat_check(var_mat):
#     if not is_2d(var_mat):
#         raise ValueError('The variance-covariance matrix must have a row and column dimension. '
#                          'Flat/1D arrays or arrays with more than 2 dimensions are not valid.')
#
#     if not is_square(var_mat):
#         raise ValueError('The variance-covariance matrix must be square.')
#
#     return
#
#
# @njit
# def lss_mat_check(y: np.ndarray,
#                   observation_matrix: np.ndarray,
#                   state_transition_matrix: np.ndarray,
#                   state_error_transformation_matrix: np.ndarray):
#     Z = observation_matrix
#     T = state_transition_matrix
#     R = state_error_transformation_matrix
#     m = Z.shape[2]
#     n = y.shape[0]
#
#     # if not all(is_2d(Z[t]) for t in range(n)):
#     #     raise ValueError('The observation matrix must have a row and column dimension for each observation. '
#     #                      'Flat/1D arrays or arrays with more than 2 dimensions are not valid.')
#
#     if not is_2d(T):
#         raise ValueError('The state transition matrix must have a row and column dimension. '
#                          'Flat/1D arrays or arrays with more than 2 dimensions are not valid.')
#
#     if not is_2d(R):
#         raise ValueError('The state error transformation matrix must have a row and column dimension. '
#                          'Flat/1D arrays or arrays with more than 2 dimensions are not valid.')
#
#     if not is_square(T):
#         raise ValueError('The state transition matrix must be square.')
#
#     if Z.shape[1] != 1:
#         raise ValueError('The number of rows in the observation matrix must match the '
#                          'number of observation variables, i.e., 1..')
#
#     if T.shape[0] != m:
#         raise ValueError('The numbers of rows/columns in the state transition matrix '
#                          'must match the number of columns in the observation matrix.')
#
#     if R.shape[0] != m:
#         raise ValueError('The number of rows in the state error transformation matrix '
#                          'must match the number of columns in the observation matrix.')
#
#     return


@njit(cache=True)
def kalman_filter(y: np.ndarray,
                  observation_matrix: np.ndarray,
                  state_transition_matrix: np.ndarray,
                  state_intercept_matrix: np.ndarray,
                  state_error_transformation_matrix: np.ndarray,
                  response_error_variance_matrix: np.ndarray,
                  state_error_variance_matrix: np.ndarray,
                  init_state: np.ndarray = np.array([[]]),
                  init_state_covariance: np.ndarray = np.array([[]])):
    # Get state and observation transformation matrices
    T = state_transition_matrix
    C = state_intercept_matrix
    Z = observation_matrix
    R = state_error_transformation_matrix

    # Establish number of state variables (m), state parameters (q), and observations (n)
    m = Z.shape[2]
    q = R.shape[1]
    n = y.shape[0]

    # Initialize Kalman filter matrices
    v = np.empty((n, 1, 1), dtype=np.float64)  # Observation residuals
    K = np.empty((n, m, 1), dtype=np.float64)  # Kalman gain
    L = np.empty((n, m, m), dtype=np.float64)  # Kalman gain transformation
    a = np.empty((n + 1, m, 1), dtype=np.float64)  # A priori state prediction
    P = np.empty((n + 1, m, m), dtype=np.float64)  # A priori state variance
    F = np.empty((n, 1, 1), dtype=np.float64)  # Total variance (state plus observation)
    F_inv = np.empty((n, 1, 1), dtype=np.float64)  # Inverse of total variance (state plus observation)

    # Initialize Kalman filter
    if init_state.size == 0:
        a[0] = np.zeros((m, 1))
    else:
        a[0] = init_state

    if init_state_covariance.size == 0:
        P[0] = np.diag(np.ones(m) * 1e6)
    else:
        P[0] = init_state_covariance

    # Check if y has NaN values. Find indices, if any, and use this to assign v[t] to 0.
    # This is functionally the same as setting y[t] = Za[t] when y[t] is missing.
    # Thus, in the absence of a measurement/observation, the Kalman Filter will impute
    # the missing observation using the state prediction a[t | t-1].
    y_nan_indicator = np.isnan(y) * 1.
    y_no_nan = ao.replace_nan(y)

    # Run Kalman Filter
    for t in range(n):
        v[t] = (1. - y_nan_indicator[t]) * (y_no_nan[t] - Z[t].dot(a[t]))
        F[t] = Z[t].dot(P[t]).dot(Z[t].T) + response_error_variance_matrix
        # Get appropriate matrix inversion procedure for F.
        # Matrix inversion is computationally expensive,
        # so it's good to avoid needlessly using matrix inversion
        # on 1x1 matrices.
        F_inv[t] = ao.mat_inv(F[t])
        K[t] = T.dot(P[t]).dot(Z[t].T).dot(F_inv[t])
        L[t] = T - K[t].dot(Z[t])
        a[t + 1] = C + T.dot(a[t]) + K[t].dot(v[t])

        if q > 0:
            P[t + 1] = T.dot(P[t]).dot(L[t].T) + R.dot(state_error_variance_matrix).dot(R.T)
        else:
            P[t + 1] = T.dot(P[t]).dot(L[t].T)

    return KF(v, K, a, P, F, F_inv, L)
