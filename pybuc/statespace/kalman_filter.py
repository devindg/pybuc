import numpy as np
from typing import NamedTuple
from numba import njit
from ..utils import array_operations as ao


class KF(NamedTuple):
    one_step_ahead_prediction: np.ndarray
    one_step_ahead_prediction_resid: np.ndarray
    kalman_gain: np.ndarray
    filtered_state: np.ndarray
    state_covariance: np.ndarray
    response_variance: np.ndarray
    inverse_response_variance: np.ndarray
    L: np.ndarray


@njit(cache=True)
def kalman_filter(y: np.ndarray,
                  observation_matrix: np.ndarray,
                  state_transition_matrix: np.ndarray,
                  state_intercept_matrix: np.ndarray,
                  state_error_transformation_matrix: np.ndarray,
                  response_error_variance_matrix: np.ndarray,
                  state_error_covariance_matrix: np.ndarray,
                  init_state: np.ndarray = np.array([[]]),
                  init_state_covariance: np.ndarray = np.array([[]])):
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
    The initial starting values for each of the m state equations. If none are
    provided, a (m, 1) matrix of zeros will be used.

    :param init_state_covariance: ndarray of dimension (m, m). Data type must be float64.
    The initial state covariance matrix for the m state equations. If one is not
    provided, a (m, m) diagonal matrix with 1e6 along the diagonal will be used.

    :return: Named tuple with the following:

    1. one_step_ahead_prediction: One-step-ahead predicted values from Kalman filter
    2. one_step_ahead_prediction_resid: One-step-ahead residuals from Kalman filter
    3. kalman_gain: Kalman gain matrix
    4. filtered_state: Filtered state values, E[state(t) | state(t-1)]
    5. state_covariance: State covariance matrix, E[dot(state(t), state(t).T) | state(t-1)]
    6. response_variance: Response variance, E[dot(y(t).T, y(t)) | state(t)]
    7. inverse_response_variance: Inverse of response variance
    8. L: State Transition Matrix - dot(Kalman Gain, Observation Matrix)
    """
    # Get state and observation transformation matrices
    T = state_transition_matrix
    C = state_intercept_matrix
    Z = observation_matrix
    R = state_error_transformation_matrix

    # Establish number of state variables (m), state parameters (q), and observations (n)
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

    # Initialize Kalman filter matrices
    y_pred = np.empty((n, 1), dtype=np.float64)
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
        y_pred[t] = Z[t].dot(a[t])
        v[t] = (1. - y_nan_indicator[t]) * (y_no_nan[t] - y_pred[t])
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
            P[t + 1] = T.dot(P[t]).dot(L[t].T) + R.dot(state_error_covariance_matrix).dot(R.T)
        else:
            P[t + 1] = T.dot(P[t]).dot(L[t].T)

    return KF(y_pred, v, K, a, P, F, F_inv, L)
