from numba import njit
import numpy as np


@njit
def diag_2d(x: np.ndarray, as_col: bool = True) -> np.ndarray:
    if as_col:
        return np.diag(x).reshape(-1, 1)
    else:
        return np.diag(x).reshape(1, -1)


@njit
def replace_nan(x):
    z = x.copy()
    shape = z.shape
    z = z.ravel()
    z[np.isnan(z)] = 0.
    z = z.reshape(shape)
    return z


# Define matrix inversion routine based on dimension.
# Ideally, the function would look like _mat_inv(dim) -> f(z, s=dim)
# so that the function type would be returned based on instantiation of dim.
# Numba doesn't support returning functions it seems. Thus, instead of
# returning the function type once based on instantiation, using mat_inv(z)
# as defined below will evaluate True/False everytime in a loop and return
# the correct function type.
@njit
def mat_inv(z):
    dim = z.shape[0]
    if dim == 1:
        return 1. / z
    else:
        return np.linalg.solve(z, np.eye(dim))


@njit
def is_2d(x: np.ndarray) -> bool:
    return x.ndim == 2


@njit
def is_square(x: np.ndarray) -> bool:
    return x.shape[0] == x.shape[1]


@njit
def is_odd(x: int) -> bool:
    return np.mod(x, 2) != 0


@njit
def is_symmetric(x: np.ndarray) -> bool:
    return (x == x.T).all()


@njit
def is_positive_definite(x: np.ndarray) -> bool:
    # noinspection PyBroadException
    try:
        np.linalg.cholesky(x)
        return True
    except Exception:
        return False


@njit
def is_positive_semidefinite(x: np.ndarray, tol=1e-12) -> bool:
    return np.all(np.linalg.eigvalsh(x) > -tol)
