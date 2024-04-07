import numpy as np
from typing import Union


def fourier_transform(time_index: np.ndarray,
                      periodicity: Union[int, float],
                      num_harmonics: int
                      ):
    """
    Creates a Fourier representation of a periodic/oscillating function of time.
    An ordered integer array of values capturing time is mapped to a
    matrix, where in general row R is a Fourier series representation of some
    unknown function f(t) evaluated at t=R. The matrix returned is F. It takes
    the following form:

    t=1: [cos(2 * pi * n * 1 / p), sin(2 * pi * n * 1 / p)], n = 1, 2, ..., N
    t=2: [cos(2 * pi * n * 2 / p), sin(2 * pi * n * 2 / p)], n = 1, 2, ..., N
    t=3: [cos(2 * pi * n * 3 / p), sin(2 * pi * n * 3 / p)], n = 1, 2, ..., N
    .
    .
    .
    t=T: [cos(2 * pi * n * T / p), sin(2 * pi * 1 * T / p)], n = 1, 2, ..., N

    Each row in F is of length 2N. Assuming a cycle of length P, row
    R is the same as row (P+1)R.

    The matrix F is intended to be augmented to a design matrix for regression,
    where the outcome variable is measured over time.


    Parameters
    ----------
    time_index : array
        Sequence of ordered integers representing the evolution of time.
        For example, t = [0,1,2,3,4,5, ..., T], where T is the terminal period.

    periodicity: float or int
        The amount of time it takes for a period/cycle to end. For example,
        if the frequency of data is monthly, then a period completes in 12
        months. If data is daily, then there could conceivably be two period
        lengths, one for every week (a period of length 7) and one for every
        year (a period of length 365.25). Must be positive.

    num_harmonics : integer
        The number of cosine-sine pairs to approximate oscillations in the
        variable t.

    Returns
    -------
    A T x 2N matrix of cosine-sine pairs that take the form

        cos(2 * pi * n * t / p), sin(2 * pi * n * t / p),
        t = 1, 2, ..., T
        n = 1, 2, ..., N

    """

    # Create cosine and sine input scalar factors, 2 * pi * n / (p / s), n=1,...,N
    c = 2 * np.pi * np.arange(1, num_harmonics + 1) / periodicity
    # Create cosine and sine input values, 2 * pi * n / (p / s) * t, t=1,...,T and n=1,...,N
    X = c * time_index[:, np.newaxis]
    # Pass X to cos() and sin() functions to create Fourier series
    f = np.c_[np.cos(X), np.sin(X)]

    return f
