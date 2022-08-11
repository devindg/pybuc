from numba import float64, vectorize
from numpy.random import normal, gamma


@vectorize([float64(float64, float64)])
def vec_norm(mean, sd):
    return normal(mean, sd)


@vectorize([float64(float64, float64)])
def vec_ig(shape, scale):
    ig = 1. / gamma(shape=shape, scale=1. / scale)
    return ig
