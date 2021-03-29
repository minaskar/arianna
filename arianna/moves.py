import numpy as np 
from itertools import permutations
import random

def get_direction(X, mu):
    r"""
    Generate direction vectors.

    Parameters
    ----------
        X : array
            Array of shape ``(nwalkers//2, ndim)`` with the walker positions of the complementary ensemble.
        mu : float
            The value of the scale factor ``mu``.
        
    Returns
    -------
        directions : array
            Array of direction vectors of shape ``(nwalkers//2, ndim)``.
    """

    nsamples = X.shape[0]

    perms = list(permutations(np.arange(nsamples), 2))
    pairs = np.asarray(random.sample(perms,nsamples)).T

    diff = (X[pairs[0]]-X[pairs[1]])
    diff /= np.linalg.norm(diff, axis=1)[:,np.newaxis]
        
    return 2.0 * mu * diff


def get_direction2(X, mu, betas):
    r"""
    Generate direction vectors.

    Parameters
    ----------
        X : array
            Array of shape ``(nwalkers//2, ndim)`` with the walker positions of the complementary ensemble.
        mu : float
            The value of the scale factor ``mu``.
        
    Returns
    -------
        directions : array
            Array of direction vectors of shape ``(nwalkers//2, ndim)``.
    """

    nsamples = X.shape[0]

    perms = list(permutations(np.arange(nsamples), 2))
    pairs = np.asarray(random.sample(perms,nsamples)).T

    bpb = (betas[pairs[0]]**0.5+betas[pairs[1]]**0.5)

    diff = (X[pairs[0]]-X[pairs[1]]) * bpb[:,np.newaxis]
    #diff /= np.linalg.norm(diff, axis=1)[:,np.newaxis]
        
    return 2.0 * mu[:,np.newaxis] * diff