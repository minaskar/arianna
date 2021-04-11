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


def get_past_direction(samples, active, mu, normalise=True):
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

    history = samples.samples[:samples.index-1]

    nsteps, ntemps, nwalkers, ndim = np.shape(history)

    diff = np.empty((ntemps, nwalkers, ndim))

    for t in range(ntemps):
        for w in range(nwalkers):
            x, y = np.random.choice(nsteps*nwalkers,2,replace=False)
            if x <= nsteps - 1:
                xi = x
                xj = 0
            else:
                xi = x % nsteps
                xj = x // nsteps

            if y <= nsteps - 1:
                yi = y
                yj = 0
            else:
                yi = y % nsteps
                yj = y // nsteps

            diff[t,w] = history[xi,t,xj] - history[yi,t,yj]
            
    if normalise:
        diff /= np.linalg.norm(diff, axis=1)[:,np.newaxis]
        
    return 2.0 * mu * diff.reshape(-1,ndim)[active]