import numpy as np 
from itertools import permutations
import random


class DifferentialMove:
    r"""
    The `Karamanis & Beutler (2020) <https://arxiv.org/abs/2002.06212>`_ "Differential Move" with parallelization.
    When this Move is used the walkers move along directions defined by random pairs of walkers sampled (with no
    replacement) from the complementary ensemble. This is the default choice and performs well along a wide range
    of target distributions.

    Parameters
    ----------
        tune : bool
            If True then tune this move. Default is True.
        mu0 : float
            Default value of ``mu`` if ``tune=False``.

    """

    def __init__(self, tune=True, mu0=1.0):
        self.tune = tune
        self.mu0 = mu0


    def get_direction(self, X, mu):
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

        if not self.tune:
            mu = self.mu0

        diff = (X[pairs[0]]-X[pairs[1]])
        diff /= np.linalg.norm(diff, axis=1)
        
        return 2.0 * mu * diff, self.tune