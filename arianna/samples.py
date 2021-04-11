import numpy as np

class samples:
    '''
    Creates object that stores the samples.
    Args:
        ndim (int): Number of dimensions/paramters
        nwalkers (int): Number of walkers.

    '''

    def __init__(self, ntemps, nwalkers, ndim):
        self.initialised = False
        self.index = 0
        self.ndim = ndim
        self.nwalkers = nwalkers
        self.ntemps = ntemps


    def extend(self, n):
        """
        Method to extend saving space.
        Args:
            n (int) : Extend space by n slots.
        """
        if self.initialised:
            ext = np.empty((n,self.ntemps,self.nwalkers,self.ndim))
            self.samples = np.concatenate((self.samples,ext),axis=0)

            ext = np.empty((n,self.ntemps,self.nwalkers))
            self.logp = np.concatenate((self.logp,ext),axis=0)

            ext = np.empty((n,self.ntemps,self.nwalkers))
            self.loglp = np.concatenate((self.loglp,ext),axis=0)

            ext = np.empty((n,self.ntemps,self.nwalkers))
            self.logll = np.concatenate((self.logll,ext),axis=0)

            ext = np.empty((n,self.ntemps))
            self.nexps = np.concatenate((self.nexps,ext),axis=0)

            ext = np.empty((n,self.ntemps))
            self.ncons = np.concatenate((self.ncons,ext),axis=0)

            ext = np.empty((n,self.ntemps-1))
            self.accept = np.concatenate((self.accept,ext),axis=0)
        else:
            self.samples = np.empty((n,self.ntemps,self.nwalkers,self.ndim))
            self.logp = np.empty((n,self.ntemps,self.nwalkers))
            self.loglp = np.empty((n,self.ntemps,self.nwalkers))
            self.logll = np.empty((n,self.ntemps,self.nwalkers))
            self.nexps = np.empty((n,self.ntemps))
            self.ncons = np.empty((n,self.ntemps))
            self.accept = np.empty((n,self.ntemps-1))
            self.initialised = True


    def save(self, x, logp, loglp, logll, nexp, ncon, accept):
        """
        Save sample into the storage.
        Args:
            x (ndarray): Samples to be appended into the storage.
            logp (ndarray): Logprob values to be appended into the storage.
        """
        self.samples[self.index] = x
        self.logp[self.index] = logp
        self.loglp[self.index] = loglp
        self.logll[self.index] = logll
        self.nexps[self.index] = nexp
        self.ncons[self.index] = ncon
        self.accept[self.index] = accept
        self.index += 1
        

    @property
    def chain(self):
        """
        Chain property.

        Returns:
            3D array of shape (nsteps,nwalkers,ndim) containing the samples.
        """
        return self.samples


    @property
    def length(self):
        """
        Number of samples per walker.

        Returns:
            The total number of samples per walker.
        """
        length, _, _, _ = np.shape(self.chain)
        return length


    def flatten(self, discard=0, thin=1):
        """
        Flatten samples by thinning them, removing the burn in phase, and combining all the walkers.

        Args:
            discard (int): Number of burn-in steps to be removed from each walker (default is 0).
            thin (int): Thinning parameter (the default value is 1).

        Returns:
            2D object containing the ndim flattened chains.
        """
        return self.chain[discard::thin,:,:].reshape((-1,self.ndim), order='F')


    @property
    def logprob(self):
        """
        Chain property.

        Returns:
            2D array of shape (nwalkers,nsteps) containing the log-probabilities.
        """
        return self.logp


    def flatten_logprob(self, discard=0, thin=1):
        """
        Flatten log probability by thinning the chain, removing the burn in phase, and combining all the walkers.

        Args:
            discard (int): Number of burn-in steps to be removed from each walker (default is 0).
            thin (int): Thinning parameter (the default value is 1).

        Returns:
            1D object containing the logprob of the flattened chains.
        """
        return self.logprob[discard::thin,:].reshape((-1,), order='F')