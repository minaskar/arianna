import numpy as np
from tqdm import tqdm
import logging

try:
    from collections.abc import Iterable
except ImportError:
    # for py2.7, will be an Exception in 3.8
    from collections import Iterable

from .samples import samples
from .fwrapper import _FunctionWrapper
from .fwrapper import _LogPosteriorWrapper
from .autocorr import AutoCorrTime
from .moves import get_direction, get_direction2
from .tools import progress_bar, OnlineCovariance

class EnsembleSampler:
    """
    A Metropolis--Coupled Slice Sampler.

    Args:
        nwalkers (int): The number of walkers in the ensemble.
        ndim (int): The number of dimensions/parameters.
        loglike_fn (callable): A python function that takes a vector in the
            parameter space as input and returns the natural logarithm of the
            likelihood function at that position.
        logprior_fn (callable): A python function that takes a vector in the
            parameter space as input and returns the natural logarithm of the
            prior pdf at that position.
        args (list): Extra arguments to be passed into the logp.
        kwargs (list): Extra arguments to be passed into the logp.
        tune (bool): Tune the scale factor to optimize performance (Default is True.)
        tolerance (float): Tuning optimization tolerance (Default is 0.05).
        patience (int): Number of tuning steps to wait to make sure that tuning is done (Default is 5).
        maxsteps (int): Number of maximum stepping-out steps (Default is 10^4).
        mu (float): Scale factor (Default value is 1.0), this will be tuned if tune=True.
        maxiter (int): Number of maximum Expansions/Contractions (Default is 10^4).
        pool (bool): External pool of workers to distribute workload to multiple CPUs (default is None).
        vectorize (bool): If true (default is False), logprob_fn receives not just one point but an array of points, and returns an array of likelihoods.
        verbose (bool): If True (default) print log statements.
        check_walkers (bool): If True (default) then check that ``nwalkers >= 2*ndim`` and even.
        shuffle_ensemble (bool): If True (default) then shuffle the ensemble of walkers in every iteration before splitting it.
        light_mode (bool): If True (default is False) then no expansions are performed after the tuning phase. This can significantly reduce the number of log likelihood evaluations but works best in target distributions that are apprroximately Gaussian.
    """
    def __init__(self,
                 nwalkers,
                 ndim,
                 loglike_fn,
                 logprior_fn,
                 loglike_args=None,
                 loglike_kwargs=None,
                 logprior_args=None,
                 logprior_kwargs=None,
                 betas=None,
                 alpha=1.0,
                 tune=True,
                 tolerance=0.05,
                 patience=5,
                 maxsteps=10000,
                 mu=1.0,
                 maxiter=10000,
                 pool=None,
                 vectorize=False,
                 verbose=True,
                 check_walkers=True,
                 shuffle_ensemble=True,
                 light_mode=False):

        # Set up logger
        self.logger = logging.getLogger()
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        handler = logging.StreamHandler()
        self.logger.addHandler(handler)
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

        # Set up Log Probability
        self.logprob_fn = _LogPosteriorWrapper(loglike_fn, loglike_args, loglike_kwargs, logprior_fn, logprior_args, logprior_kwargs)

        # Set up walkers
        self.nwalkers = int(nwalkers)
        self.ndim = int(ndim)
        self.check_walkers = check_walkers
        if self.check_walkers:
            if self.nwalkers < 2 * self.ndim:
                raise ValueError("Please provide at least (2 * ndim) walkers.")
            elif self.nwalkers % 2 == 1:
                raise ValueError("Please provide an even number of walkers.")
        self.shuffle_ensemble = shuffle_ensemble

        # Set up Slice parameters
        self.mu = mu
        self.mus = []
        self.mus.append(self.mu)
        self.tune = tune
        self.maxsteps = maxsteps
        self.patience = patience
        self.tolerance = tolerance

        # Set up maximum number of Expansions/Contractions
        self.maxiter = maxiter

        # Set up temperature swap frequency
        self.alpha = alpha
        if betas is None:
            self.betas = ((np.arange(1,self.nwalkers+1)-1)/(self.nwalkers - 1))**(1/0.3)
            self.betas = self.betas[::-1]
        else:
            self.betas = betas

        # Set up pool of workers
        self.pool = pool
        self.vectorize = vectorize

        # Initialise Saving space for samples
        self.samples = samples(self.ndim, self.nwalkers)

        # Initialise iteration counter and state
        self.iteration = 0
        self.state_X = None
        self.state_Z = None
        self.state_Zp = None
        self.state_Zl = None
        self.state_idx = None

        # Light mode
        self.light_mode = light_mode


    def run_mcmc(self,
                 start,
                 nsteps=1000,
                 thin=1,
                 progress=True,
                 log_like0=None,
                 log_prior0=None,
                 thin_by=1):
        '''
        Run MCMC.

        Args:
            start (float) : Starting point for the walkers. If ``None`` then the sampler proceeds
                from the last known position of the walkers.
            nsteps (int): Number of steps/generations (default is 1000).
            thin (float): Thin the chain by this number (default is 1, no thinning).
            progress (bool): If True (default), show progress bar.
            log_prob0 (float) : Log probability values of the walkers. Default is ``None``.
            thin_by (float): If you only want to store and yield every
                ``thin_by`` samples in the chain, set ``thin_by`` to an
                integer greater than 1. When this is set, ``iterations *
                thin_by`` proposals will be made.
        '''
        
        for _ in self.sample(start,
                             log_like0=log_like0,
                             log_prior0=log_prior0,
                             iterations=nsteps,
                             thin=thin,
                             thin_by=thin_by,
                             progress=progress):
            pass


    def sample(self,
            start,
            log_like0=None,
            log_prior0=None,
            iterations=1,
            thin=1,
            thin_by=1,
            progress=True):
        '''
        Advance the chain as a generator. The current iteration index of the generator is given by the ``sampler.iteration`` property.

        Args:
            start (float) : Starting point for the walkers.
            log_prob0 (float) : Log probability values of the walkers. Default is ``None``.
            iterations (int): Number of steps to generate (default is 1).
            thin (float): Thin the chain by this number (default is 1, no thinning).
            thin_by (float): If you only want to store and yield every
                ``thin_by`` samples in the chain, set ``thin_by`` to an
                integer greater than 1. When this is set, ``iterations *
                thin_by`` proposals will be made.
            progress (bool): If True (default), show progress bar.
        '''
        # Define task distributer
        if self.pool is None:
            self.distribute = map
        else:
            self.distribute = self.pool.map

        # Initialise ensemble of walkers
        logging.info('Initialising ensemble of %d walkers...', self.nwalkers)
        if start is not None:
            if np.shape(start) != (self.nwalkers, self.ndim):
                raise ValueError('Incompatible input dimensions! \n' +
                                 'Please provide array of shape (nwalkers, ndim) as the starting position.')
            X = np.copy(start)
            if log_like0 is None:
                Z, Zp, Zl, ncall = self.compute_log_prob(X, self.betas)
                idx = np.arange(self.nwalkers)
            else:
                Zl = np.copy(log_like0)
                Zp = np.copy(log_prior0)
                Z = Zp + self.betas * Zl
                ncall = 0
        elif (self.state_X is not None) and (self.state_Z is not None):
            X = np.copy(self.state_X)
            Z = np.copy(self.state_Z)
            Zp = np.copy(self.state_Zp)
            Zl = np.copy(self.state_Zl)
            ncall = 0
        else:
            raise ValueError("Cannot have `start=None` if run_mcmc has never been called before.")


        if not np.all(np.isfinite(Z)):
            raise ValueError('Invalid walker initial positions! \n' +
                             'Initialise walkers from positions of finite log probability.')
        batch = list(np.arange(self.nwalkers))

        # Extend saving space
        self.thin = int(thin)
        self.thin_by = int(thin_by)
        
        if self.thin_by < 0:
            raise ValueError('Invalid `thin_by` argument.')
        elif self.thin < 0:
            raise ValueError('Invalid `thin` argument.')
        elif self.thin > 1 and self.thin_by == 1:
            self.nsteps = int(iterations)
            self.samples.extend(self.nsteps//self.thin)
            self.ncheckpoint = self.thin
        elif self.thin_by > 1 and self.thin == 1:
            self.nsteps = int(iterations*self.thin_by)
            self.samples.extend(self.nsteps//self.thin_by)
            self.ncheckpoint = self.thin_by
        elif self.thin == 1 and self.thin_by == 1:
            self.nsteps = int(iterations)
            self.samples.extend(self.nsteps)
            self.ncheckpoint = 1
        else:
            raise ValueError('Only one of `thin` and `thin_by` arguments can be used.')
        

        # Define Number of Log Prob Evaluations vector
        self.neval = np.zeros(self.nsteps, dtype=int)

        # Define tuning count
        ncount = 0

        # Initialise progress bar
        pbar = progress_bar(self.nsteps, show=progress)

        # keep distances
        self.distances = []

        # Main Loop
        for i in range(self.nsteps):

            distance = np.zeros(self.nwalkers)

            # Initialise number of expansions & contractions
            nexp = np.zeros(self.nwalkers)
            ncon = np.zeros(self.nwalkers)

            # Shuffle and split ensemble
            if self.shuffle_ensemble:
                np.random.shuffle(batch)
            batch0 = batch[:int(self.nwalkers/2)]
            batch1 = batch[int(self.nwalkers/2):]
            sets = [[batch0,batch1],[batch1,batch0]]

            # Loop over two sets
            for ensembles in sets:
                indeces = np.arange(int(self.nwalkers/2))
                # Define active-inactive ensembles
                active, inactive = ensembles

                betas_active = self.betas[active]

                # Compute directions
                directions = get_direction(X[inactive], self.mu)

                if i > 2:
                    directions *= np.mean(np.array(self.distances)[:,active],axis=0)[:,np.newaxis]

                # Get Z0 = LogP(x0)
                Z0 = Z[active] - np.random.exponential(size=int(self.nwalkers/2))

                # Set Initial Interval Boundaries
                L = - np.random.uniform(0.0,1.0,size=int(self.nwalkers/2))
                R = L + 1.0

                # Parallel stepping-out
                J = np.floor(self.maxsteps * np.random.uniform(0.0,1.0,size=int(self.nwalkers/2)))
                K = (self.maxsteps - 1) - J

                # Initialise number of Log prob calls
                ncall = 0

                # Left stepping-out initialisation
                mask_J = np.full(int(self.nwalkers/2),True)
                Z_L = np.empty(int(self.nwalkers/2))
                X_L = np.empty((int(self.nwalkers/2),self.ndim))

                # Right stepping-out initialisation
                mask_K = np.full(int(self.nwalkers/2),True)
                Z_R = np.empty(int(self.nwalkers/2))
                X_R = np.empty((int(self.nwalkers/2),self.ndim))

                nexp_prime = np.zeros(int(self.nwalkers/2))

                cnt = 0
                # Stepping-Out procedure
                while len(mask_J[mask_J])>0 or len(mask_K[mask_K])>0:
                    if len(mask_J[mask_J])>0:
                        cnt += 1
                    if len(mask_K[mask_K])>0:
                        cnt += 1
                    if cnt > self.maxiter:
                        raise RuntimeError('Number of expansions exceeded maximum limit! \n' +
                                           'Make sure that the pdf is well-defined. \n' +
                                           'Otherwise increase the maximum limit (maxiter=10^4 by default).')

                    for j in indeces[mask_J]:
                        if J[j] < 1:
                            mask_J[j] = False

                    for j in indeces[mask_K]:
                        if K[j] < 1:
                            mask_K[j] = False

                    X_L[mask_J] = directions[mask_J] * L[mask_J][:,np.newaxis] + X[active][mask_J]
                    X_R[mask_K] = directions[mask_K] * R[mask_K][:,np.newaxis] + X[active][mask_K]

                    if len(X_L[mask_J]) + len(X_R[mask_K]) < 1:
                        Z_L[mask_J] = np.array([])
                        Z_R[mask_K] = np.array([])
                        cnt -= 1
                    else:
                        Z_LR_masked, _, _, ncall_new = self.compute_log_prob(np.concatenate([X_L[mask_J],X_R[mask_K]]),np.concatenate([betas_active[mask_J],betas_active[mask_K]]))
                        Z_L[mask_J] = Z_LR_masked[:X_L[mask_J].shape[0]]
                        Z_R[mask_K] = Z_LR_masked[X_L[mask_J].shape[0]:]
                    
                    ncall += ncall_new

                    for j in indeces[mask_J]:
                        if Z0[j] < Z_L[j]:
                            L[j] = L[j] - 1.0
                            J[j] = J[j] - 1
                            nexp_prime[j] += 1
                        else:
                            mask_J[j] = False

                    for j in indeces[mask_K]:
                        if Z0[j] < Z_R[j]:
                            R[j] = R[j] + 1.0
                            K[j] = K[j] - 1
                            nexp_prime[j] += 1
                        else:
                            mask_K[j] = False


                # Shrinking procedure
                Widths = np.empty(int(self.nwalkers/2))
                Z_prime = np.empty(int(self.nwalkers/2))
                Zp_prime = np.empty(int(self.nwalkers/2))
                Zl_prime = np.empty(int(self.nwalkers/2))
                X_prime = np.empty((int(self.nwalkers/2),self.ndim))
                ncon_prime = np.zeros(int(self.nwalkers/2))
                mask = np.full(int(self.nwalkers/2),True)

                cnt = 0
                while len(mask[mask])>0:
                    # Update Widths of intervals
                    Widths[mask] = L[mask] + np.random.uniform(0.0,1.0,size=len(mask[mask])) * (R[mask] - L[mask])

                    # Compute New Positions
                    X_prime[mask] = directions[mask] * Widths[mask][:,np.newaxis] + X[active][mask]
                    

                    # Calculate LogP of New Positions
                    Z_prime[mask], Zp_prime[mask], Zl_prime[mask], ncall_new = self.compute_log_prob(X_prime[mask], betas_active[mask])

                    # Count LogProb calls
                    #ncall += len(mask[mask])
                    ncall += ncall_new

                    # Shrink slices
                    for j in indeces[mask]:
                        if Z0[j] < Z_prime[j]:
                            mask[j] = False
                        else:
                            if Widths[j] < 0.0:
                                L[j] = Widths[j]
                                ncon_prime[j] += 1
                            elif Widths[j] > 0.0:
                                R[j] = Widths[j]
                                ncon_prime[j] += 1

                    cnt += 1
                    if cnt > self.maxiter:
                        raise RuntimeError('Number of contractions exceeded maximum limit! \n' +
                                           'Make sure that the pdf is well-defined. \n' +
                                           'Otherwise increase the maximum limit (maxiter=10^4 by default).')

                # Update Positions
                distance[active] = np.copy(np.sqrt(np.sum((X_prime - X[active])**2,axis=1)))

                X[active] = X_prime
                Z[active] = Z_prime
                Zp[active] = Zp_prime
                Zl[active] = Zl_prime
                ncon[active] = ncon_prime
                nexp[active] = nexp_prime
                self.neval[i] += ncall
            
            self.distances.append(distance)

            # Swap temperatures
            if np.random.uniform() > 1.0 - self.alpha:

                schedule = np.arange(self.nwalkers - 1, 0, -1)

                accept = np.empty(self.nwalkers - 1)

                for j in schedule:
                    
                    alpha = min(1, np.exp((self.betas[j-1]-self.betas[j])*(Zl[j]-Zl[j-1])))

                    accept[j-1] = alpha

                    if alpha > np.random.uniform(0,1):
                        Zl_temp = np.copy(Zl[j-1])
                        Zl[j-1] = np.copy(Zl[j])
                        Zl[j] = np.copy(Zl_temp)

                        Zp_temp = np.copy(Zp[j-1])
                        Zp[j-1] = np.copy(Zp[j])
                        Zp[j] = np.copy(Zp_temp)         
                        
                        X_temp = np.copy(X[j-1])
                        X[j-1] = np.copy(X[j])
                        X[j] = np.copy(X_temp)

                        idx_temp = np.copy(idx[j-1])
                        idx[j-1] = np.copy(idx[j])
                        idx[j] = np.copy(idx_temp)

                        Z[j-1] = Zl[j-1] * self.betas[j-1] + Zp[j-1]
                        Z[j] = Zl[j] * self.betas[j] + Zp[j]


            # Tune scale factor using Robbins-Monro optimization
            if self.tune:
                nexp_sum = max(1, np.sum(nexp)) # This is to prevent the optimizer from getting stuck
                ncon_sum = np.sum(ncon)
                self.mu *= 2.0 * nexp_sum / (nexp_sum + ncon_sum)
                if np.abs(nexp_sum / (nexp_sum + ncon_sum) - 0.5) < self.tolerance:
                    ncount += 1
                if ncount > self.patience:
                    self.tune = False
                    if self.light_mode:
                        self.mu *= (1.0 + nexp_sum/self.nwalkers)
                        self.maxsteps = 1
                #self.mus.append(np.copy(self.mu))
                self.mus.append(self.mu)

            # Save samples
            if (i+1) % self.ncheckpoint == 0:
                self.samples.save(X, Z, Zp, Zl, idx, nexp, ncon, accept)

            # Update progress bar
            pbar.update(np.sum(nexp), np.sum(ncon))

            # Update iteration counter and state variables
            self.iteration = i + 1
            self.state_X = np.copy(X)
            self.state_Z = np.copy(Z)
            self.state_Zp = np.copy(Zp)
            self.state_Zl = np.copy(Zl)
            self.state_idx = np.copy(idx)

            # Yield current state
            if (i+1) % self.ncheckpoint == 0:
                yield (X, Z, Zp, Zl, idx)

        # Close progress bar
        pbar.close()


    def reset(self):
        """
        Reset the state of the sampler. Delete any samples stored in memory.
        """
        self.samples = samples(self.ndim, self.nwalkers)
    

    def get_chain(self, flat=False, thin=1, discard=0):
        """
        Get the Markov chain containing the samples.

        Args:
            flat (bool) : If True then flatten the chain into a 2D array by combining all walkers (default is False).
            thin (int) : Thinning parameter (the default value is 1).
            discard (int) : Number of burn-in steps to be removed from each walker (default is 0). A float number between
                0.0 and 1.0 can be used to indicate what percentage of the chain to be discarded as burnin.

        Returns:
            Array object containg the Markov chain samples (2D if flat=True, 3D if flat=False).
        """

        if discard < 1.0:
            discard  = int(discard * np.shape(self.chain)[0])

        if flat:
            return self.samples.flatten(discard=discard, thin=thin)
        else:
            return self.chain[discard::thin,:,:]

    
    def get_log_prob(self, flat=False, thin=1, discard=0):
        """
        Get the value of the log probability function evalutated at the samples of the Markov chain.

        Args:
            flat (bool) : If True then flatten the chain into a 1D array by combining all walkers (default is False).
            thin (int) : Thinning parameter (the default value is 1).
            discard (int) : Number of burn-in steps to be removed from each walker (default is 0). A float number between
                0.0 and 1.0 can be used to indicate what percentage of the chain to be discarded as burnin.

        Returns:
            Array containing the value of the log probability at the samples of the Markov chain (1D if flat=True, 2D otherwise).
        """
        if discard < 1.0:
            discard  = int(discard * np.shape(self.chain)[0])

        if flat:
            return self.samples.flatten_logprob(discard=discard, thin=thin)
        else:
            return self.samples.logprob[discard::thin,:]


    @property
    def chain(self):
        """
        Returns the chains.

        Returns:
            Returns the chains of shape (nsteps, nwalkers, ndim).
        """
        return self.samples.chain


    @property
    def act(self):
        """
        Integrated Autocorrelation Time (IAT) of the ``beta = 1`` Markov Chain.

        Returns:
            Array with the IAT of each parameter.
        """
        return AutoCorrTime(self.chain[int(self.nsteps/(self.thin*2.0)):,0,:])


    @property
    def ess(self):
        """
        Effective Sampling Size (ESS) of the ``beta = 1``  Markov Chain.

        Returns:
            ESS
        """
        return self.samples.length / np.mean(self.act)


    @property
    def ncall(self):
        """
        Number of Log Prob calls.

        Returns:
            ncall
        """
        return np.sum(self.neval)


    @property
    def efficiency(self):
        """
        Effective Samples per Log Probability Evaluation.

        Returns:
            efficiency
        """
        return self.ess / self.ncall


    @property
    def scale_factor(self):
        """
        Scale factor values during tuning.

        Returns:
            scale factor mu
        """
        return np.asarray(self.mus)

    
    def get_last_sample(self):
        """
        Return the last position of the walkers.
        """
        return self.chain[-1]

    
    def get_last_log_prob(self):
        """
        Return the log probability values for the last position of the walkers.
        """
        return self.samples.logprob[-1]

    
    def get_logz(self, discard=0.5, correction=True):
        """
        Calculate the natural logarithm of model evidence (aka marginal likelihood) logZ using the trapezoidal rule.

        Args:
            discard: (float or int) Number of burn-in steps to be removed from each walker (default is 0).
                A float number between 0.0 and 1.0 can be used to indicate what percentage of the chain to be discarded as burnin.
            correction: (bool) If True (default) then the result is calculated using the *Frier et al. (2014)* corrected trapezoidal
                rule that takes into account the variance of the log-likelihood.
        Returns:
            (float) log model evidence
        """

        if discard < 1.0:
            discard  = int(discard * np.shape(self.chain)[0])

        A = np.mean(self.samples.logll[discard:,1:], axis=0)
        B = np.mean(self.samples.logll[discard:,:-1], axis=0)

        dbeta = np.diff(self.betas)

        logz = np.sum(dbeta*(A+B)/2.0)

        if correction:
            C = np.var(self.samples.logll[discard:,1:], axis=0)
            D = np.var(self.samples.logll[discard:,:-1], axis=0)
            logz -= np.sum(dbeta**2.0 * (C-D)/12.0)
        return logz
    

    def compute_log_prob(self, coords, betas):
        """
        Calculate the vector of log-probability for the walkers

        Args:
            coords: (ndarray[..., ndim]) The position vector in parameter space where the probability should be calculated.
            betas: (ndarray[...]) The beta values for each walker in ``coords``.
        Returns:
            log_prob: A vector of log-probabilities with one entry for each walker in this sub-ensemble.
            log_prior: A vector of log-prior values with one entry for each walker in this sub-ensemble.
            log_like: A vector of log-likelihood values with one entry for each walker in this sub-ensemble.
        """
        p = coords

        # Check that the parameters are in physical ranges.
        if np.any(np.isinf(p)):
            raise ValueError("At least one parameter value was infinite")
        if np.any(np.isnan(p)):
            raise ValueError("At least one parameter value was NaN")

        # Run the log-probability calculations (optionally in parallel).
        if self.vectorize:
            results = self.logprob_fn(p)
        else:
            results = list(self.distribute(self.logprob_fn, (p[i] for i in range(len(p)))))

        log_prior = np.array([float(l[0]) for l in results])
        log_like = np.array([float(l[1]) for l in results])

        mask = np.isfinite(log_like)

        log_prob = np.empty(len(p))

        log_prob[mask] = log_prior[mask] + betas[mask] * log_like[mask]
        log_prob[~mask] = -np.inf

        ncall = len(log_prob[mask])

        # Check for log_prob returning NaN.
        if np.any(np.isnan(log_prob)):
            raise ValueError("Probability function returned NaN")

        return log_prob, log_prior, log_like, ncall