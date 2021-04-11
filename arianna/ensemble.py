import numpy as np
from tqdm import tqdm
import logging

try:
    from collections.abc import Iterable
except ImportError:
    # for py2.7, will be an Exception in 3.8
    from collections import Iterable

from .samples import samples
from .fwrapper import _LogPosteriorWrapper, _FunctionWrapper
from .autocorr import AutoCorrTime
from .moves import get_direction, get_past_direction
from .tools import progress_bar, AdjustBetas, SwapReplicas, TqdmLoggingHandler, SamplePrior, TuneScaleFactor, MeanDistance
from .slice import Slice



class ReplicaExchangeSampler:
    """
    A Replica Exchange Slice Sampler.

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
                 ntemps,
                 nwalkers,
                 ndim,
                 loglike_fn,
                 logprior_fn,
                 loglike_args=None,
                 loglike_kwargs=None,
                 logprior_args=None,
                 logprior_kwargs=None,
                 prior_transform=None,
                 betas=None,
                 swap=1.0,
                 tune=False,
                 tolerance=0.05,
                 patience=5,
                 maxsteps=10000,
                 mu=1.4,
                 maxiter=10000,
                 pool=None,
                 vectorize=False,
                 verbose=True,
                 check_walkers=True,
                 shuffle_ensemble=True,
                 light_mode=False,
                 adapt=False,
                 t0=1000.0,
                 gamma=1.0,
                 hist=0.0):

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
        #self.logger.addHandler(TqdmLoggingHandler())

        # Set up Log Probability
        self.logprob_fn = _LogPosteriorWrapper(loglike_fn, loglike_args, loglike_kwargs, logprior_fn, logprior_args, logprior_kwargs)
        self.logprior_fn = _FunctionWrapper(logprior_fn, logprior_args, logprior_kwargs)
        self.loglike_fn = _FunctionWrapper(loglike_fn, loglike_args, loglike_kwargs)

        # Set up walkers
        self.ntemps = int(ntemps)
        self.nwalkers = int(nwalkers)
        self.ndim = int(ndim)
        self.check_walkers = check_walkers
        if self.check_walkers:
            if self.ntemps * self.nwalkers < 2 * self.ndim:
                raise ValueError("Please provide at least (2 * ndim) walkers.")
            elif self.ntemps * self.nwalkers % 2 == 1:
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
        self.swap = swap
        if betas is None:
            self.betas = ((np.arange(1,self.ntemps+1)-1)/(self.ntemps - 1))**(1/0.3)
            self.betas = self.betas[::-1]
        else:
            self.betas = betas
        self.betas_flat = np.repeat(self.betas,self.nwalkers)

        # Set up pool of workers
        self.pool = pool
        self.vectorize = vectorize

        # Initialise Saving space for samples
        self.samples = samples(self.ntemps, self.nwalkers, self.ndim)

        # Initialise iteration counter and state
        self.iteration = 0
        self.state_X = None
        self.state_Z = None
        self.state_Zp = None
        self.state_Zl = None

        # Light mode
        self.light_mode = light_mode
        self.adapt = adapt
        self.t0 = t0
        self.gamma = gamma
        self.hist = hist

        self.prior_transform = prior_transform

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
            if np.shape(start) != (self.ntemps, self.nwalkers, self.ndim):
                raise ValueError('Incompatible input dimensions! \n' +
                                 'Please provide array of shape (nwalkers, ndim) as the starting position.')
            X = np.copy(start)
            if log_like0 is None:
                Z, Zp, Zl, ncall = self.compute_log_prob(X, self.betas)
            else:
                Zl = np.copy(log_like0)
                Zp = np.copy(log_prior0)
                Z = Zp + self.betas[:,np.newaxis] * Zl
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
        self.betas_all = []
        #self.mean_estimator = MeanDistance(np.copy(X))
        self.mean_distance = np.ones(self.ntemps)

        batch = np.empty((self.ntemps,self.nwalkers),dtype=int)
        for i in range(self.ntemps):
            batch[i] = np.arange(self.nwalkers,dtype=int)


        # Main Loop
        for i in range(self.nsteps):

            # Initialise number of expansions & contractions
            nexp = np.zeros((self.ntemps,self.nwalkers))
            ncon = np.zeros((self.ntemps,self.nwalkers))
            D = np.zeros((self.ntemps,self.nwalkers))

            # Shuffle and split ensemble
            for b in batch:
                np.random.shuffle(b)
            batch0 = batch[:,:self.nwalkers//2]
            batch1 = batch[:,self.nwalkers//2:]
            sets = [[batch0,batch1],[batch1,batch0]]

            # Loop over two sets
            for ensembles in sets:
                # Define active-inactive ensembles
                active, inactive = ensembles                

                X, Z, Zp, Zl, nexp, ncon, ncall, D = Slice(self.compute_log_prob, X, Z, Zp, Zl, nexp, ncon, D,
                                                        self.betas, active, inactive, self.mu,
                                                        self.mean_distance,
                                                        self.maxsteps, self.maxiter, i)



                self.neval[i] += ncall
            
            self.mean_distance += (np.mean(D, axis=1) - self.mean_distance) / (i+1)

            if self.prior_transform is not None:
                X[-1], Zp[-1], Zl[-1], Z[-1] = SamplePrior(self.nwalkers, self.ndim, self.logprior_fn, self.loglike_fn, self.prior_transform)

            # Swap temperatures
            X, Zp, Zl, accept, ratio = SwapReplicas(X, Zp, Zl, self.betas, self.swap)

            if self.adapt:
                self.betas = AdjustBetas(i+1, self.betas, accept, self.t0, self.gamma)
                self.betas_all.append(self.betas.copy())

            Z = Zl * self.betas[:,np.newaxis] + Zp

            # Tune scale factor using Robbins-Monro optimization
            if self.tune:
                self.mu, self.tune, self.maxsteps = TuneScaleFactor(self.mu, i, nexp, ncon, ncount, self.nwalkers, self.tolerance, self.patience, self.maxsteps, self.light_mode)
                self.mus.append(self.mu)

            # Save samples
            if (i+1) % self.ncheckpoint == 0:
                self.samples.save(X, Z, Zp, Zl, np.sum(nexp,axis=1), np.sum(ncon,axis=1), accept)

            # Update progress bar
            pbar.update(np.sum(nexp), np.sum(ncon), np.mean(accept))

            # Update iteration counter and state variables
            self.iteration = i + 1
            self.state_X = np.copy(X)
            self.state_Z = np.copy(Z)
            self.state_Zp = np.copy(Zp)
            self.state_Zl = np.copy(Zl)

            # Yield current state
            if (i+1) % self.ncheckpoint == 0:
                yield (X, Z, Zp, Zl)

        # Close progress bar
        pbar.close()


    def reset(self):
        """
        Reset the state of the sampler. Delete any samples stored in memory.
        """
        self.samples = samples(self.ntemps, self.nwalkers, self.ndim)
    

    def get_chain(self, flat=False, thin=1, discard=0, warm=False):
        """
        Get the Markov chain containing the samples.

        Args:
            flat (bool) : If True then flatten the chain into a 2D array by combining all walkers (default is False).
            thin (int) : Thinning parameter (the default value is 1).
            discard (int) : Number of burn-in steps to be removed from each walker (default is 0). A float number between
                0.0 and 1.0 can be used to indicate what percentage of the chain to be discarded as burnin.
            hot (bool) : If True then return warm chains too, else return only the cold (beta=1) chain (default is False).

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
        Effective Sampling Size (ESS) of the ``beta = 1`` Markov chain walkers.

        Returns:
            ESS
        """
        return self.nwalkers * self.samples.length / np.mean(self.act)


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

    
    def get_logz(self, discard=0.5, correction=False):
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

        A = np.mean(self.samples.logll[discard:,1:], axis=(0,2))
        B = np.mean(self.samples.logll[discard:,:-1], axis=(0,2))

        dbeta = np.diff(self.betas)

        logz = -np.sum(dbeta*(A+B)/2.0)

        if correction:
            C = np.var(self.samples.logll[discard:,1:], axis=(0,2))
            D = np.var(self.samples.logll[discard:,:-1], axis=(0,2))
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

        if coords.ndim == 3:
            ntemps, nwalkers, ndim = np.shape(coords)
            p = coords.reshape(-1, ndim)
            betas = np.repeat(betas, nwalkers)
        else:
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

        if coords.ndim == 3:
            return log_prob.reshape(ntemps,nwalkers), log_prior.reshape(ntemps,nwalkers), log_like.reshape(ntemps,nwalkers), ncall
        else:
            return log_prob, log_prior, log_like, ncall