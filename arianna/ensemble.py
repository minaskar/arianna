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
#from .autocorr import AutoCorrTime
from .moves import DifferentialMove


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
        blobs_dtype (list): List containing names and dtypes of blobs metadata e.g. ``[("log_prior", float), ("mean", float)]``. It's useful when you want to save multiple species of metadata. Default is None.
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
                 args=None,
                 kwargs=None,
                 moves=None,
                 tune=True,
                 tolerance=0.05,
                 patience=5,
                 maxsteps=10000,
                 mu=1.0,
                 maxiter=10000,
                 pool=None,
                 vectorize=False,
                 blobs_dtype=None,
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
        self.loglike_fn = _FunctionWrapper(loglike_fn, args, kwargs)
        self.logprior_fn = logprior_fn

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
        self.betas = ((np.arange(1,self.nwalkers+1)-1)/(self.nwalkers - 1))**(1/0.3)

        # Set up Slice parameters
        self.mu = mu * np.ones(self.nwalkers)
        self.mus = []
        self.mus.append(self.mu)
        self.tune = tune
        self.maxsteps = maxsteps
        self.patience = patience
        self.tolerance = tolerance
        self.nexps = []
        self.ncons =  []

        # Set up maximum number of Expansions/Contractions
        self.maxiter = maxiter

        # Set up pool of workers
        self.pool = pool
        self.vectorize = vectorize

        # Set up blobs dtype
        self.blobs_dtype = blobs_dtype

        # Initialise Saving space for samples
        self.samples = samples(self.ndim, self.nwalkers)

        # Initialise iteration counter and state
        self.iteration = 0
        self.state_X = None
        self.state_Z = None
        self.state_Zp = None
        self.state_Zl = None
        self.state_blobs = None

        # Light mode
        self.light_mode = light_mode


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


    def get_blobs(self, flat=False, thin=1, discard=0):
        """
        Get the values of the blobs at each step of the chain.

        Args:
            flat (bool) : If True then flatten the chain into a 1D array by combining all walkers (default is False).
            thin (int) : Thinning parameter (the default value is 1).
            discard (int) : Number of burn-in steps to be removed from each walker (default is 0). A float number between 0.0 and 1.0 can be used to indicate what percentage of the chain to be discarded as burnin.

        Returns:
            (structured) numpy array containing the values of the blobs at each step of the chain.
        """
        if discard < 1.0:
            discard  = int(discard * np.shape(self.chain)[0])

        if flat:
            return self.samples.flatten_blobs(discard=discard, thin=thin)
        else:
            return self.samples.blobs[discard::thin,:]


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
        Integrated Autocorrelation Time (IAT) of the Markov Chain.

        Returns:
            Array with the IAT of each parameter.
        """
        return AutoCorrTime(self.chain[int(self.nsteps/(self.thin*2.0)):,:,:])


    @property
    def ess(self):
        """
        Effective Sampling Size (ESS) of the Markov Chain.

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

    @property
    def get_last_sample(self):
        logging.warning('The ``get_last_sample`` property is deprecated and it will be removed in a future release.\n' + 'Please use the method ``get_last_sample()`` instead.')

        return self.chain[-1]

    
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

    
    def get_last_blobs(self):
        """
        Return the blobs for the last position of the walkers.
        """
        return self.samples.blobs[-1]


    @property
    def summary(self):
        """
        Summary of the MCMC run.
        """
        logging.info('Summary')
        logging.info('-------')
        logging.info('Number of Generations: ' + str(self.samples.length))
        logging.info('Number of Parameters: ' + str(self.ndim))
        logging.info('Number of Walkers: ' + str(self.nwalkers))
        logging.info('Number of Tuning Generations: ' + str(len(self.mus)))
        logging.info('Scale Factor: ' + str(round(self.mu,6)))
        logging.info('Mean Integrated Autocorrelation Time: ' + str(round(np.mean(self.act),2)))
        logging.info('Effective Sample Size: ' + str(round(self.ess,2)))
        logging.info('Number of Log Probability Evaluations: ' + str(self.ncall))
        logging.info('Effective Samples per Log Probability Evaluation: ' + str(round(self.efficiency,6)))
        if self.thin > 1:
            logging.info('Thinning rate: ' + str(self.thin))
    

    def compute_log_prob(self, coords, betas):
        """
        Calculate the vector of log-probability for the walkers

        Args:
            coords: (ndarray[..., ndim]) The position vector in parameter space where the probability should be calculated.
        Returns:
            log_prob: A vector of log-probabilities with one entry for each walker in this sub-ensemble.
            blob: The list of meta data returned by the ``log_post_fn`` at this position or ``None`` if nothing was returned.
        """
        p = coords

        # Check that the parameters are in physical ranges.
        if np.any(np.isinf(p)):
            raise ValueError("At least one parameter value was infinite")
        if np.any(np.isnan(p)):
            raise ValueError("At least one parameter value was NaN")

        # Run the log-probability calculations (optionally in parallel).
        if self.vectorize:
            results = self.loglike_fn(p)
        else:
            results = list(self.distribute(self.loglike_fn, (p[i] for i in range(len(p)))))

        log_prior = np.array(list(map(self.logprior_fn, (p[i] for i in range(len(p))))))

        try:
            log_like = np.array([float(l[0]) for l in results])
            blob = [l[1:] for l in results]
        except (IndexError, TypeError):
            log_like = np.array([float(l) for l in results])
            blob = None
        else:
            # Get the blobs dtype
            if self.blobs_dtype is not None:
                dt = self.blobs_dtype
            else:
                try:
                    dt = np.atleast_1d(blob[0]).dtype
                except ValueError:
                    dt = np.dtype("object")
            blob = np.array(blob, dtype=dt)

            # Deal with single blobs properly
            shape = blob.shape[1:]
            if len(shape):
                axes = np.arange(len(shape))[np.array(shape) == 1] + 1
                if len(axes):
                    blob = np.squeeze(blob, tuple(axes))

        log_prob = log_prior + betas * log_like

        # Check for log_prob returning NaN.
        if np.any(np.isnan(log_prob)):
            raise ValueError("Probability function returned NaN")

        return log_prob, log_prior, log_like, blob


    def run_mcmc(self,
                 start,
                 nsteps=1000,
                 thin=1,
                 progress=True,
                 log_prob0=None,
                 blobs0=None,
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
            blobs0 (float) : Blob value of the walkers. Default is ``None``.
            thin_by (float): If you only want to store and yield every
                ``thin_by`` samples in the chain, set ``thin_by`` to an
                integer greater than 1. When this is set, ``iterations *
                thin_by`` proposals will be made.
        '''
        
        for _ in self.sample(start,
                             log_prob0=log_prob0,
                             blobs0=blobs0,
                             iterations=nsteps,
                             thin=thin,
                             thin_by=thin_by,
                             progress=progress):
            pass
        

    def sample(self,
            start,
            log_prob0=None,
            blobs0=None,
            iterations=1,
            thin=1,
            thin_by=1,
            progress=True):
        '''
        Advance the chain as a generator. The current iteration index of the generator is given by the ``sampler.iteration`` property.

        Args:
            start (float) : Starting point for the walkers.
            log_prob0 (float) : Log probability values of the walkers. Default is ``None``.
            blobs0 (float) : Blob value of the walkers. Default is ``None``.
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
            if log_prob0 is None:
                Z, Zp, Zl, blobs = self.compute_log_prob(X, self.betas)
            else:
                Z = np.copy(log_prob0)
                blobs = blobs0
        elif (self.state_X is not None) and (self.state_Z is not None):
            X = np.copy(self.state_X)
            Z = np.copy(self.state_Z)
            blobs = self.state_blobs
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
            self.samples.extend(self.nsteps//self.thin, blobs)
            self.ncheckpoint = self.thin
        elif self.thin_by > 1 and self.thin == 1:
            self.nsteps = int(iterations*self.thin_by)
            self.samples.extend(self.nsteps//self.thin_by, blobs)
            self.ncheckpoint = self.thin_by
        elif self.thin == 1 and self.thin_by == 1:
            self.nsteps = int(iterations)
            self.samples.extend(self.nsteps, blobs)
            self.ncheckpoint = 1
        else:
            raise ValueError('Only one of `thin` and `thin_by` arguments can be used.')
        

        # Define Number of Log Prob Evaluations vector
        self.neval = np.zeros(self.nsteps, dtype=int)

        # Define tuning count
        ncount = 0

        # Initialise progress bar
        if progress:
            t = tqdm(total=self.nsteps, desc='Sampling progress : ')

        # Main Loop
        for i in range(self.nsteps):

            # Initialise number of expansions & contractions
            nexp = np.zeros(self.nwalkers)
            ncon = np.zeros(self.nwalkers)

            #move = np.random.choice(self._moves, p=self._weights)
            move = DifferentialMove()

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

                # Compute directions
                directions, tune_once = move.get_direction(X[inactive], self.mu[active])

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
                        Z_LR_masked, _, _, _ = self.compute_log_prob(np.concatenate([X_L[mask_J],X_R[mask_K]]))
                        Z_L[mask_J] = Z_LR_masked[:X_L[mask_J].shape[0]]
                        Z_R[mask_K] = Z_LR_masked[X_L[mask_J].shape[0]:]

                    for j in indeces[mask_J]:
                        ncall += 1
                        if Z0[j] < Z_L[j]:
                            L[j] = L[j] - 1.0
                            J[j] = J[j] - 1
                            nexp += 1
                        else:
                            mask_J[j] = False

                    for j in indeces[mask_K]:
                        ncall += 1
                        if Z0[j] < Z_R[j]:
                            R[j] = R[j] + 1.0
                            K[j] = K[j] - 1
                            nexp += 1
                        else:
                            mask_K[j] = False


                # Shrinking procedure
                Widths = np.empty(int(self.nwalkers/2))
                Z_prime = np.empty(int(self.nwalkers/2))
                Zp_prime = np.empty(int(self.nwalkers/2))
                Zl_prime = np.empty(int(self.nwalkers/2))
                X_prime = np.empty((int(self.nwalkers/2),self.ndim))
                ncon_prime = np.zeros(int(self.nwalkers/2))
                if blobs is not None:
                    blobs_prime = np.empty(int(self.nwalkers/2), dtype=np.dtype((blobs[0].dtype, blobs[0].shape)))
                mask = np.full(int(self.nwalkers/2),True)

                cnt = 0
                while len(mask[mask])>0:
                    # Update Widths of intervals
                    Widths[mask] = L[mask] + np.random.uniform(0.0,1.0,size=len(mask[mask])) * (R[mask] - L[mask])

                    # Compute New Positions
                    X_prime[mask] = directions[mask] * Widths[mask][:,np.newaxis] + X[active][mask]
                    

                    # Calculate LogP of New Positions
                    if blobs is None:
                        Z_prime[mask], Zp_prime[mask], Zl_prime[mask], _ = self.compute_log_prob(X_prime[mask], self.betas[mask])
                    else:
                        Z_prime[mask], Zp_prime[mask], Zl_prime[mask], blobs_prime[mask] = self.compute_log_prob(X_prime[mask], self.betas[mask])

                    # Count LogProb calls
                    ncall += len(mask[mask])

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
                X[active] = X_prime
                Z[active] = Z_prime
                Zp[active] = Zp_prime
                Zl[active] = Zl_prime
                if blobs is not None:
                    blobs[active] = blobs_prime
                ncon[active] = ncon_prime
                self.neval[i] += ncall

            # Tune scale factor using Robbins-Monro optimization
            if self.tune and tune_once:
                #self.nexps.append(nexp)
                #self.ncons.append(ncon)
                nexp = max(1, nexp) # This is to prevent the optimizer from getting stuck
                self.mu *= 2.0 * nexp / (nexp + ncon)
                #self.mus.append(self.mu)
                if np.abs(nexp / (nexp + ncon) - 0.5) < self.tolerance:
                    ncount += 1
                if ncount > self.patience:
                    self.tune = False
                    if self.light_mode:
                        self.mu *= (1.0 + nexp/self.nwalkers)
                        self.maxsteps = 1

            # Save samples
            if (i+1) % self.ncheckpoint == 0:
                self.samples.save(X, Z, Zp, Zl, blobs)

            # Update progress bar
            if progress:
                t.update()

            # Update iteration counter and state variables
            self.iteration = i + 1
            self.state_X = np.copy(X)
            self.state_Z = np.copy(Z)
            self.state_Zp = np.copy(Zp)
            self.state_Zl = np.copy(Zl)
            self.state_blobs = blobs

            # Yield current state
            if (i+1) % self.ncheckpoint == 0:
                yield (X, Z, Zp, Zl, blobs)

        # Close progress bar
        if progress:
            t.close()