import numpy as np 
import tqdm
import logging


class progress_bar:

    def __init__(self, nsteps, show=True):
        self.show = show
        if self.show:
            self.progress_bar = tqdm.tqdm(total=nsteps, desc='Sampling progress')

    def update(self, nexp, ncon, accept):
        
        if self.show:
            self.progress_bar.update(1)
            self.progress_bar.set_postfix(ordered_dict={'nexp':nexp,
                                                        'ncon':ncon,
                                                        'accept':accept
                                                        })

    def close(self):
        if self.show:
            self.progress_bar.close()

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record) 


def AdjustBetas(time, betas0, ratios, t0=1000.0, gamma=1.0):
    """
    Execute temperature adjustment according to dynamics outlined in
    `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_.
     """

    betas = betas0.copy()

    # Modulate temperature adjustments with a hyperbolic decay.
    kappa = t0/(time+t0)/gamma

    # Construct temperature adjustments.
    dSs = kappa * (ratios[:-1] - ratios[1:])

    # Compute new ladder (hottest and coldest chains don't move).
    deltaTs = np.diff(1 / betas[:-1])
    deltaTs *= np.exp(dSs)
    betas[1:-1] = 1 / (np.cumsum(deltaTs) + 1 / betas[0])

    return betas


def SwapReplicas(X, Zp, Zl, betas, probability):

    X = X.copy()
    Zp = Zp.copy()
    Zl = Zl.copy()

    ntemps, nwalkers, ndim = np.shape(X)

    accept = np.zeros(ntemps - 1)
    ratio = np.zeros(ntemps - 1)

    if np.random.uniform() > 1.0 - probability:

        schedule = np.arange(ntemps - 1, 0, -1)
        walkers_prev = np.arange(nwalkers)

        for j in schedule:
            np.random.shuffle(walkers_prev)
            for k in np.arange(nwalkers):
                k_prev = walkers_prev[k]
                dF = (betas[j-1]-betas[j])*(Zl[j,k]-Zl[j-1,k_prev])
                if dF < 0.0:
                    alpha = min(1, np.exp(dF))
                else:
                    alpha = 1.0
                accept[j-1] += alpha

                if alpha > np.random.uniform(0,1):
                    Zl_temp = np.copy(Zl[j-1, k_prev])
                    Zl[j-1, k_prev] = np.copy(Zl[j, k])
                    Zl[j, k] = np.copy(Zl_temp)

                    Zp_temp = np.copy(Zp[j-1, k_prev])
                    Zp[j-1, k_prev] = np.copy(Zp[j, k])
                    Zp[j, k] = np.copy(Zp_temp)         
                        
                    X_temp = np.copy(X[j-1, k_prev])
                    X[j-1, k_prev] = np.copy(X[j, k])
                    X[j, k] = np.copy(X_temp)

                    ratio[j-1] += 1.0
        
        accept /= nwalkers 
        ratio /= nwalkers

    return X, Zp, Zl, accept, ratio


def SamplePrior(nwalkers, ndim, logprior, loglike, prior_transform):
    u = np.random.rand(nwalkers, ndim)

    X = np.empty((nwalkers,ndim))
    logp = np.empty(nwalkers)
    logl = np.empty(nwalkers)

    for i in range(nwalkers):
        X[i] = prior_transform(u[i])
        logp[i] = logprior(X[i])
        if ~np.isfinite(logp[i]):
            logl[i] = -np.inf
        else:
            logl[i] = loglike(X[i])

    return X, logp, logl, logp


def TuneScaleFactor(mu, i, nexp, ncon, ncount, nwalkers, tolerance, patience, maxsteps, light_mode):
    nexp_sum = max(1, np.sum(nexp)) # This is to prevent the optimizer from getting stuck
    ncon_sum = np.sum(ncon)
    kappa = 100/(100+i+1)/1
    mu += +kappa * (2.0 * nexp_sum / (nexp_sum + ncon_sum) - 1.0)
    #self.mu *= 2.0 * nexp_sum / (nexp_sum + ncon_sum)
    #self.mu *= self.nwalkers * self.ntemps / ncon_sum
    if np.abs(nexp_sum / (nexp_sum + ncon_sum) - 0.5) < tolerance:
        ncount += 1
    #if np.abs(self.nwalkers * self.ntemps / ncon_sum - 1.0) < self.tolerance:
    #    ncount += 1
    if ncount > patience:
        tune = False
        logging.info('\nScale factor optimisation finished after %d iterations', i+1)
        if light_mode:
            mu *= (1.0 + nexp_sum/nwalkers)
            maxsteps = 1
    else:
        tune = True

    return mu, tune, maxsteps


class MeanDistance:

    def __init__(self, X0):
        ntemps, nwalkers, ndim = np.shape(X0)

        self.X0 = np.copy(X0)
        self.mean_distance = np.zeros(ntemps)
        self.k = 1.0

    def add(self, X):

        X1 = np.copy(X)
        ntemps, nwalkers, ndim = np.shape(X1)

        for i in range(nwalkers):
            self.k += 1.0
            distances = np.sqrt(np.sum((X1[:,i] - self.X0[:,i])**2, axis=1))
            self.mean_distance += (distances - self.mean_distance) / self.k

        self.X0 = X1.copy()