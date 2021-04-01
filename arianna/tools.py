import numpy as np 
from tqdm import tqdm 

class progress_bar:

    def __init__(self, nsteps, show=True):
        self.show = show
        if self.show:
            self.progress_bar = tqdm(total=nsteps, desc='Sampling progress')

    def update(self, nexp, ncon):
        
        if self.show:
            self.progress_bar.update(1)
            self.progress_bar.set_postfix(ordered_dict={'nexp':nexp,
                                                        'ncon':ncon,
                                                        })

    def close(self):
        if self.show:
            self.progress_bar.close()


def AdjustBetas(time, betas0, ratios):
    """
    Execute temperature adjustment according to dynamics outlined in
    `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_.
     """

    betas = betas0.copy()

    # Modulate temperature adjustments with a hyperbolic decay.
    kappa = 10000/(time+10000)/1.0

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
        walkers = np.arange(nwalkers)

        for j in schedule:
            np.random.shuffle(walkers_prev)
            np.random.shuffle(walkers)
            for k in walkers:
                k_prev = walkers_prev[k]
                alpha = min(1, np.exp((betas[j-1]-betas[j])*(Zl[j,k]-Zl[j-1,k_prev])))
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