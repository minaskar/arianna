import numpy as np
from itertools import permutations
import random


def SteppingOut(logprob, X_active, Z_active, betas_active, directions, indeces,ntemps,nwalkers,ndim,maxsteps,maxiter):
    # Get Z0 = LogP(x0)
    Z0 = Z_active - np.random.exponential(size=int(ntemps*nwalkers/2))

    # Set Initial Interval Boundaries
    L = - np.random.uniform(0.0,1.0,size=int(ntemps*nwalkers/2))
    R = L + 1.0

    # Parallel stepping-out
    J = np.floor(maxsteps * np.random.uniform(0.0,1.0,size=int(ntemps*nwalkers/2)))
    K = (maxsteps - 1) - J

    # Initialise number of Log prob calls
    ncall = 0

    # Left stepping-out initialisation
    mask_J = np.full(int(ntemps*nwalkers/2),True)
    Z_L = np.empty(int(ntemps*nwalkers/2))
    X_L = np.empty((int(ntemps*nwalkers/2),ndim))

    # Right stepping-out initialisation
    mask_K = np.full(int(ntemps*nwalkers/2),True)
    Z_R = np.empty(int(ntemps*nwalkers/2))
    X_R = np.empty((int(ntemps*nwalkers/2),ndim))

    nexp_prime = np.zeros(int(ntemps*nwalkers/2))

    cnt = 0
    # Stepping-Out procedure
    while len(mask_J[mask_J])>0 or len(mask_K[mask_K])>0:
        if len(mask_J[mask_J])>0:
            cnt += 1
        if len(mask_K[mask_K])>0:
            cnt += 1
        if cnt > maxiter:
            raise RuntimeError('Number of expansions exceeded maximum limit! \n' +
                                           'Make sure that the pdf is well-defined. \n' +
                                           'Otherwise increase the maximum limit (maxiter=10^4 by default).')

        for j in indeces[mask_J]:
            if J[j] < 1:
                mask_J[j] = False

        for j in indeces[mask_K]:
            if K[j] < 1:
                mask_K[j] = False

        X_L[mask_J] = directions[mask_J] * L[mask_J][:,np.newaxis] + X_active[mask_J]
        X_R[mask_K] = directions[mask_K] * R[mask_K][:,np.newaxis] + X_active[mask_K]

        if len(X_L[mask_J]) + len(X_R[mask_K]) < 1:
            Z_L[mask_J] = np.array([])
            Z_R[mask_K] = np.array([])
            cnt -= 1
            ncall_new = 0
        else:
            Z_LR_masked, _, _, ncall_new = logprob(np.concatenate([X_L[mask_J],X_R[mask_K]]),np.concatenate([betas_active[mask_J],betas_active[mask_K]]))
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

    return L, R, Z0, nexp_prime, ncall


def Shrinking(logprob, X_active, L, R, Z0, betas_active, directions, indeces, ntemps, nwalkers, ndim, maxiter, ncall):
    # Shrinking procedure
    Widths = np.empty(int(ntemps*nwalkers/2))
    Z_prime = np.empty(int(ntemps*nwalkers/2))
    Zp_prime = np.empty(int(ntemps*nwalkers/2))
    Zl_prime = np.empty(int(ntemps*nwalkers/2))
    X_prime = np.empty((int(ntemps*nwalkers/2),ndim))
    ncon_prime = np.zeros(int(ntemps*nwalkers/2))
    mask = np.full(int(ntemps*nwalkers/2),True)

    cnt = 0
    while len(mask[mask])>0:
        # Update Widths of intervals
        Widths[mask] = L[mask] + np.random.uniform(0.0,1.0,size=len(mask[mask])) * (R[mask] - L[mask])

        # Compute New Positions
        X_prime[mask] = directions[mask] * Widths[mask][:,np.newaxis] + X_active[mask]
                    
        # Calculate LogP of New Positions
        Z_prime[mask], Zp_prime[mask], Zl_prime[mask], ncall_new = logprob(X_prime[mask], betas_active[mask])

        # Count LogProb calls
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
        if cnt > maxiter:
            raise RuntimeError('Number of contractions exceeded maximum limit! \n' +
                               'Make sure that the pdf is well-defined. \n' +
                               'Otherwise increase the maximum limit (maxiter=10^4 by default).')

    return X_prime, Zp_prime, Zl_prime, Z_prime, ncon_prime, ncall


def Slice(logprob, X, Z, Zp, Zl,  nexp, ncon, D, betas, active, inactive, mu, mean_distance, maxsteps, maxiter, i):

    X = X.copy()
    Z = Z.copy()
    Zp = Zp.copy()
    Zl = Zl.copy()

    X0 = X.copy()
    Z0 = Z.copy()
    Zp0 = Zp.copy()
    Zl0 = Zl.copy()

    ntemps, nwalkers, ndim = np.shape(X)

    betas_active = np.repeat(betas, nwalkers//2)
    indeces = np.arange(ntemps*nwalkers//2)

    X_active = X[np.arange(ntemps)[:,np.newaxis], active].reshape(-1,ndim)
    Z_active = Z[np.arange(ntemps)[:,np.newaxis], active].reshape(-1)

    # Directions
    X_inactive = X[np.arange(ntemps)[:,np.newaxis], inactive].reshape(-1,ndim)
        
    perms = list(permutations(np.arange(len(X_inactive)), 2))
    pairs = np.asarray(random.sample(perms,len(X_inactive))).T

    diff = (X_inactive[pairs[0]]-X_inactive[pairs[1]])
    diff /= np.linalg.norm(diff, axis=1)[:,np.newaxis]
    directions = 2.0 * mu * diff

    if i > 3:
        mean_dist = np.repeat(mean_distance, nwalkers).reshape(ntemps, nwalkers)[np.arange(ntemps)[:,np.newaxis],active]
        directions *= mean_dist.reshape(-1,1)

    # Stepping-out
    L, R, Z0, nexp_prime, ncall = SteppingOut(logprob,
                                              X_active,
                                              Z_active,
                                              betas_active,
                                              directions,
                                              indeces,
                                              ntemps,
                                              nwalkers,
                                              ndim,
                                              maxsteps=maxsteps,
                                              maxiter=maxiter)

    X_prime, Zp_prime, Zl_prime, Z_prime, ncon_prime, ncall = Shrinking(logprob,
                                                                        X_active,
                                                                        L,
                                                                        R,
                                                                        Z0,
                                                                        betas_active,
                                                                        directions,
                                                                        indeces,
                                                                        ntemps,
                                                                        nwalkers,
                                                                        ndim,
                                                                        maxiter=maxiter,
                                                                        ncall=ncall)
    

    X[np.arange(ntemps)[:,np.newaxis],active] = X_prime.reshape((ntemps, nwalkers//2, ndim))
    Z[np.arange(ntemps)[:,np.newaxis],active] = Z_prime.reshape(ntemps, nwalkers//2)
    Zp[np.arange(ntemps)[:,np.newaxis],active] = Zp_prime.reshape(ntemps, nwalkers//2)
    Zl[np.arange(ntemps)[:,np.newaxis],active] = Zl_prime.reshape(ntemps, nwalkers//2)
    nexp[np.arange(ntemps)[:,np.newaxis],active] = nexp_prime.reshape(ntemps, nwalkers//2)
    ncon[np.arange(ntemps)[:,np.newaxis],active] = ncon_prime.reshape(ntemps, nwalkers//2)

    D[np.arange(ntemps)[:,np.newaxis],active] = np.sqrt(np.sum((X[np.arange(ntemps)[:,np.newaxis],active] - X0[np.arange(ntemps)[:,np.newaxis],active])**2.0,axis=2))


    return X, Z, Zp, Zl, nexp, ncon, ncall, D
