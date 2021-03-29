import numpy as np

class _FunctionWrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included.

    Args:
        f (callable) : Log Probability function.
        args (list): Extra arguments to be passed into the logprob.
        kwargs (dict): Extra arguments to be passed into the logprob.

    Returns:
        Log Probability function.
    """

    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, x):
        return self.f(x, *self.args, **self.kwargs)


class _LogPosteriorWrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included.

    Args:
        f (callable) : Log Probability function.
        args (list): Extra arguments to be passed into the logprob.
        kwargs (dict): Extra arguments to be passed into the logprob.

    Returns:
        Log Probability function.
    """

    def __init__(self, loglike_fn, loglike_args, loglike_kwargs, logprior_fn, logprior_args, logprior_kwargs):
        self.loglike_fn = loglike_fn
        self.loglike_args = [] if loglike_args is None else loglike_args
        self.loglike_kwargs = {} if loglike_kwargs is None else loglike_kwargs
        self.logprior_fn = logprior_fn
        self.logprior_args = [] if logprior_args is None else logprior_args
        self.logprior_kwargs = {} if logprior_kwargs is None else logprior_kwargs

    def __call__(self, x):

        lp = self.logprior_fn(x, *self.logprior_args, **self.logprior_kwargs)

        if ~np.isfinite(lp):
            return lp, lp

        return lp, self.loglike_fn(x, *self.loglike_args, **self.loglike_kwargs)
