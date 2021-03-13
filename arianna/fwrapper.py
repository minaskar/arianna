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
