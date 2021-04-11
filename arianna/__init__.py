__version__ = "0.0.1"
__url__ = "https://arianna.readthedocs.io"
__author__ = "Minas Karamanis"
__email__ = "minaskar@gmail.com"
__license__ = "GPL-3.0"
__description__ = "Replica Exchange Slice Sampling"


from .ensemble import *
from .parallel import ChainManager
from .autocorr import AutoCorrTime
#from .plotting import cornerplot