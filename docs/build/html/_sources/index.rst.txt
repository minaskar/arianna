.. title:: Arianna documentation

.. figure:: ./../../logo.png
    :scale: 30 %
    :align: center


Introduction
============

**arianna** [#f1]_ is a Python implementation of the *Metropolis--Coupled Slice Sampling* method that
generates posterior samples from high-dimensional and strongly multimodal distributions. Apart
from *Bayesian parameter inference*, **arianna** also provides unbiased and low-variance estimates of
the *model evidence (aka marginal likelihood)* at no additional cost. The sampler is modular and
does not require any hand-tuning from the user. Its parallel and black-box nature renders it ideal
for computationally expensive models with high number of parameters often met in the physical sciences.

.. [#f1] `Named after Dr. Arianna W. Rosenbluth, one of the main developers of the Metropolis algorithm
   and the first person in history to ever code a Markov Chain Monte Carlo algorithm. <https://www.nytimes.com/2021/02/09/science/arianna-wright-dead.html>`_


Documentation
=============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   pages/quickstart
   api

Attribution
===========

Please cite the following if you find this code useful in your
research. The BibTeX entry for the paper is::

    @article{arianna,
        title={arianna: A Metropolis--Coupled Slice Sampler for Bayesian Inference and Model Selection},
        author={Minas Karamanis and Florian Beutler},
        year={2021},
        note={in prep}
    }


Authors & License
=================

Copyright 2021 Minas Karamanis and contributors.

``arianna`` is free software made available under the ``GPL-3.0 License``.


Changelog
=========

**0.0.1 (14/03/21)**

- First version