********************************************************************************************
A Simple Recommender System with Observed Attributes and Time-Varying Taste Parameters
********************************************************************************************
This document provides a brief explanation of the ``recommendx`` package for Python. This package implements a recommender
system, similar to the matrix factorization-based algorithms (SVD) available in the **excellent**
`Surprise <http://surpriselib.com/>`_ package.

This package extends the standard SVD recommender system by allowing researchers to include observed item attributes and
also to allow user taste parameters to vary over time. The model is fit using stochastic gradient descent.

If you use this package, please cite the following working paper:

- Adam D. Rennhoff (2020): "A Simple Recommender System with Observed Attributes and Time-Varying Parameters,"MTSU Working Paper #XXXX (TO BE UPDATED WITH NUMBER AND LINK)

The paper contains a number of helpful examples and suggestions for implementation.

Installation
##############

``recommendx`` has the following dependencies:

- Python (>=3.5)
- NumPy (>= 1.10)
- Pandas (>= 0.18)

The easiest way to install ``recommendx`` is using ``pip``:
::

    pip install recommendx

Alternatively, you could fork the package from Github and install it locally on your own.

The package contains two related prediction algorithms: ``RWR`` and ``RWT``. These are discussed below.



