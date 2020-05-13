********************************************************************************************
A Simple Recommender System with Observed Attributes and Time-Varying Taste Parameters
********************************************************************************************
This document provides a brief explanation of the ``recommendx`` package for Python. This package implements a recommender
system, similar to the matrix factorization-based algorithms (SVD) available in the **excellent**
`Surprise <http://surpriselib.com/>`_ package.

This package extends the standard SVD recommender system by allowing researchers to include observed item attributes and
also user taste parameters that vary over time. The model is fit using stochastic gradient descent.

If you use this package, please cite the following working paper:

- Adam D. Rennhoff (2020): "A Simple Recommender System with Observed Attributes and Time-Varying Parameters,"MTSU Working Paper #XXXX (TO BE UPDATED WITH NUMBER AND LINK)

The paper contains a number of helpful examples and suggestions for implementation.
Please consult the paper for help on using ``recommendx`` for your research.

This package is distributed under the `BSD 3-Clause license <https://opensource.org/licenses/BSD-3-Clause>`_.

Installation
######################################

``recommendx`` has the following dependencies:

- Python (>=3.5)
- NumPy (>= 1.10)
- Pandas (>= 0.18)

The easiest way to install ``recommendx`` is using ``pip``:
::

    pip install recommendx

Alternatively, the package can be accessed from Github.

The package contains two related prediction algorithms: ``RWR`` and ``RWT``. These are discussed below.

Recommendation with Regressors (RWR)
######################################

``RWR`` implements a slightly modified version of what might we might call the "classic" SVD algorithm. This is often
attributed to `Simon Funk <https://sifter.org/~simon/journal/20061211.html>`_, who famously used it during the
`Netflix Prize <https://www.netflixprize.com/>`_ competition. The classic SVD approach relies only upon latent item
attributes. ``RWR`` extends this framework by allowing the researcher to specify observed item attributes, as well.

We can define :math:`\hat{r}_{ui}` as user :math:`u`'s predicted rating for item :math:`i`:

.. math::
    \hat{r}_{ui} = \mu + b_u + X_i\beta_u + Z_i\alpha_u

In this specification,

- :math:`\mu` is the average rating in the data
- :math:`b_u` is the bias for user :math:`u`
- :math:`X_i` is a vector of **observed** attributes for item :math:`i`
- :math:`Z_i` is a vector of **latent** attributes for item :math:`i`
- :math:`\beta_u` and :math:`\alpha_u` are user :math:`u`'s preferences for observed and latent item attributes, respectively

This specification is similar to the usual matrix factorization set-up, with the standard item bias term (:math:`b_i`)
replaced by observed attributes.

Defining :math:`R` as the set of all observed user-item ratings
and imposing L2-regularization on our parameters, we seek to minimize the following objective function:

.. math::
    \sum_{r_{ui}\in R} = (r_{ui}-\hat{r}_{ui})^2 + \lambda(b_u^2 + \vert\vert\beta_u\vert\vert^2 + \vert\vert Z_i \vert\vert^2 + \vert\vert\alpha_u\vert\vert^2)

The minimization is done using stochastic gradient descent (SGD).
The relevant gradients, which can easily be obtained by hand, lead to the following update rules:

- :math:`b_u\text{ } \Longleftarrow\text{ } b_u + \gamma(e_{ui} - \lambda b_u)`
- :math:`\beta_u\text{ } \Longleftarrow\text{ } \beta_u + \gamma(e_{ui} X_i - \lambda\beta_u)`
- :math:`\alpha_u\text{ } \Longleftarrow\text{ } \alpha_u + \gamma(e_{ui} Z_i - \lambda\alpha_u)`
- :math:`Z_i\text{ } \Longleftarrow\text{ } Z_i + \gamma(e_{ui} \alpha_i - \lambda Z_i)`

where :math:`e_{ui} = r_{ui} - \hat{r}_{ui}`, :math:`\text{ }\lambda` is the regularization penalty term, and
:math:`\gamma` is the learning rate. The learning rates determines how large of a "step" to take when we update
parameters.

Parameters
---------------

Note: I have purposely chosen the parameter names to be similar to those in `Surprise <http://surpriselib.com/>`_
in order to facilitate easy movement between packages.

- **n_factors** - The number of latent factors in :math:`Z`. Default is ``50``.

- **n_epochs** - The number of iterations of the SGD procedure. Default is ``50``.

- **init_mean** - The mean of the normal distribution used to initialize parameter values. Default is ``0``.

- **init_std_dev** - The standard deviation of the normal distribution used to initialize parameter values. Default is ``0.1``.

- **reg** - The regularization term used for all parameters (:math:`\lambda`). Default is ``0.02``.

- **lr** - The learning rate for all parameters (:math:`\gamma`). Default is ``0.005``.

Attributes
---------------

Once an ``RWR`` instance is ``fit()``, the resulting parameter values are returned
as attributes of the instance.

- ``intercept_``: (:math:`\mu`)
    Scalar intercept term

- ``bu``: (:math:`b_u`)
    NumPy array with shape (n_users, 1)

- ``B``: (:math:`\beta_u`)
    If :math:`X_i` is provided to ``fit()`` (see below), ``B`` is a NumPy array with shape (n_users, n_Xs)

- ``alpha_``: (:math:`\alpha_u`)
    NumPy array with shape (n_users, n_factors)

- ``Z``: (:math:`Z_i`)
    NumPy array with shape (u_items, n_factors)

Methods
---------------

- ``fit(self,df,Xi=None)``
    - Fits the recommender system model
    - ``df`` must be a NumPy array
        - Each row corresponds to a rating (:math:`r_{ui}`)
        - Columns **must** be ordered: [user, item, rating]
        - ``user`` and ``item`` may be strings or integers
    - ``Xi`` (if supplied) must be a NumPy array
        - If no observed item attributes are supplied, ``fit()`` returns the same results as SVD
        - First column of ``Xi`` must be item identifier that corresponds with item labels used in ``df``
        - Shape of array is (n_items, 1 + n_Xs)

- ``accuracy(self,df,Xi=None)``
    - Returns the mean squared prediction error
    - Requires the recommender system to be fit first
    - All provided values must be in the same format as supplied to ``fit()``

- ``predict(self,u_p,i_p)``
    - Returns predicted ratings
    - ``u_p`` is a user value
    - ``i_p`` is an item value
    - Both ``u_p`` and ``i_p`` must be provided in the same format as ``fit()``

Sample Syntax
---------------

If we assume that ``dat`` is a NumPy array of ratings data and ``att`` is a NumPy array
of observed item attributes, we can use the following code:

::

    from recommendx import RWR
    rwr = RWR(n_factors = 5)
    rwr.fit(dat,att)
    rwr.accuracy(dat,att)
    rwr.predict('userA','item10')

Recommendation with Time (RWT)
######################################

``RWT`` implements the same basic model as ``RWR`` but allows for time-varying taste parameters.

Our main ratings prediction equation becomes:

.. math::
    \hat{r}_{uit} = \mu + b_u + X_i\beta_{u,t} + Z_i\alpha_{u,t}

Neither observed (:math:`X_i`) nor unobserved (:math:`Z_i`) item attributes vary with time
(although one could "trick" the model into allowing that by creating items that are time-specific).

User tastes parameters :math:`\beta_{u,t}` and :math:`\alpha_{u,t}` are assumed to vary by time period.
This allows for the possibility, for example, that a Netflix viewer might be more inclined
to enjoy a horror movie at night. Or a coffee drinker may prefer espresso drinks more in the
morning than in the evening.

``RWT`` requires that *time* be defined categorically (more on this below). A simple example might be
time = ["Morning","Afternoon","Evening"]. These categorical labels must be assigned by the researcher prior
to fitting the recommender.

The model is fit using stochastic gradient descent. The equations are identical to those for
``RWR`` with the exception that the :math:`\beta` and :math:`\alpha` parameters are now subscripted with time,
as well.

Parameters
---------------

``RWT`` has the same model parameters as ``RWR``. Parameter arrays :math:`\beta_{u,t}` and :math:`\alpha_{u,t}`
are identified only using ratings observations for each specific time period.
To account for this, the default value of **n_epochs** has been increased to ``100``.

Attributes
---------------

Once an ``RWT`` instance is ``fit()``, the resulting parameter values are returned
as attributes of the instance.

- ``intercept_``: (:math:`\mu`)
    Scalar intercept term

- ``bu``: (:math:`b_u`)
    NumPy array with shape (n_users, 1)

- ``B``: (:math:`\beta_{u,t}`)
    If :math:`X_i` is provided to ``fit()`` (see below), ``B`` is a 3-dimensional NumPy array with shape (n_times, n_users, n_Xs)

- ``alpha_``: (:math:`\alpha_{u,t}`)
    A 3-dimensional NumPy array with shape (n_times, n_users, n_factors)

- ``Z``: (:math:`Z_i`)
    NumPy array with shape (u_items, n_factors)

Methods
---------------

- ``fit(self,df,Xi=None)``
    - Fits the recommender system model
    - ``df`` must be a NumPy array
        - Each row corresponds to a rating (:math:`r_{uit}`)
        - Columns **must** be ordered: [user, item, rating, time]
        - ``user`` and ``item`` may be strings or integers
        - ``time`` should be the time label for :math:`r_{uit}`. Can be string or integer but is treated as categorical
    - ``Xi`` (if supplied) must be a NumPy array
        - If no observed item attributes are supplied, ``fit()`` returns results for SVD with time-varying parameters
        - First column of ``Xi`` **must** be item identifier that corresponds with item labels used in ``df``
        - Shape of array is (n_items, 1 + n_Xs)

- ``accuracy(self,df,Xi=None)``
    - Returns the mean squared prediction error
    - Requires the recommender system to be fit first
    - All provided values must be in the same format as supplied to ``fit()``

- ``predict(self,u_p,i_p,tee)``
    - Returns predicted ratings
    - ``u_p`` is a user value
    - ``i_p`` is an item value
    - ``tee`` is a time value
    - ``u_p``, ``i_p``, and ``tee`` must be provided in the same format as ``fit()``

Sample Syntax
---------------

If we assume that ``dat`` is a NumPy array of ratings data (with time label) and ``att`` is a NumPy array
of observed item attributes, we can use the following code:

::

    from recommendx import RWT
    rwt = RWT(n_factors = 4)
    rwt.fit(dat,att)
    rwt.accuracy(dat,att)
    rwt.predict('userA','item10','AM')
    rwt.predict('userA','item10','PM')

