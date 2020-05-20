# A Simple Recommender System with Observed Attributes and Time-Varying Parameters

This package implements a recommender system, similar to matrix factorization-based algorithms (SVD) available in the
**excellent** [Surprise](http://surpriselib.com/). 

This package extends the standard SVD recommender system by allowing researchers to include observed items attributes and
also user taste parameters that vary over time. The model is fit using stochastic gradient descent.

## Getting Started
The easiest way to obtain the package is to install using ``pip``:
```python
pip install recommendx
```
The following resources may be helpful to use in getting started using ``recommendx``.

1. [Documentation.pdf](https://github.com/adrennhoff/recommendx/blob/master/Extras/Documentation.pdf) - contains documentation and technical details, including information on installation, methods, and class attributes
2. [Demonstration.ipynb](https://github.com/adrennhoff/recommendx/blob/master/Extras/Demonstration.ipynb) - this Jupyter notebook provides a brief tutorial on using ``recommendx``

## Recommendation Algorithms

The package contains two recommendation methods: `RWR` and `RWT`. 

### Recommendation with Regressors (RWR)

`RWR` implements a slightly modified version of what might we might call the "classic" SVD algorithm. This is often attributed to [Simon Funk](https://sifter.org/~simon/journal/20061211.html), who famously used it during the [Netflix Prize](https://www.netflixprize.com/) competition. The classic SVD approach relies only upon latent item attributes. `RWR` extends this framework by allowing the researcher to specify observed item attributes, as well.

#### Sample Syntax
```python
from recommendx import RWR
rwr = RWR(n_factors = 5)
rwr.fit(dat,att)
rwr.accuracy(dat,att)
rwr.predict('userA','item10')
```

### Recommendation with Time (RWT)

`RWT` implements the same basic model as `RWR` but allows for time-varying taste parameters.

#### Sample Syntax
```python
from recommendx import RWT
rwt = RWT(n_factors = 4)
rwt.fit(dat,att)
rwt.accuracy(dat,att)
rwt.predict('userA','item10','PM')
rwt.predict('userA','item10','AM')
```
