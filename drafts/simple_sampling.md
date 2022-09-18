Simple methods for sampling

# How do we sample from a distribution? Why would we want to?

We often know a distribution analytically, but can't sample from it. Sampling from a distribution allows us to perform numerical analysis of many of its properties. Most commonly, we are interested in its integrals or moments, which may be hard to acquire analytically. This kind of problem shows up all the time in Bayesian inference, in which we might look derive the posterior distribution for our favorite parameters, but it may have an unpleasant analytical form.

MCMC provides a general (but costly) solution. For simpler, especially lower dimensional distributions, there are some less complex methods we can use. It's useful to have these at your disposal when MCMC is overkill, and can be quick solutions to simple sampling problems.

# A running example: an asymmetric, unimodal, unnormalized density function

$f(X) = x (1-x)^3$

# Sampling from a one-dimensional distribution: Inverse transform sampling of the CDF

https://en.wikipedia.org/wiki/Inverse_transform_sampling

- Compute the CDF, $F$
- Invert it to get $F^{-1}$
- Sample $u$ uniformly on the unit interval
- Compute $F^{-1}(u)$

## An analytical solution with Sympy

```python
from sympy import symbols, integrate, latex
x, a, b = symbols('x a b')
f = (x**(a-1))*((1-x)**(b-1))
F = integrate(f, (x, 0, 1), meijerg=True)
const = F.evalf(subs={a:2, b:4}) # normalizing constant; don't use meijerg if x is the only variable
print(latex(F))
```

## A numerical solution with Scipy

```python
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import uniform

# https://stackoverflow.com/questions/47275957/invert-interpolation-to-give-the-variable-associated-with-a-desired-interpolatio

def f(x):
  return x * (1-x)**3
  
n_points = 1000
grid_points = np.linspace(0, 1, n_points)
density = f(grid_points)

approx_f = InterpolatedUnivariateSpline(grid_points, density)

plt.scatter(grid_points, density)
plt.plot(grid_points, approx_f(grid_points), color='orange')
plt.show()

approx_cdf = np.vectorize(lambda x: approx_f.integral(0, x) / approx_f.integral(0, 1))
plt.plot(grid_points, approx_cdf(grid_points))
plt.show()

inv_approx_cdf = InterpolatedUnivariateSpline(approx_cdf(grid_points), grid_points)
plt.plot(grid_points, inv_approx_cdf(grid_points))
plt.show()

x_uniform = uniform(0, 1).rvs(1000)
samples = inv_approx_cdf(x_uniform)
sns.distplot(samples)
plt.plot(grid_points, density / approx_f.integral(0, 1))
plt.show()
```

# Low-dimensional distributions: Grid sampling

- Select grid bounds and resolution
- Evaluate $f$ over the grid
- Weighted sample according to $f$

```python
import numpy as np
from scipy.stats import norm 
from matplotlib import pyplot as plt
import seaborn as sns

x_plot = np.linspace(-10, 10, 5000)
log_p = norm(0, 1).logpdf(x_plot) + 500
p_sample = np.exp(log_p - np.logaddexp.reduce(log_p))
sns.distplot(np.random.choice(x_plot, p=p_sample, size=10000))
plt.plot(x_plot, norm(0, 1).pdf(x_plot))
plt.show()
```

# Unimodal normal-like distributions, however many dimensions they have: Laplace's approximation

- Find maximum
- Compute inverse hessian
- Construct normal approximation

```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from numdifftools import Hessian

def approx_dist(lnprob, v_init=None, v_max=None, hessian=None):
    """
    Construct a laplace approximation to the distribution with the given log-density.
    Arguments:
    -lnprob: The log-density which you would like to approximate. Should take a vector and return a real number.
    -v_init: The initial value at which to start the search for the mode of lnprob. If it is not given.
    -v_max: The mode of lnprob. If it is not provided, it will be calculated numerically.
    -hessian: A function which will compute the Hessian. If it is not given, it will be approximated numerically.
    Returns:
    -approximate distribution, a scipy.stats.multivariate_normal object
    """
    neg_lnprob = lambda v: -lnprob(v)
    if v_max is None and v_init is not None:
        result = minimize(neg_lnprob, v_init)
        x_max = result.x
    elif v_max is not None:
        x_max = v_max
    else:
        raise Exception('You must provide either an initial value at which to start the search for the mode (v_init) or the value of the mode (v_max)')
    if hessian is None:
        hess_calc = Hessian(lnprob)
    else:
        hess_calc = hessian
    h = hess_calc(x_max)
    dist = multivariate_normal(x_max, -np.linalg.inv(h))
    return dist

```

## An interesting connection between the large-sample properties of the Bayesian and Frequentist worlds: Laplace's approximation and the Fisher information

http://gregorygundersen.com/blog/2019/11/28/asymptotic-normality-mle/

https://stephens999.github.io/fiveMinuteStats/asymptotic_normality_mle.html

https://ocw.mit.edu/courses/mathematics/18-443-statistics-for-applications-fall-2006/lecture-notes/lecture3.pdf

# Comparing these methods

|Method|Dimensionality|Summary|
|-|-|-|
|Inverse transform sampling|1D only|Invert the CDF and sample from the uniform distribution on [0, 1]|
|Grid sampling|Low|Compute a subset of PDF values, and treat the distribution as discrete|
|Laplace approximation|Any, as long as it's unimodal|Find the maximum, and fit a normal approximation around it|
