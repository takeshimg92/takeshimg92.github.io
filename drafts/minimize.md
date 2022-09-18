_?_

```python
# Imports and data
from pydataset import data
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize
import pandas as pd

trees = data('Sitka')

trees['time_scaled'] = trees['Time']
trees['time_scaled'] -= trees['time_scaled'].min() # Time starts at t=0
trees['time_scaled'] /= trees['time_scaled'].max() # And goes until t = 1
```

```python
# Let's see what the data looks like
plt.title('Growth of Sitka Spruce Trees')
sns.regplot(trees['time_scaled'], trees['size'], fit_reg=False, x_jitter=.03)
plt.xlabel('Time')
plt.ylabel('Tree size')
plt.show()
```

```python
# Let's see what the data looks like
plt.title('Growth of Sitka Spruce Trees')
sns.regplot(trees['time_scaled'], trees['size'], x_estimator=np.mean, fit_reg=False)
plt.xlabel('Time')
plt.ylabel('Tree size')
plt.show()
```

$G(t, a, b, c) = ae^{-be^{-ct}}$

- $a$ - Ceiling value
- $b$ - X-shift
- $c$ - Growth speed

```python
def gompertz(a, b, c, t):
  return a * np.exp(-b*np.exp(-c*t))
```

$$\underbrace{y_t}_\textrm{Size at time t} \sim N(\underbrace{ae^{-be^{-ct}}}_\textrm{Mean at time t}, \underbrace{\sigma}_\textrm{Noise})$$

$\text{ln } \mathcal{L}(a, b, c, \sigma) = \sum_t \text{ln } f_{N}(y_t \mid G(t, a, b, c), \sigma)$

```python
def fit(t, y):
  def neg_log_likelihood(v):
    a, b, c, log_s = v
    expected_y = gompertz(a, b, c, t)
    l = norm(expected_y, np.exp(log_s)).logpdf(y)
    return -np.sum(l)
  return minimize(neg_log_likelihood, [1, 1, 1, 1])
```

```python
result = fit(trees['time_scaled'], trees['size'])

a_mle, b_mle, c_mle, log_s_mle = result.x
s_mle = np.exp(log_s_mle)

plt.scatter(trees['time_scaled'], trees['size'])
t_plot = np.linspace(trees['time_scaled'].min(), trees['time_scaled'].max())
y_plot = gompertz(a_mle, b_mle, c_mle, t_plot)
low_pred = y_plot - 2 * s_mle
high_pred = y_plot + 2 * s_mle
plt.plot(t_plot, y_plot, label='MLE')
plt.fill_between(t_plot, low_pred, high_pred, alpha=.2, label='Prediction interval')
plt.xlabel('Time')
plt.ylabel('Tree size')
plt.show()
```

```python
print(result)
print('Standard errors', np.sqrt(np.diag(result.hess_inv)))
```

$Fisher information matrix$

$P(a, b, c, s \mid data)$

Bernstein-von Mises, Laplace

```python
posterior = multivariate_normal(result.x, result.hess_inv)

posterior_samples = pd.DataFrame(posterior.rvs(1000), columns=['a', 'b', 'c', 'log_sigma'])

sns.distplot(posterior_samples['a'], label='a')
sns.distplot(posterior_samples['b'], label='b')
sns.distplot(posterior_samples['c'], label='c')
plt.legend()
plt.title('Posterior samples for parameters')
plt.show()
```

```python
for a_sim, b_sim, c_sim, _ in posterior_samples.values:
  y_plot = gompertz(a_sim, b_sim, c_sim, t_plot)
  plt.plot(t_plot, y_plot, color='blue', alpha=.01)
plt.xlabel('Time')
plt.ylabel('Tree size')
plt.title('Posterior samples of growth curves')
plt.show()
```

----------------------
# V1

# We often want to tailor our model to the situation

We commonly want to fit a model that can't be expressed as a basis-expanded linear regression

Example of diminishing returns/dose response - satiation, Cat treats, marketing spend, organic growth;

Lots of relationships between two real variables go something like this: "As x increases, so does y. But the next x causes less increase than the last one"

The problem - I can't do `from statsmodels import gompertz_regression`

In this case, we can use scipy to find the best parameters and their SEs

# ?

```python
from scipy.stats import norm, sem
from scipy.optimize import minimize
import numpy as np

x = norm(0, 1).rvs(1000)

def neg_log_likelihood(param_vector):
  mu, log_sd = param_vector
  return -np.sum(norm(mu, np.exp(log_sd)).logpdf(x))
  
result = minimize(neg_log_likelihood, [10, 10])

print(np.sqrt(np.diag(result.hess_inv)), sem(x))
```

```python
from pydataset import data
from matplotlib import pyplot as plt
import seaborn as sns

trees = data('Sitka')

trees['Time'] -= trees['Time'].min() # Time starts at t=0
trees['Time'] /= trees['Time'].max() # And goes until t = 1

plt.scatter(trees['Time'], trees['size'])
plt.show()

def gompertz(a, b, c, t):
  return a * np.exp(-b*np.exp(-c*t))
  
plt.scatter(trees['Time'], trees['size'])

from scipy.stats import norm

def gompertz(a, b, c, t):
  return a * np.exp(-b*np.exp(-c*t))
  
def fit(t, y):
  def neg_log_likelihood(v):
    a, b, c, log_s = v
    expected_y = gompertz(a, b, c, t)
    l = norm(expected_y, np.exp(log_s)).logpdf(y)
    return -np.sum(l)
  return minimize(neg_log_likelihood, [1, 1, 1, 1])
  
result = fit(trees['Time'], trees['size'])

a_mle, b_mle, c_mle, log_s_mle = result.x
s_mle = np.exp(log_s_mle)

plt.scatter(trees['Time'], trees['size'])
t_plot = np.linspace(trees['Time'].min(), trees['Time'].max())
y_plot = gompertz(a_mle, b_mle, c_mle, t_plot)
low_pred = y_plot - 2 * s_mle
high_pred = y_plot + 2 * s_mle
plt.plot(t_plot, y_plot)
plt.fill_between(t_plot, low_pred, high_pred, alpha=.1)
plt.show()

print(result)
print('Standard errors', np.sqrt(np.diag(result.hess_inv)))
```

$y_t \sim N(ae^{-be^{-ct}}, \sigma)$
