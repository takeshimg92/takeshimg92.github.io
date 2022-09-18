---
layout: post
title: "Speed up your analysis of giant datasets with sufficient statistics"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: jellybeans.png
---

Applying 

# Lots of rows means lots of waiting

In the 

Sometimes it's more convenient to work with summaries of subgroups in the data than with the raw data itself

- Combining multiple datasets, as in [Meta-analysis](https://en.wikipedia.org/wiki/Meta-analysis)
- A large number of datapoints but a small number of parameters
- Observational analysis from data matched in strata

# Sufficient statistics

https://en.wikipedia.org/wiki/Sufficient_statistic
https://web.ma.utexas.edu/users/gordanz/notes/likelihood_color.pdf
# Binomial outcomes: Easy with statsmodels

https://www.statsmodels.org/devel/generated/statsmodels.genmod.generalized_linear_model.GLM.html

```python
import pandas as pd
from statsmodels.api import formula as smf
from statsmodels import api as sm

df_long = pd.DataFrame({'x': [0] * 100 + [1]*100, 'y':[0] * 95 + [1]*5 + [0] * 5 + [1]*95})

long_model = smf.logit('y ~ x', df_long)
long_fit = long_model.fit()
print(long_fit.summary())

df_short = pd.DataFrame({'x': [0, 0, 1, 1], 'y':[1, 0, 1, 0]})
n_trials = pd.Series([5, 95, 95, 5])

short_model = smf.glm('y ~ x', df_short, n_trials=n_trials, family=sm.families.Binomial())
short_fit = short_model.fit()
print(short_fit.summary())
```

# Continuous outcomes: Some assembly required

WLS implementation in statsmodels doesn't work here? Try GLM https://www.statsmodels.org/devel/examples/notebooks/generated/glm_weights.html

```python
from scipy.optimize import curve_fit
import numpy as np
from scipy.stats import norm
import pandas as pd
from statsmodels.api import formula as smf
from statsmodels import api as sm

n_per_group = 1000

df = pd.DataFrame({'x': [0] * n_per_group + [1] * n_per_group, 'y': np.concatenate((norm(1, 4).rvs(n_per_group), norm(3, 4).rvs(n_per_group)))})

def summarize(df, x_cols, y_col):
  X_summary = []
  y_summary = []
  se_summary = []
  for X_values, group_df in df.groupby(x_cols):
    X_summary.append(X_values)
    y_summary.append(np.mean(group_df[y_col]))
    se_summary.append(np.std(group_df[y_col]) / np.sqrt(len(group_df)))
  X_summary = pd.DataFrame(X_summary, columns=x_cols)
  y_summary = np.array(y_summary)
  se_summary = np.array(se_summary)
  return X_summary, y_summary, se_summary
  
X_summary, y_summary, se_summary = summarize(df, ['x'], 'y')

def ols_summaries(X, y, se):
  def f(X, *v):
    a, b = v[0], v[1:]
    return a + np.dot(X, b)
  r, c = X.shape
  params, cov = curve_fit(f, X, y, sigma=se, absolute_sigma=True, p0=np.ones(c+1))
  params_se = np.sqrt(np.diag(cov))
  return params, params_se
  
p_summary, p_se_summary = ols_summaries(X_summary, y_summary, se_summary)

print(p_summary, p_se_summary)

print(smf.ols('y ~ x', df).fit().summary())

print(sm.WLS(y_summary, sm.add_constant(X_summary), 1./se_summary**2).fit().summary())
```

## Digression: Sufficient statistic for the normal distribution

https://en.wikipedia.org/wiki/Sufficient_statistic#Normal_distribution

$$\mathcal{L}(\mu, \sigma | \bar{y}, s^2, n) = (2 \pi \sigma^2)^{\frac{n}{2}} exp \left(\frac{n - 1}{2 \sigma^2} s^2 \right ) exp \left (-\frac{n}{2 \sigma^2} (\mu - \bar{y})^2 \right)$$

$$ln \mathcal{L}(\mu, \sigma | \bar{y}, s^2, n) = -\frac{n}{2} ln(2 \pi \sigma^2) - \left( \frac{n-1}{2 \sigma^2}s^2 - \frac{n}{2\sigma^2} (\mu - \bar{y})^2 \right) $$

```python
import numpy as np
from scipy.optimize import minimize

def logpdf_sufficient(mu, sigma_sq, sample_mean, sample_var, n):
  return -(n/2) * np.log(2*np.pi*sigma_sq) - (((n-1) / (2*sigma_sq)) * sample_var) - ((n / (2*sigma_sq)) * (mu - sample_mean)**2) 

data = np.random.normal(0, 1, 1000)
n = len(data)
m, v = np.mean(data), np.var(data)

def neg_log_likelihood(p):
  return -logpdf_sufficient(p[0], np.exp(p[1]), m, v, n)

result = minimize(neg_log_likelihood, np.array([5, -1]))
cov = result.hess_inv

se_mean = np.sqrt(cov[0][0])
```
