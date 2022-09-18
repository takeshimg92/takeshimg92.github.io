---
layout: post
title: "Confidence intervals for ratios with the Jackknife and Bootstrap"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: jellybeans.png
---

*Practical questions often revolve around ratios of interest - open rates, costs per impression, percentage increases - but the statistics of ratios is more complex than you might realize. The sample ratio is biased, and its standard error is surprisingly hard to pin down. We'll see that despite this, we can use the bootstrap (or its older sibling, the jackknife) to handle both of these problems. Along the way, we'll learn a little about how these methods work and when they're useful.*

# Ratios are everywhere

Something that might surprise students of statistics embarking on their first job is that quite a lot of practical questions are not framed in terms of the difference, $X - Y$, but rather the ratio, $\frac{X}{Y}$. Despite the fact that it seems very natural to ask questions about relative changes, a lot of initial statistics education focuses on the difference because it is easier to deal with. It is easy to find the standard error of $X - Y$ if we know the standard errors of $X$ and $Y$ and their correlation; we simply use the fact that variances add in this situation, perhaps with a covariance term. If you attempt to find an explanation of the standard error of $\frac{X}{Y}$ though, you suddenly encounter [a bewildering amount of calculus](http://www.stat.cmu.edu/~hseltman/files/ratio.pdf), and a stomach-churning number of Taylor expansions. This eventually yields an approximation for the standard error, but it's not always clear when this applies. That lack of clarity is unfortunate, because in my work I see ratios all the time, like:

- Open rates: $\text{Open rate} = \frac{\text{Opened}}{\text{Sent}}$
- Revenue per action: $\text{Revenue per action} = \frac{\text{Total Revenue received}}{\text{Total Actions performed}}$
- Cost per impression: $\text{Cost per impression} = \frac{\text{Total spend}}{\text{Impression count}}$
- Percent increase: $\text{Lift} = \frac{\text{Total new metric}}{\text{Total old metric}}$

Synthetic advertising example: Open rate, with dependence between send count and open likelihood

Paired vs unpaired observations?

# But the "obvious" ratio estimate is biased, and its standard errors can be tricky

```
from scipy.stats import binom, pareto, sem, t, norm
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.utils import resample
import pandas as pd
from tqdm import tqdm

cost_dist = pareto(2)
widget_dist = binom(2*10, 0.1)

TRUE_R = 1

def gen_data(n):
  a, b = cost_dist.rvs(n), widget_dist.rvs(n)
  while sum(b) == 0:
    a, b = cost_dist.rvs(n), widget_dist.rvs(n)
  return a, b

datasets = [gen_data(5) for _ in range(1000)]
```
```
def naive_estimate(n, d):
  return np.sum(n) / np.sum(d)
  
naive_sampling_distribution_n_5 = [naive_estimate(n, d) for n, d in datasets] # This is biased
```


The naive estimator is biased though this is less of an issue with large sample sizes

The variance of a corrected estimator is not obvious

# Cookbook solutions

The "cookbook" taylor series solution produces reasonable results, though it is biased

```
def taylor_estimate(n, d, alpha):
  k = len(n)
  t_val = t(k-1).interval(1.-alpha)[1]
  r = naive_estimate(n, d)
  s_n = (1/(k * (k-1))) * np.sum((n - np.mean(n))**2)
  s_d = (1/(k * (k-1))) * np.sum((d - np.mean(d))**2)
  s_nd = (1/(k * (k-1))) * np.sum((n - np.mean(n))*(d - np.mean(d)))
  point = r
  se = np.abs(r) * np.sqrt((s_d/np.mean(d)**2) + (s_n/np.mean(n)**2) - 2*(s_nd / np.mean(n*d)))
  return point, point - t_val * se, point + t_val * se
  
taylor_simulation_results = pd.DataFrame([taylor_estimate(a, b, .05) for a, b in datasets], columns=['point', 'lower', 'upper'])
taylor_simulation_results['bias'] = taylor_simulation_results['point'] - TRUE_R
taylor_simulation_results['covered'] = (taylor_simulation_results['lower'] < TRUE_R) & (taylor_simulation_results['upper'] > TRUE_R)
```

https://arxiv.org/pdf/0710.2024.pdf - p 10 + 8

http://www3.stat.sinica.edu.tw/statistica/oldpdf/A5n110.pdf Fieller and bootstrap extensions?

Does this work poorly when there is a nonlinear dependence? https://stats.stackexchange.com/questions/243510/how-to-interpret-the-delta-method

# The Jackknife as a method for correcting bias and computing standard errors

```
def jackknife_estimate(n, d, alpha):
  total_n, total_d = np.sum(n), np.sum(d)
  k = len(n)
  r = naive_estimate(n, d)
  r_i = (total_n - n) / (total_d - d) 
  se = np.sqrt(((k-1)/k) * np.sum(np.power(r_i - np.mean(r_i), 2)))
  t_val = t(k-1).interval(1.-alpha)[1]
  if np.any(np.isinf(r_i)):
    point = r
    se = 0
  else:
    point = k * r - (k-1)*np.mean(r_i)
  return point, point - t_val * se, point + t_val * se
  
jackknife_simulation_results = pd.DataFrame([jackknife_estimate(a, b, .05) for a, b in datasets], columns=['point', 'lower', 'upper'])
jackknife_simulation_results['bias'] = jackknife_simulation_results['point'] - TRUE_R
jackknife_simulation_results['covered'] = (jackknife_simulation_results['lower'] < TRUE_R) & (jackknife_simulation_results['upper'] > TRUE_R)
```

The idea behind the jackknife standard error

the idea behind the jackknife standard error

formulas

this is an early bootstrap

the jackknife is conservative

# A better bias-correction method: The bootstrap

```
def standard_bootstrap_estimate(n, d, alpha, n_sim=100):
  r = naive_estimate(n, d)
  k = len(n)
  boot_samples = [naive_estimate(*resample(n, d)) for _ in range(n_sim)]
  boot_samples = [s for s in boot_samples if not np.isinf(s)]
  bootstrap_bias = np.mean(boot_samples) - r
  if np.isinf(bootstrap_bias):
    bootstrap_bias = 0
  point = r - bootstrap_bias
  se = np.std(boot_samples)
  t_val = t(k-1).interval(1.-alpha)[1]
  return point, point - t_val * se, point + t_val * se
  
standard_bootstrap_simulation_results = pd.DataFrame([standard_bootstrap_estimate(a, b, .05) for a, b in datasets], columns=['point', 'lower', 'upper'])
standard_bootstrap_simulation_results['bias'] = standard_bootstrap_simulation_results['point'] - TRUE_R
standard_bootstrap_simulation_results['covered'] = (standard_bootstrap_simulation_results['lower'] < TRUE_R) & (standard_bootstrap_simulation_results['upper'] > TRUE_R)
```

the jackknife is a good first approximation

but doesn't deal with asymmetry correctly

```
def percentile_bootstrap_estimate(n, d, alpha, n_sim=10000):
  r = naive_estimate(n, d)
  k = len(n)
  boot_samples = [naive_estimate(*resample(n, d)) for _ in range(n_sim)]
  boot_samples = [s for s in boot_samples if not np.isinf(s)]
  bootstrap_bias = np.mean(boot_samples) - r
  if np.isinf(bootstrap_bias):
    bootstrap_bias = 0
  point = r - bootstrap_bias
  q = 100* (alpha/2.)
  return point, np.percentile(boot_samples, q),np.percentile(boot_samples, 100.-q)
  
percentile_bootstrap_simulation_results = pd.DataFrame([percentile_bootstrap_estimate(a, b, .05) for a, b in tqdm(datasets)], columns=['point', 'lower', 'upper'])
percentile_bootstrap_simulation_results['bias'] = percentile_bootstrap_simulation_results['point'] - TRUE_R
percentile_bootstrap_simulation_results['covered'] = (percentile_bootstrap_simulation_results['lower'] < TRUE_R) & (percentile_bootstrap_simulation_results['upper'] > TRUE_R)
```

# Variations on the bootstrap theme

Percentile bootstrap and Bayesian bootstrap

```
def percentile_bootstrap_estimate(n, d, alpha, n_sim=10000):
  r = naive_estimate(n, d)
  k = len(n)
  boot_samples = [naive_estimate(*resample(n, d)) for _ in range(n_sim)]
  boot_samples = [s for s in boot_samples if not np.isinf(s)]
  bootstrap_bias = np.mean(boot_samples) - r
  if np.isinf(bootstrap_bias):
    bootstrap_bias = 0
  point = r - bootstrap_bias
  q = 100* (alpha/2.)
  return point, np.percentile(boot_samples, q),np.percentile(boot_samples, 100.-q)
  
percentile_bootstrap_simulation_results = pd.DataFrame([percentile_bootstrap_estimate(a, b, .05) for a, b in tqdm(datasets)], columns=['point', 'lower', 'upper'])
percentile_bootstrap_simulation_results['bias'] = percentile_bootstrap_simulation_results['point'] - TRUE_R
percentile_bootstrap_simulation_results['covered'] = (percentile_bootstrap_simulation_results['lower'] < TRUE_R) & (percentile_bootstrap_simulation_results['upper'] > TRUE_R)
```

```
def bca_bootstrap_estimate(n, d, alpha, n_sim=10000):
  total_n, total_d = np.sum(n), np.sum(d)
  k = len(n)
  r = naive_estimate(n, d)
  r_i = (total_n - n) / (total_d - d) 
  d_i = r_i - np.mean(r_i)
  boot_samples = [naive_estimate(*resample(n, d)) for _ in range(n_sim)]
  boot_samples = [s for s in boot_samples if not np.isinf(s)]
  p0 =  np.sum(boot_samples < r) / n_sim
  z0 = norm.ppf(p0)
  a = (1./6) * (np.sum(d_i**3) / (np.sum(d_i**2))**(3./2.))
  if np.isnan(a):
    a = 0 # Why does this happen?
    print('A')
  alpha_half = (alpha/2.)
  p_low = norm.cdf(z0 + ((z0 + norm.ppf(alpha_half)) / (1. - a*(z0 + norm.ppf(alpha_half)))))
  p_high = norm.cdf(z0 + ((z0 + norm.ppf(1.-alpha_half)) / (1. - a*(z0 + norm.ppf(1.-alpha_half)))))
  return r, np.percentile(boot_samples, p_low*100.), np.percentile(boot_samples, p_high*100.)
  
bca_bootstrap_simulation_results = pd.DataFrame([bca_bootstrap_estimate(a, b, .05) for a, b in tqdm(datasets)], columns=['point', 'lower', 'upper'])
bca_bootstrap_simulation_results['bias'] = bca_bootstrap_simulation_results['point'] - TRUE_R
bca_bootstrap_simulation_results['covered'] = (bca_bootstrap_simulation_results['lower'] < TRUE_R) & (bca_bootstrap_simulation_results['upper'] > TRUE_R)
```

```
def bayesian_bootstrap_estimate(n, d, alpha, n_sim=10000):
  r = naive_estimate(n, d)
  k = len(n)
  w = np.random.dirichlet([1.]*len(n), n_sim)
  boot_samples = [naive_estimate(w_i*n, w_i*d) for w_i in w]
  boot_samples = [s for s in boot_samples if not np.isinf(s)]
  q = 100* (alpha/2.)
  return r, np.percentile(boot_samples, q),np.percentile(boot_samples, 100.-q)
  
bayes_bootstrap_simulation_results = pd.DataFrame([bayesian_bootstrap_estimate(a, b, .05) for a, b in tqdm(datasets)], columns=['point', 'lower', 'upper'])
bayes_bootstrap_simulation_results['bias'] = bayes_bootstrap_simulation_results['point'] - TRUE_R
bayes_bootstrap_simulation_results['covered'] = (bayes_bootstrap_simulation_results['lower'] < TRUE_R) & (bayes_bootstrap_simulation_results['upper'] > TRUE_R)
```

# Putting it together: Ratio analysis with the jackknife and the Bootstrap

formula

code example

show it's not biased and the variance is right

you can even do it in SQL - https://www.db-fiddle.com/f/vtVZzMKdNsDpQH9G3L7XwL/3

# Appendix: Some other approaches

Taylor series/delta method

Fieller?

BCa bootstrap

# More reading

Q's paper, review paper

Blog post

CASI

# Material

Naive estimate is biased https://en.wikipedia.org/wiki/Ratio_estimator
The sampling distribution can be nuts: https://en.wikipedia.org/wiki/Ratio_distribution

Probably the easiest thing for both bias correction and confidence intervals is the jackknife, and you can do it in SQL

- Open rates
- Cost per item
- Percentage difference between matched pairs

Do some simulations for an example with a funny-looking sampling distribution; show that the naive estimate is biased; jackknife estimates the variance correctly

Uncorrelated parto/nbinom distribution - revenue per...session?

$\text{test}$

```python
from scipy.stats import binom, pareto

numerator_dist = pareto(2)
denominator_dist = binom(2*10, 0.1)

def gen_data(n):
  return numerator_dist.rvs(n), denominator_dist.rvs(n)

def naive_estimate(n, d):
  return np.sum(n) / np.sum(d)
  
def jackknife_estimate(n, d):
  pass

naive_sampling_distribution_n_5 = [naive_estimate(n, d) for n, d in [gen_data(5) for _ in range(10000)]] # This is biased
jackknife_sampling_distribution_n_5 = [naive_estimate(n, d) for n, d in [gen_data(5) for _ in range(10000)]] # This is not (?)

```

# Other stuff

http://www.stat.cmu.edu/~hseltman/files/ratio.pdf

https://arxiv.org/pdf/0710.2024.pdf

https://stats.stackexchange.com/questions/16349/how-to-compute-the-confidence-interval-of-the-ratio-of-two-normal-means

https://i.stack.imgur.com/vO8Ip.png

https://statisticaloddsandends.wordpress.com/2019/02/20/what-is-the-jackknife/amp/

http://www.math.ntu.edu.tw/~hchen/teaching/LargeSample/references/Miller74jackknife.pdf

https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1015.9344&rep=rep1&type=pdf

https://www.researchgate.net/publication/220136520_Bootstrap_confidence_intervals_for_ratios_of_expectations - Bootstrap confidence intervals for ratios of expectations

```python
# Fieller vs Taylor method; assume independence
from scipy.stats import norm, sem
import numpy as np

n_sim = 1000

m_x = 1000
m_y = 1500

s_x = 1900
s_y = 1900

n_x = 100
n_y = 100

true_ratio = m_y / m_x

total_success_taylor = 0
total_success_fieller = 0

z = 1.96

for i in range(n_sim):
  x = np.clip(norm(m_x, s_x).rvs(n_x), 0, np.inf)
  y = np.clip(norm(m_y, s_y).rvs(n_y), 0, np.inf)
  
  point_ratio = np.mean(y) / np.mean(x)
  se_ratio = np.sqrt(((sem(x)**2 / np.mean(x)**2)) + (sem(y)**2 / np.mean(y)**2))

  l_taylor, r_taylor = point_ratio - z * se_ratio, point_ratio + z * se_ratio
  total_success_taylor += int(l_taylor < true_ratio < r_taylor)
  
  #t_unbounded = (np.mean(x)**2 / sem(x)**2) + (((np.mean(y) * sem(x)**2))**2 / (sem(x)**2 * sem(x)**2 * sem(y)**2))
  
  fieller_num_right = np.sqrt((np.mean(x)*np.mean(y))**2 - (np.mean(x)**2 - z*sem(x)) - (np.mean(y) - z*sem(y)**2))
  fieller_num_left = np.mean(x)*np.mean(y)
  fieller_denom = np.mean(x)**2 - z * sem(x)**2
  l_fieller = (fieller_num_left - fieller_num_right) / fieller_denom
  r_fieller = (fieller_num_left + fieller_num_right) / fieller_denom
  total_success_fieller += int(l_fieller < true_ratio < r_fieller)
print(1.*total_success_taylor / n_sim)
print(1.*total_success_fieller / n_sim)
```
