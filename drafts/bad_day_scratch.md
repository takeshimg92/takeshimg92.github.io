---
layout: post
title: "Beyond the mean: What's the worst day I need to plan for? How did my experiment affect my distribution of revenue? Quantiles and their confidence intervals in Python"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: bad_day.png
---

*Analytics teams are often confronted with a wide variety of metrics that change every day, week, and month. In order to stay on top of the business, we'd like to know what the "normal range" of a metric is, so we can understand when something unexpected is happening. Additionally, this will let you understand the upper and lower boundaries of your metric - it will help you understand what kinds of "worst cases" you might need to plan for. All of these are instances of quantile estimation - we'll introduce the statistics of quantile estimates and quantile regression.*

What will my "real bad day" look like? How much do I need to keep in reserve to stay safe in that case? Looking forward, what observations are unusually bad? What is the size of the bottom - top difference? How can I establish "normal bounds" so I can know when things are not normal?

# An example: What does a "normal day" of web traffic look like?

If you run a website, you've probably spent time thinking about how many people actually saw your website - that's the point of having one, after all. Let's imagine that you are the supernaturally gifted CEO of the greatest tech company in history, [Zombo.com](http://zombo.com/). You've collected the number of daily visitors over 100 days of traffic to your website, and you want to use it to do some planning. 

[Picture of traffic]

Specifically, you want to know: On an "ordinary" day, what's the lowest or highest number of visitors I might expect? This sort of information is a valuable planning tool as you monitor your web traffic. For example, you might want to know the highest total amount of traffic you might get, so you can ensure you have enough bandwidth for all your visitors on your highest-traffic days. Or you might keep this range in mind as you look over each days traffic report, in order to see whether or not you had an abnormally high or low amount of traffic, indicating something has changed and you need to re-evaluate your strategy.

# Upper and lower bounds for the "ordinary" range - Quantiles

## The Population and Sample Quantiles

Let's take our main question, and be a little more specific with it. We want to know:

> On an "ordinary" day, what's the lowest or highest number of visitors I might expect?

One way we could translate this question into more quantitative language is to reframe it as:

> Can we define a range of values, so that almost all (say, 95%) of of daily observations will fall within this range?

Let's start by thinking about the **lower side** of the range, the lowest number of visitors we might expect on a day. We'd like to know a value such that 97.5% of future observations will be above it (we'll do the same on upper end, so overall 2.5 × 2=5% of observations will be outside this range). This value has a name - it is called the 0.025-quantile. More generally, the $q$-quantile is the value such that $q$% of observations are less than it. So, for example, the 0.5-quantile is the median, and the 0.25-quantile is the first quartile.

As usual, we want the population quantile, but we only have a sample to estimate from. We can compute the sample quantile using [the appropriate numpy method](https://numpy.org/devdocs/reference/generated/numpy.quantile.html). Plotting these on our chart from before, we see they look the way we expect:

[Plot of data with lower and upper sample quantiles marked]

That is, the sample 0.025- and 0.975-quantiles cover the central 95% of the data.

## A Simple CI for a quantile

Just now we did two inferences, computing the sample quantiles of our dataset. As good students of statistics we should try to understand how much uncertainty there is around these estimates - we want to compute a standard error or a confidence interval.

1. Sort the observations.
2. l = 
3. u = 

```
import seaborn as sns
from scipy.stats import norm, binom, pareto
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def compute_quantile_ci(q, data, alpha):
  data = sorted(data)
  n = len(data)
  l = int(binom(n, p=q).ppf(alpha/2))
  u = int(binom(n, p=q).ppf(1.-alpha/2)) - 1 
  return data[l], data[u]

gen_dist = pareto(2)

n = 100

Q = np.linspace(0.0275, 0.975)
alpha = .05

coverage = []

for q in tqdm(Q):
  TRUE_QUANTILE = gen_dist.ppf(q)

  n_sim = 200
  results = 0
  lower_dist = []
  upper_dist = []

  for _ in range(n_sim):
    data = gen_dist.rvs(n)
    l, u = compute_quantile_ci(q, data, alpha)
    if l <= TRUE_QUANTILE <= u:
      results += 1
    lower_dist.append(l)
    upper_dist.append(u)
      
  lower_dist = np.array(lower_dist)
  upper_dist = np.array(upper_dist)
      
  coverage += [results / n_sim]
  
plt.plot(data)
plt.show()

sns.distplot(data, kde=False)
plt.show()
  
plt.plot(Q, coverage)
plt.show()
```

```
http://mqala.co.za/veed/Introduction%20to%20Robust%20Estimation%20and%20Hypothesis%20Testing.pdf

def mj_quantile_se(data, q): # Introduction to Robust Estimation and Hypothesis testing, Wilcox, 3.5.3
    data = np.sort(data)
    n = len(data)
    I = np.arange(n)+1
    m = np.round(q*n + 0.5)
    cdf = beta(m-1, n-m).cdf 
    W = cdf(I/n) - cdf((I-1)/n)
    c2 = np.sum(W*data**2)
    c1 = np.sum(W*data)
    se = np.sqrt(c2 - c1**2)
    return se

import seaborn as sns
from scipy.stats import norm, binom, pareto
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def compute_quantile_ci(q, data, alpha):
  se = mj_quantile_se(data, q)
  z = norm.ppf(1.-alpha/2)
  emp_q = np.quantile(data, q)
  return emp_q - z*se, emp_q + z*se

gen_dist = pareto(2)
#gen_dist = norm(0, 1)

n = 100

Q = np.linspace(0.0275, 0.975)
alpha = .05

coverage = []

for q in tqdm(Q):
  TRUE_QUANTILE = gen_dist.ppf(q)

  n_sim = 200
  results = 0
  lower_dist = []
  upper_dist = []

  for _ in range(n_sim):
    data = gen_dist.rvs(n)
    l, u = compute_quantile_ci(q, data, alpha)
    if l <= TRUE_QUANTILE <= u:
      results += 1
    lower_dist.append(l)
    upper_dist.append(u)
      
  lower_dist = np.array(lower_dist)
  upper_dist = np.array(upper_dist)
      
  coverage += [results / n_sim]
  
plt.plot(data)
plt.show()

sns.distplot(data, kde=False)
plt.show()
  
plt.plot(Q, coverage)
plt.axhline(1.-alpha, linestyle='dotted')
plt.ylim(0, 1)
plt.show()

###### Quantile curve and CIs
gen_dist = poisson(1000) # Consider skewnorm? to show tail diffs
alpha = .05

data = gen_dist.rvs(100)
ppfs = np.quantile(data, Q)
SEs = np.array([mj_quantile_se(data, q) for q in Q])
CIs = [compute_quantile_ci(q, data, .05) for q in Q]
L, U = zip(*CIs)
plt.fill_between(Q, L, U, alpha=.5)
plt.plot(Q, ppfs)
plt.show()

# Diff of two quantile curves

data2 = gen_dist.rvs(100)
ppfs2 = np.quantile(data2, Q)
SEs2 = np.array([mj_quantile_se(data2, q) for q in Q])
diff = ppfs2 - ppfs
diff_ses = np.sqrt(SEs**2 + SEs2**2)
z = norm.ppf(1.-(alpha/len(Q))) # Bonferroni
plt.fill_between(Q, diff - z*diff_ses, 
                    diff + z*diff_ses, alpha=.5)
plt.plot(Q, diff)
plt.axhline(0, linestyle='dotted')
plt.show()
```

## Our model assumes every day has the same distribution, which is probably not true

So far, we've put together a method that tells us: What is the

Howver, the range may change depending on some covariates

For example spend

Unsurprisingly the quickest way to do this is with linear regression, though there are other methods too

# Including covariates - Quantile Regression

# Appendix: Imports and data generation

```python
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import datetime
import pandas as pd

dates = [datetime.datetime(year=2020, month=1, day=1) + datetime.timedelta(days=i) for i in range(365)]
spend = np.abs(np.random.normal(0, 1, len(dates)))
visitors = np.random.poisson(5 + 5*spend)

traffic_df = pd.DataFrame({'date': dates, 'spend': spend, 'visitors': visitors})
```

# Appendix: Comparison of Quantile CIs

------------------------------------------------------------------

# Estimating quantiles

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mquantiles.html

I. Quantiles of a sample

1. Jackknife and many flavors of Bootstrap

Noted as an example for the bootstrap in Shalizi's article

https://www.americanscientist.org/article/the-bootstrap

Which bootstrap works best? Is there a pretty way of writing the jackknife estimate

Bootstrap diagnostic - https://www.cs.cmu.edu/~atalwalk/boot_diag_kdd_final.pdf

```
from scipy.stats import norm, sem, skew, t, lognorm
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from scipy.stats.mstats import mquantiles

TRUE_VALUE = lognorm(s=1).ppf(.01)

def gen_data(n):
  return lognorm(s=1).rvs(n)
  
datasets = [gen_data(1000) for _ in range(1000)] # what happens to each of these methods as we vary the sample size

def percentile_bootstrap_estimate(x, alpha, n_sim=2000):
  s = np.percentile(x, 1)
  boot_samples = [np.percentile(np.random.choice(x, len(x)), 1) for _ in range(n_sim)]
  q = 100* (alpha/2.)
  return x, np.percentile(boot_samples, q),np.percentile(boot_samples, 100.-q)
  
percentile_bootstrap_simulation_results = pd.DataFrame([percentile_bootstrap_estimate(x, .05) for x in tqdm(datasets)], columns=['point', 'lower', 'upper'])
percentile_bootstrap_simulation_results['covered'] = (percentile_bootstrap_simulation_results['lower'] < TRUE_VALUE) & (percentile_bootstrap_simulation_results['upper'] > TRUE_VALUE)
print(percentile_bootstrap_simulation_results.covered.mean())

def standard_bootstrap_estimate(x, alpha, n_sim=2000):
  s = np.percentile(x, 1)
  k = len(x)
  boot_samples = [np.percentile(np.random.choice(x, len(x)), 1) for _ in range(n_sim)]
  se = np.std(boot_samples)
  t_val = t(k-1).interval(1.-alpha)[1]
  return s, s - t_val * se, s + t_val * se
  
standard_bootstrap_simulation_results = pd.DataFrame([standard_bootstrap_estimate(x, .05) for x in tqdm(datasets)], columns=['point', 'lower', 'upper'])
standard_bootstrap_simulation_results['covered'] = (standard_bootstrap_simulation_results['lower'] < TRUE_VALUE) & (standard_bootstrap_simulation_results['upper'] > TRUE_VALUE)
print(standard_bootstrap_simulation_results.covered.mean())

def bca_bootstrap_estimate(x, alpha, n_sim=2000):
  k = len(x)
  r = np.percentile(x, 1)
  r_i = (np.sum(x) - x)/(k-1)
  d_i = r_i - np.mean(r_i)
  boot_samples = [np.percentile(np.random.choice(x, len(x)), 1) for _ in range(n_sim)]
  p0 =  np.sum(boot_samples < r) / n_sim
  z0 = norm.ppf(p0)
  a = (1./6) * (np.sum(d_i**3) / (np.sum(d_i**2))**(3./2.))
  alpha_half = (alpha/2.)
  p_low = norm.cdf(z0 + ((z0 + norm.ppf(alpha_half)) / (1. - a*(z0 + norm.ppf(alpha_half)))))
  p_high = norm.cdf(z0 + ((z0 + norm.ppf(1.-alpha_half)) / (1. - a*(z0 + norm.ppf(1.-alpha_half)))))
  return r, np.percentile(boot_samples, p_low*100.), np.percentile(boot_samples, p_high*100.)
  
bca_bootstrap_simulation_results = pd.DataFrame([bca_bootstrap_estimate(x, .05) for x in tqdm(datasets)], columns=['point', 'lower', 'upper'])
bca_bootstrap_simulation_results['covered'] = (bca_bootstrap_simulation_results['lower'] < TRUE_VALUE) & (bca_bootstrap_simulation_results['upper'] > TRUE_VALUE)
print(bca_bootstrap_simulation_results.covered.mean())
```

2. Exact methods ("Binomial idea")

https://staff.math.su.se/hoehle/blog/2016/10/23/quantileCI.html

https://stats.stackexchange.com/questions/99829/how-to-obtain-a-confidence-interval-for-a-percentile

```
from scipy.stats import norm, binom, pareto
import numpy as np

MU = 100
S = 1100

#gen_dist = norm(MU, S)

gen_dist = pareto(2)

n = 100

q = 0.95
alpha = .05

TRUE_QUANTILE = gen_dist.ppf(q)

l = int(binom(n, p=q).ppf(alpha/2))
u = int(binom(n, p=q).ppf(1.-alpha/2) ) # ???? Check against R's qbinom

n_sim = 10000
results = 0
lower_dist = []
upper_dist = []

for _ in range(n_sim):
  data = sorted(gen_dist.rvs(n))
  if data[l] <= TRUE_QUANTILE <= data[u]:
    results += 1
  lower_dist.append(data[l])
  upper_dist.append(data[u])
    
lower_dist = np.array(lower_dist)
upper_dist = np.array(upper_dist)
    
print(results / n_sim)
```

3. Easy mode - Asymptotic estimate

Cookbook estimate: http://www.tqmp.org/RegularArticles/vol10-2/p107/p107.pdf

Looks like https://stats.stackexchange.com/a/99833/29694 where we assume the data is normally distributed

Eq 25

4. Methods from Wasserman's all of nonparametric statistics

https://web.stanford.edu/class/ee378a/books/book2.pdf
Use ebook p. 25 to estimate upper and lower bounds on the CDF, then invert them at `q`.

Example 2.17 - 

3.7 Theorem

# For when you want to relate covariates to quantiles - Conditional quantiles

5. Quantile regression

https://www.statsmodels.org/dev/examples/notebooks/generated/quantile_regression.html - Linear models of quantiles by picking a different loss function

https://jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf - An ML approach

We can check these using cross-validation on the probability of being greater than q, which is cool

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor

Other approaches: Conditional variance - Section 5 - https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/24/lecture-24--25.pdf

III. This isn't magic

look out for small samples and edge cases
