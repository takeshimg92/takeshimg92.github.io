---
layout: post
title: "PyMC3 makes Bayesian Inference in Python Easy"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: test.png
---

# Everything you need to run a Bayesian analysis

I've met lots of folks who are interested in Bayesian methods but don't know where to start; this is for them. It's only a little bit about code of PyMC3 (there's a lot of that in the docs [link]) and more about the process of how to use bayesian analysis to do stuff

For stats people but not Bayesians

Do not focus too much on frequentist/Bayesian diffs

Create a clear Bayesian Model building process; offload technical details to technical references

Why?
* Note access to other methods that it allows, like hierarchical
* Cite pragmatic statistics
* Cite a strong defense of the Bayesian perspective

# What does a Bayesian Analysis look like?

The absolute shortest, quickest, Hemingwayest description of Bayesian Analysis that I can think of is this:

> We have a question about some parameter, $\Theta$. We have some prior beliefs about $\Theta$, which is represented by the probability distribution $\mathbb{P}(\Theta)$. We collect some data, $X$, which we believe will tell us something additional about $\Theta$. We update our belief based on the data, using Bayes Rule to obtain $\mathbb{P}(\Theta \mid X)$, which is called the posterior distribution. We check to ensure the model does not fail to capture any relevant characteristics of the data. Armed with $\mathbb{P}(\Theta \mid X)$, we answer our question about $\Theta$ to the best of our knowledge.

This short synopsis of the Bayesian update process gives us a playbook for doing Bayesian statistics:

(1) Decide on a parameter of interest, $\Theta$, which we want to answer a question about.

(2) Specify the relationship between $\Theta$ and the data you could collect. This relationship is expressed as $\mathbb{P}(X \mid \Theta)$, which is called the data generating distribution.

(3) Specify a prior distribution $\mathbb{P}(\Theta)$ which represents our beliefs about $\Theta$ before observing the data.

(4) Go out and collect the data, $X$. Go on, you could probably use a break from reading anyhow. I'll wait here until you get back.

(5) Obtain the posterior distribution $\mathbb{P}(\Theta \mid X)$. We can do this analytically by doing some math, which is usually unpleasant unless you have a [conjugate prior]. If you don't want to do integrals today (and who can blame you?), you can obtain samples from the posterior by using [MCMC].

(6) Attempt to falsify the model

(7) You now have either a formula for $\mathbb{P}(\Theta \mid X)$, so you can answer your question.

# Ad spend analysis

We've collected a sample of customer orders, and we want to know about their mean and variance. According to tradition, we will denote the mean $\mu$ and the variance $\sigma$. That means that our parameter has two dimensions, $\Theta=(\mu, \sigma)$.

We believe that the ... are normally distributed. In that case, $\mathbb{P}(\Theta \mid X)$ is a $N(\mu, \sigma)$ distribution. Alternatively, we might write $X \sim N(\mu, \sigma)$. 

$\mu \sim N(100, 10)$

$\sigma \sim HalfNormal(1000000)$


```python
import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

x = np.random.normal(115, 5, size=1000)

with pm.Model() as normal_model:
  mean = pm.Normal('mean', mu=100, sigma=10)
  standard_deviation = pm.HalfNormal('standard_deviation', sigma=1000000)
  observations = pm.Normal('observations', mu=mean, sigma=standard_deviation, observed=x) # Observations is the pymc3 object, x is the vector of observations
  
  posterior_samples = pm.sample(draws=1000, tune=1000)
```

```python
sns.distplot(posterior_samples['mean'])

np.quantile(posterior_samples['mean'], .05)
```

# Detour: What happened when we call the sample function

intuitive MCMC

# How do we know it worked

## Diagnostic checks of the MCMC process

Traceplot

```
pm.traceplot(posterior_samples)
plt.show()
```

Sampling statistics for diagnosing issues

[Gelman-Rubin](https://pymc3-testing.readthedocs.io/en/rtd-docs/api/diagnostics.html#pymc3.diagnostics.gelman_rubin)

```python
print(pm.rhat(posterior_samples))
```

[Effective Sample size](https://pymc3-testing.readthedocs.io/en/rtd-docs/api/diagnostics.html#pymc3.diagnostics.effective_n)

```python
pm.ess(posterior_samples)
```

## Checking the observations against our model

Posterior Predictive, Model checks

```python
with normal_model:
  spp =  pm.sample_posterior_predictive(posterior_samples, 5000)
  
simulated_observations = spp['observations'] # 5000 data sets we might see under the posterior
```

Compare CDFs to the observed CDF

```
from statsmodels.distributions.empirical_distribution import ECDF

for sim_x in simulated_observations:
  plt.plot(ECDF(sim_x).x, ECDF(sim_x).y, color='blue', alpha=.1)

plt.plot(ECDF(x).x, ECDF(x).y, color='orange')
plt.show()
```

```
observed_mean = np.mean(x)
sim_means = np.array([np.mean(sim_x) for sim_x in simulated_observations])
sns.distplot(sim_means)
plt.axvline(observed_mean)
plt.show()

p = np.mean(observed_mean <= sim_means)
print(p)

observed_variance = np.var(x)
sim_vars = np.array([np.var(sim_x) for sim_x in simulated_observations])
sns.distplot(sim_vars)
plt.axvline(observed_variance)
plt.show()

p = np.mean(observed_variance <= sim_vars)
print(p)
```

```
observation_p_values = []

for i, observation in enumerate(x):
  sim_ith_observation = simulated_observations[:,i]
  p = np.mean(observation <= sim_ith_observation)
  observation_p_values.append(p)
  
sns.distplot(observation_p_values, fit=uniform)
plt.show()
```

# Good books on Bayesian Inference

Gelman
Kruschke

# Other cools Bayes-related libraries

emcee
bayes boot

# Appendix: Alternative model specification using summary statistics

```python
import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import sem

x = np.random.normal(115, 5, size=1000)

with pm.Model() as normal_model:
  mean = pm.Normal('mean', mu=100, sigma=10)
  standard_deviation = pm.HalfNormal('standard_deviation', sigma=1000000)
  observations = pm.Normal('observations', mu=mean, sigma=standard_deviation, observed=x) # Observations is the pymc3 object, x is the vector of observations
  
  posterior_samples = pm.sample(draws=1000)

with pm.Model() as reduced_model:
  mean = pm.Normal('mean', mu=100, sigma=10)
  observations = pm.Normal('observations', mu=mean, sigma=sem(x), observed=np.mean(x)) # Observations is the pymc3 object, x is the vector of observations
  
  reduced_posterior_samples = pm.sample(draws=1000)

sns.distplot(posterior_samples['mean'])
sns.distplot(reduced_posterior_samples['mean'])
plt.show()
```
