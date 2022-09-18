---
layout: post
title: "A Crash Course in Bayesian Analysis with PyMC3"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: test.png
---

*blurb*

# Intro

I've met lots of folks who are interested in Bayesian methods but don't know where to start; this is for them. It's only a little bit about code of PyMC3 (there's a lot of that in the docs [link]) and more about the process of how to use bayesian analysis to do stuff

For stats people but not Bayesians

Do not focus too much on frequentist/Bayesian diffs

Create a clear Bayesian Model building process; offload technical details to technical references

Why?

* Note access to other methods that it allows, like hierarchical models, GPs, Bandits
* Often has a very intuitive interpretation
* Cite pragmatic statistics
* Cite a strong defense of the Bayesian perspective

# The short version: A synopsis of the Bayesian Analysis process

The absolute shortest, quickest, Hemingwayest description of Bayesian Analysis that I can think of is this:

We have a question about some parameter, $\Theta$. We have some prior beliefs about $\Theta$, which is represented by the probability distribution $\mathbb{P}(\Theta)$. We collect some data, $X$, which we believe will tell us something additional about $\Theta$. We update our belief based on the data, using Bayes Rule to obtain $\mathbb{P}(\Theta \mid X)$, which is called the posterior distribution. We check to ensure the model does not fail to capture any relevant characteristics of the data. Armed with $\mathbb{P}(\Theta \mid X)$, we answer our question about $\Theta$ to the best of our knowledge.

This short synopsis of the Bayesian update process gives us a playbook for doing Bayesian statistics:

(1) Decide on a parameter of interest, $\Theta$, which we want to answer a question about. This might be a multidimensional, like a vector of numbers or a (mean, variance) pair.

(2) Specify the relationship between $\Theta$ and the data you could collect. This relationship is expressed as $\mathbb{P}(X \mid \Theta)$, which is called the data generating distribution.

(3) Specify a prior distribution $\mathbb{P}(\Theta)$ which represents our beliefs about $\Theta$. 

(4) Go out and collect the data, $X$. Go on, you could probably use a break from reading anyhow. I'll wait here until you get back.

(5) Obtain the posterior distribution $\mathbb{P}(\Theta \mid X)$. We can do this analytically by doing some math, which is usually unpleasant unless you have a [conjugate prior]. If you don't want to do integrals today (and who can blame you?), you can obtain samples from the posterior by using [MCMC].

(6) Check how well your model fits the data. The fitted model has implications about the kind of data we would expect to see, if the model were true. We can use the fitted model to generate data sets that we _could_ have seen if the model were true, and compare the actual data to this. If we find that our model wouldn't produce data like the observations we actually saw, we should reconsider our model before we use it to make inferences.

(7) You now have either a formula for $\mathbb{P}(\Theta \mid X)$, so you can answer your question.

# A running example: Quality control from a large batch

Want to know whether we're very sure that param > target

Formulating a question in a Bayesian analysis

# Specifying the model: The data-generating story and our prior

```python
import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

with pm.Model() as negative_binomial_model:
  mu = pm.HalfNormal('mean', sigma=1000)
  alpha = pm.HalfNormal('alpha', sigma=1000)
  observations = pm.NegativeBinomial('observations', mu=mu, alpha=alpha, observed=observed_x)
```  

# MCMC

## What the heck even is MCMC?

Examples:
https://chi-feng.github.io/mcmc-demo/app.html

## Running the Sampler

```python
with negative_binomial_model:
  posterior_samples = pm.sample(draws=1000, tune=100000)
```

default is nuts

## Diagnostic checks

Traceplot

```python
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

# Model checking: Can we find examples where the model doesn't fit?

## Model checks via simulation

Posterior Predictive, Model checks

```python
with negative_binomial_model:
  spp =  pm.sample_posterior_predictive(posterior_samples, 5000)
  
simulated_observations = spp['observations'] # 5000 data sets we might see under the posterior
```

Compare KDE countours to the observed one

```python
for sim_x in simulated_observations[:1000]:
  sns.kdeplot(sim_x, color='blue', alpha=.1)

sns.kdeplot(observed_x, color='orange')
plt.show()
```

Compare CDFs to the observed CDF

```python
from statsmodels.distributions.empirical_distribution import ECDF

for sim_x in simulated_observations:
  plt.plot(ECDF(sim_x).x, ECDF(sim_x).y, color='blue', alpha=.1)

plt.plot(ECDF(observed_x).x, ECDF(observed_x).y, color='orange')
plt.show()
```

Compare simulated statistics to actual one; are we likely to be representing the mean and variance correctly?

```python
observed_mean = np.mean(observed_x)
sim_means = np.array([np.mean(sim_x) for sim_x in simulated_observations])
sns.distplot(sim_means)
plt.axvline(observed_mean)
plt.show()

p = np.mean(observed_mean <= sim_means)
print(p)

observed_variance = np.var(observed_x)
sim_vars = np.array([np.var(sim_x) for sim_x in simulated_observations])
sns.distplot(sim_vars)
plt.axvline(observed_variance)
plt.show()

p = np.mean(observed_variance <= sim_vars)
print(p)
```

```python
from scipy.stats import uniform

observation_p_values = []

for i, observation in enumerate(observed_x):
  sim_ith_observation = simulated_observations[:,i]
  p = np.mean(observation <= sim_ith_observation)
  observation_p_values.append(p)
  
observation_p_values = np.array(observation_p_values)
sns.distplot(observation_p_values, fit=uniform, kde=False, norm_hist=True)
plt.show()

print('Proportion of observations with significant P-values:', sum(observation_p_values) / len(observed_x))
```

## What do we do when we find evidence of a bad fit? Model expansion

An example of an overly-restrictive prior for the variance, found by a check above; led to more diffuse prior over the variance

Generally, the goal is the make the prior more expansive, to include the entire universe of plausible models. the goal is not to overfit

## Some other things to consider: Model selection and Bayesian Model averaging

The goal above is to try and poke holes in our model so we can improve it; starts with a model that may be too simple and attempts to expand it

Gelman's recommendation of continuous model expansion: "building a larger model that includes the separate models as separate cases" https://statmodeling.stat.columbia.edu/2017/01/05/30811/

This is not really what we're doing here but it's worth knowing about

https://docs.pymc.io/en/v3/pymc-examples/examples/diagnostics_and_criticism/model_averaging.html

# Using the posterior samples to answer our question

```python
np.quantile(posterior_samples['mean'], .05)
```

# Appendix: How the data was actually generated

```python
from scipy.stats import skellam

observed_x = skellam(100, 10).rvs(100)
```

# Appendix: Falsification and Bayesian models

G+S

Model comparison

# Appendix: Frequency properties in Bayesian analysis
