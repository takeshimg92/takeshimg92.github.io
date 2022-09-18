
# Quantity ☯ Quality

if set our sights on quality alone we become perfectionists; if we aim for pure quantity we will produce a large volume of low-value output

It's often easy to measure quantity so metrics usuall do; it's usually more expensive to measure quality

but we need to

Often, we can only measure the quality of a subset of outputs due to the cost, so we need to do some statistics

luckily its easy

# A touch of statistics makes QA much easier

QA questions related to estimating the number of defective units pop up all the time, even outside the manufacturing context

QA is expensive, so we should be careful about how much of it we do

# Estimating the proportion of defects in a finite population: The finite population correction

Finite population correction

```python
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import sem

x = np.array([1]*3000 + [0]*3000)

n_population = len(x)

n_sample = 1000

n_sim = 10000

fpc = np.sqrt((n_population - n_sample) / (n_population - 1))

samples = [np.random.choice(x, size=n_sample, replace=False) for _ in range(n_sim)]
sampled_means = [np.mean(s) for s in samples]
sampled_sems = [sem(s) for s in samples]
sampled_corrected_sems = [sem(s) * fpc for s in samples]

actual_sem = np.std(sampled_means)

sns.distplot(sampled_sems)
sns.distplot(sampled_corrected_sems)
plt.axvline(actual_sem)
plt.show()

sns.distplot(sampled_sems - actual_sem)
sns.distplot(sampled_corrected_sems - actual_sem)
plt.axvline(np.mean(sampled_corrected_sems - actual_sem))
plt.axvline(np.mean(sampled_sems - actual_sem))
plt.show()

```

Plot (n / N) vs standard error inflation (Inverse of the correction factor?)
Swap X with a different distribution
Is this exact for the binomial distribution!? Note that the correct SE is correct event for tiny sample sizes

Bayesian view: Predictive posterior for the unobserved values for binomial (exact I think), else bootstrap unknown values
(Draw a beta, then draw a binomial - so beta binomial)

# Prediction the number defective units based on the sample: the Beta-Binomial model

```python
from scipy.stats import betabinom, beta, binomial

betabinom(a, b, n).rvs(k)

binom(n, beta(a, b).rvs(k)).rvs() # Same deal

```

# A shortcut when defects are rare events: The rule of three

compare with beta interval using jeffrey's prior
