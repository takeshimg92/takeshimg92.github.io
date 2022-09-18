---
layout: post
title: "Generalizing from biased samples with Multilvel regression with Poststratification using PyMC3"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: jellybeans.png
---

*Often, we would like to make an inference about a population, but we can only sample from the population using a biased sampling process, like an email survey. In this case, the sample statistics, like the mean, will be misleading since the sample will be biased. A popular, powerful technique that lets us correct this bias is Multilevel Regression with Poststratification. We explore the assumptions and theory of MRP, and we implement it in Python using PyMC3.*

# The problem: Our sample doesn't look like the population we want to understand

In most practical settings, we can't inspect every member of a group of interest. We count events, track actions, and survey opinions of individuals to make generalizations to the population that the individual came from. This is the process of statistical inference from the data at hand, which we spend so much of our time trying to do well. For example, perhaps you run a startup and you'd like to survey your users to understand if they'd be interested in new product feature you've been thinking about. Developing a new feature is pretty costly, so you only want to do it if a large portion of your user base will be interested in it. You send an email survey to a small number of users, and you'll use that to infer what your overall user base thinks of the idea.

In the simplest case, every user has the same likelihood of responding to your survey. In that case, the average member of your sample looks like the average member of your user base, and you have a [simple random sample](https://en.wikipedia.org/wiki/Simple_random_sample) to which you can apply all the usual analyses. For example, in this case the sample approval rate is a reasonable estimate of the population approval rate.

[Short tour of the data - show a table of the observed and define a "cell"]

Simple random sampling is often not an option for someone running such a survey. In most situations like this, every user is not equally likely to respond - for example, there's a good chance that your most enthusiastic users are more likely to respond to the survey. This leaves you with an estimate that over-represents these enthusiastic users. In this case, the usual estimate of the approval rate (the sample approval rate) is not an exact reflection of the population approval rate, because our sampling was not simply random. Instead, we'll need to do some work to uncover how our sample differs from the population of users, so we can account for the bias in our sampling process.

# The solution: Poststratification

We'll correct this problem using a technique called [poststratification](https://online.stat.psu.edu/stat506/lesson/6/6.3). Poststratification goes something like this:

- Collect your sample, which is biased because it oversamples some subgroups and undersamples others. For example, perhaps your survey is likely to be easier to answer for some demographics, but harder for others.
- Use the data to estimate the mean result (such as approval rating) for each subgroup.
- Compute the population mean by calculating a weighted average of the subgroup means, using the proportion of each subgroup in the population as the weight.

The reason that this technique has such a fancy sounding name is because it assumes that users are sample from a bunch of discrete subgroups (the strata), and we are adjusting the observed average after we collected the data using the strata information (so, we are doing the adjustment "after subgroup analysis", or "post-stratification").

For our "do our users approve of the potential new feature" problem, poststratification might go like this:

- Send out your survey to all your users, or a simple random sample of them. For a bunch of different reasons, a lot of them will not repond, causing bias in your sample.
- Your users fall into a bunch of pre-determined subgroups based on their region and previous level of use. See the next section for where these subgroups might come from; for now, assume we know them a priori. Calculate the sample approval rate for each subgroup.
- Estimate the population mean by calculating a weighted average of all the subgroup approval rates, where the weight corresponds to what percentage of the population that subgroup comprises.

[Symbolically show calculation of final estimate]

There are a number of ways that we can perform poststratification. The technique above is about the simplest kind that I can imagine - we estimate the subgroup average using the sample average, and use that for the final weighted average. However, we can often do a little better than this. In particular, we can get better estimates of the subgroup means by using a Bayesian technique called multilevel (or hierarchical) regression, leading us to [Multilevel regression with poststratification](https://en.wikipedia.org/wiki/Multilevel_regression_with_poststratification). At this time, MRP is one of the state-of-the-art methods for generalizing samples of public opinion like polls. In 2016, [Wang et al](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/04/forecasting-with-nonrepresentative-polls.pdf) demonstrated the power of this technique by showing that it could be used to accurately predict US national public opinion from a highly nonrepresentative survey. In that case, the sample consisted of responders to an Xbox Live poll, which very strongly oversampled certain subgroups (like young men) and undersampled others (like older women). However, using MRP the authors were able to understand the bias in the data and adjust the survey results accordingly.

# What are these subgroups, exactly? Where do they come from?

The explanation just now mentioned "subgroups", without explaining where such a grouping might come from. If you regularly analyze some population, like the set of users for your website, you might already have some subgroups in mind. There might be certain demographic variables that you regularly use to group your observations. When we introduced the website survey example, we mentioned groups like "enthusiastic users". In the Xbox example, the subgroups are defined by familiar US Census categories, like gender, age, and race. What motivates this particular set of subgroups? 

If we return to our original goal, we see that we wanted to make a non-representative sample into a representative one. The problem is that some of our users do not respond to the survey; some of our observations are "missing" in the sense that we requested them by sending a survey, but never received them. The main question is what causes these missing responses. 

One argument might be that sometimes folks don't have time to respond to a survey, and everyone has about the same chance of being a non-responder. In this case, it's actually a non-issue, because we still have a simple random sample on our hands (though maybe a smaller one than we would like). We can kind of imagine this like sampling with an extra step: We randomly sample some fixed percentage of the population (send emails), then each member of our sample has a fixed percentage chance of disappearing from the sample (everyone flips a coin or rolls a die to figure out if they respond to the survey) before we get the data.

Another possibility is that certain groups possess attributes which make them less likely to respond (for example: being a less frequent user of the website --> more likely to have sent your survey to the spam folder). We can still imagine this as "sampling with an extra step", but in this case the probability of non-response varies depending on a user's attributes, rather than depending on an event with a fixed chance of success like a coin toss. This kind of bias even has a fancy name, which is always fun. We call this [differential non-response bias](https://methods.sagepub.com/reference/encyclopedia-of-survey-research-methods/n136.xml), a great phrase if you really need to convince someone that you always know the most difficult way to say a simple thing.

These two possibilities and one more give us three mutually exclusive assumptions to choose from in thinking about the sampling bias in our data:

- [Missing completely at random ](https://en.wikipedia.org/wiki/Missing_data#Missing_completely_at_random): Data is MCAR when the probabilty of non-response is the same for everyone. This is the "everyone flips a coin" scenario. It's the same as simple random sampling, making it the easiest to analyze. Unfortunately, it's often not true.
- [Missing at random](https://en.wikipedia.org/wiki/Missing_data#Missing_at_random): Data is MCAR when the probability of non-response depends on a units attributes. This is the assumption we'll make when we analyze our survey.
- [Missing not at random](https://en.wikipedia.org/wiki/Missing_data#Missing_not_at_random): Data is MNAR when a unit's probability of non-response depends on its response value. For example, if some users were "superfans", and "superfans" are much more likely to respond, but superfan-ness is not reflected in any attributes we have. In this case, we can't perform a correction using the unit's attributes, and would need to adopt a different approach. We might treat the data as censored in this case, which often makes things more complicated. For now, we'll assume this is not tre.

Our analysis assumes that the survey data is missing at random - this is an assumption, and is _not_ verifiable from the data that we have collected. For example, there's no procedure we can run to determine whether we actually controlled for all the attributes we needed to, or that the data is not MNAR. However, even so, the MAR assumption is usually much more plausible than the MCAR assumption.

We've mentioned "attributes" of a unit a number of times in this description. For the purposes of this analysis, we'll assume these attributes are categorical. This wiggles us out of having to think about how we should model continuous attributes, which we'll leave for another day. In some cases, it is appropriate to discretize continuous values, which means that the model presented could at least be a useful baseline. An analyst is free to choose how to take the attributes of interest and coarsen them appropriately, though we should note that this involves assumptions we should think carefully about.

# The first step is admitting that you have a problem: Understanding if a sample is non-representative

Actually let's move all this to the appendix: What we care about is how our sample doesn't look like the population

So far, we've assumed that we don't have a simple random sample on our hands based on a priori reasoning. We're almost certainly right that our sample is not a simple random one; randomness is hard to achieve, and it is immensely likely that we did not stumble into collecting a representative sample by accident. Even so, it's valuable to aid our intuition by looking at how we know the sample doesn't seem representative. A sample is representative if each subgroup's proportion is what we'd expect based on the population weight.

What kinds of users did we oversample? undersample? 

```python
n = all_subgroups_df['total_responders'].sum()

region_df = all_subgroups_df.groupby('name_region').sum()
region_df['sampling_ratio'] =  (region_df['total_responders'] / n) / region_df['pop_weight']
for r, s in zip(region_df.index, region_df['sampling_ratio']):
  print('For region {0} we sampled {1}x the expected number'.format(r, s))

freq_df = all_subgroups_df.groupby('name_frequency').sum()
freq_df['sampling_ratio'] =  (freq_df['total_responders'] / n) / freq_df['pop_weight']
for r, s in zip(freq_df.index, freq_df['sampling_ratio']):
  print('For frequency group {0} we sampled {1}x the expected number'.format(r, s))
```


# Post-stratification with a Logit model in statsmodels

So our sample isn't representative. Now what?

```python
naive_estimate = all_subgroups_df['total_approve'].sum() / all_subgroups_df['total_responders'].sum()
print('The observed approval rate is: {0}'.format(naive_estimate))
```
```
The observed approval rate is: 
```

```python
positive_examples_df = all_subgroups_df[['name_region', 'name_frequency']]
positive_examples_df['y'] = 1
positive_examples_df['n'] = all_subgroups_df['total_approve']
negative_examples_df = all_subgroups_df[['name_region', 'name_frequency']]
negative_examples_df['y'] = 0
negative_examples_df['n'] = all_subgroups_df['total_responders'] - all_subgroups_df['total_approve']

logit_df = pd.concat((positive_examples_df, negative_examples_df))

logit_model = smf.glm('y ~ name_region + C(name_frequency)', logit_df, n_trials=logit_df['n'], family=sm.families.Binomial())
logit_fit = logit_model.fit()

logit_predictions = logit_fit.get_prediction(all_subgroups_df).summary_frame()

print('Point estimate: ', np.dot(all_subgroups_df['pop_weight'], logit_predictions['mean']))
print('Point SE: ', np.sqrt(np.dot(all_subgroups_df['pop_weight']**2, logit_predictions['mean_se']**2)))
```

# What assumptions did we make just now?

## We have controlled for all the covariates

And this is not testable

## The true model is the logit model

This is not ideal, though it's not ridiculous

# Preparing for a multilevel model - a Bayesian Logit model with PyMC3

```python
unique_regions = all_subgroups_df['name_region'].unique()
region_lookup = {v: i for i, v in enumerate(unique_regions)}
region_idx = [region_lookup[v] for v in all_subgroups_df['name_region']]

unique_freq = all_subgroups_df['name_frequency'].unique()
freq_lookup = {v: i for i, v in enumerate(unique_freq)}
freq_idx = [freq_lookup[v] for v in all_subgroups_df['name_frequency']]

with pm.Model() as unpooled_model:
  a = pm.Normal('a', mu=0, sigma=100)
  a_region = pm.Normal('a_region', 0, sigma=100, shape=len(unique_regions))
  a_freq = pm.Normal('a_freq', 0, sigma=100, shape=len(unique_freq))
  
  response_est = a + a_region[region_idx] + a_freq[freq_idx]
  x = pm.Binomial('x', n=all_subgroups_df['total_responders'].values, p=pm.math.invlogit(response_est), observed=all_subgroups_df['total_approve'].values)
  unpooled_trace = pm.sample(2000)

predicted_responses = []

for a_, a_r, a_f in zip(unpooled_trace['a'], unpooled_trace['a_region'], unpooled_trace['a_freq']):
  predicted_responses.append(expit(a_ + a_r[region_idx] + a_f[freq_idx]))
  
predicted_responses = np.array(predicted_responses)

all_subgroups_df['mean_unpooled'] = np.mean(predicted_responses, axis=0)
all_subgroups_df['low_unpooled'] = np.quantile(predicted_responses, .025, axis=0)
all_subgroups_df['high_unpooled'] = np.quantile(predicted_responses, .975, axis=0)
```

```python
plt.scatter(all_subgroups_df['total_approve'] / all_subgroups_df['total_responders'], all_subgroups_df['mean_unpooled'])
plt.vlines(all_subgroups_df['total_approve'] / all_subgroups_df['total_responders'], all_subgroups_df['low_unpooled'], all_subgroups_df['high_unpooled'])
plt.plot([0, 1],[0,1], linestyle='dotted')
plt.show()

```

# Hierarchical logit

```python
with pm.Model() as partial_pooled_model:
  a = pm.Normal('a', mu=0, sigma=100)
  sigma_region = pm.HalfNormal('sigma_region', 5.)
  sigma_freq = pm.HalfNormal('sigma_freq', 5.)
  a_region = pm.Normal('a_region', mu=0, sigma=sigma_region, shape=len(unique_regions))
  a_freq = pm.Normal('a_freq', mu=0, sigma=sigma_freq, shape=len(unique_freq))
  
  response_est = a + a_region[region_idx] + a_freq[freq_idx]
  x = pm.Binomial('x', n=all_subgroups_df['total_responders'], p=pm.math.invlogit(response_est), observed=all_subgroups_df['total_approve'])
  partial_pooled_trace = pm.sample(2000)

predicted_responses = []

for a_, a_r, a_f in zip(partial_pooled_trace['a'], partial_pooled_trace['a_region'], partial_pooled_trace['a_freq']):
  predicted_responses.append(expit(a_ + a_r[region_idx] + a_f[freq_idx]))
  
predicted_responses = np.array(predicted_responses)

poststratified_outcomes = np.array([np.dot(r, all_subgroups_df['pop_weight']) for r in predicted_responses])

all_subgroups_df['mean_partial_pooled'] = np.mean(predicted_responses, axis=0)
all_subgroups_df['low_partial_pooled'] = np.quantile(predicted_responses, .025, axis=0)
all_subgroups_df['high_partial_pooled'] = np.quantile(predicted_responses, .975, axis=0)
```

```python
plt.scatter(all_subgroups_df['total_approve'] / all_subgroups_df['total_responders'], all_subgroups_df['mean_partial_pooled'])
plt.vlines(all_subgroups_df['total_approve'] / all_subgroups_df['total_responders'], all_subgroups_df['low_partial_pooled'], all_subgroups_df['high_partial_pooled'])
plt.plot([0, 1],[0,1], linestyle='dotted')
plt.show()

```

# Appendix: Non-representativity hypothesis testing

If we wanted to, we could test the hypothesis that all the subgroups simultaneously have the correct proportion. However, we're aready pretty sure of that; instead we'll look at which specific subgroups appear to have been over- or undersampled. For each subgroup, we can test the hypothesis "is the probability of being sampled into a subgroup the same as the population probability"? using an [exact test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom_test.html).

```python
n = all_subgroups_df['total_responders'].sum()

p_values = [binom_test(r, n, p) for r, p in all_subgroups_df[['total_responders', 'pop_weight']].values]

print(p_values)
```

For each subgroup, we can graphically compare their expected inclusion likelihood to the observed one

```python
binom_cis = [proportion_confint(r, n, method='beta') for r, p in all_subgroups_df[['total_responders', 'pop_weight']].values]
colors = ['red' if p <= (.05 / len(all_subgroups_df)) else 'grey' for p in p_values]
low, high = zip(*binom_cis)
plt.vlines(all_subgroups_df['pop_weight'], low, high, color=colors)
plt.plot([min(low), max(high)], [min(low), max(high)], color='grey', linestyle='dotted')
plt.xlabel('Expected proportion')
plt.ylabel('Sample proportion')
plt.title('Comparing expected subgroup sizes with actual')
```

Of course the null hypothesis is never true but still

And NHST is exactly the wrong framework, since we can't accept the null

# Appendix: Imports and data generation

```python
import pandas as pd
import numpy as np
import pymc3 as pm
from scipy.special import expit
from statsmodels.stats.proportion import proportion_confint
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels import api as sm
from statsmodels.api import formula as smf
from scipy.stats import binom_test

region_df = pd.DataFrame({'name': ['A', 'B', 'C', 'D', 'E'], 
                                  'pop_weight': [0.4, 0.3, 0.2, 0.05, 0.05], 
                                  'sample_weight': [0.05, 0.4, 0.3, 0.2, 0.05],
                                  'approve_rate': [.3, .5, .6, .3, .5],
                                  'key': 0})
frequency_df = pd.DataFrame({'name': [1, 2, 3, 4, 5], 
                                     'pop_weight': [.15, .2, .3, .25, .1], 
                                     'sample_weight': [.1, .15, .2, .25, .3],
                                     'approve_rate': [.9, .8, .5, .3, .1],
                                     'key': 0})

all_subgroups_df = pd.merge(region_df, frequency_df, on='key', suffixes=('_region', '_frequency'))
all_subgroups_df['pop_weight'] = (all_subgroups_df['pop_weight_region'] * all_subgroups_df['pop_weight_frequency'])
all_subgroups_df['sample_weight'] = (all_subgroups_df['sample_weight_region'] * all_subgroups_df['sample_weight_frequency'])
all_subgroups_df['approve_rate'] = 0.5*(all_subgroups_df['approve_rate_region'] + all_subgroups_df['approve_rate_frequency'])

rng = np.random.default_rng(184972)

all_subgroups_df['total_responders'] = rng.multinomial(1000, all_subgroups_df['sample_weight'])
all_subgroups_df['total_approve'] = rng.binomial(all_subgroups_df['total_responders'], all_subgroups_df['approve_rate'])

all_subgroups_df.drop(['key', 'pop_weight_region', 'pop_weight_frequency', 
                              'sample_weight_region', 'sample_weight_frequency', 
                              'approve_rate_region', 'approve_rate_frequency',
                              'sample_weight', 'approve_rate'], inplace=True, axis=1)
```


