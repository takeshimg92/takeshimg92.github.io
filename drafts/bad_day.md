---
layout: post
title: "How active are my most active users? My least active ones? Quantiles and their confidence intervals in Python"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: bad_day.png
---

*Analysts spend a lot of time thinking about the mean, which is by far the most common descriptive statistic we extract. But looking at the average observation leaves out a lot of information about the shape of the distribution . Comparing the shapes of two distributions . Example*

# An example: Characterizing the distribution of email opens

Number of newsletter opens - 

If you run an email newsletter or campaign, you've probably spent time thinking about how many people actually saw your campaign - that's the point of having one, after all. Let's imagine that you are the supernaturally gifted CEO of the greatest tech company in history, Zombo.com. That makes you the proprietor of the much-beloved [Zombo.com newZletter](https://www.zombo.com/join1.htm) (now in its 21st year!).

Example of very different distributions with the same mean 

```python
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import poisson, skellam, nbinom, randint, geom

n = 10000

for d in [poisson(100), skellam(1101, 1000), randint(0, 200), geom(1./100)]:
  samples = d.rvs(n)
  sns.kdeplot(samples)
plt.xlim(0)  
plt.xlabel('Count')
plt.ylabel('Density')
plt.title('Four distributions with a mean of 100')
plt.show()
```

Similar examples: Anomaly detection (if the process is serially uncorrelated)



# Quantile idea and point estimate

Histogram

Pandas describe

IQR, Box and whisker

# Quantile inference uncertainty

MJ standard errors

CDF/Quantile curve

# Two-sample quantile comparison

Two sample histograms

Quantile curve with uncertainty bands, significant changes marked\

# Outro: Quantile regression

Mean is to linear regression as quantiles are to quantile regression

# Appendix: Simulated coverage of the quantile's SEs

The method works in a simulation similar to one that might have generated our data; note where coverage is good and where it is not

# Appendix: Alternative ways of getting the standard errors

Sampling distribution with known pdf, Exact method, bootstrap
