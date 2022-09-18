---
layout: post
title: "?"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: supervised_unsupervised.png
---

https://gist.github.com/lmc2179/ae47161cc4125db228ca79f4e6859d5b

ESL section

Examples

- Anomaly detection
- Basket recommendation
- ~K~DE on integer discrete data
- Density for likelihood-ratio weighting

```python
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

x = np.random.normal(0, 1, 5000)
x_sim = np.random.uniform(np.min(x), np.max(x), 5000)

X = np.concatenate((x, x_sim)).reshape(-1, 1)
y = [1]*5000 + [0]*5000

m = MLPClassifier(hidden_layer_sizes=(100,))
m.fit(X, y)

uniform_density = 1 / (np.max(x) - np.min(x))

x_plot = np.linspace(np.min(x), np.max(x), 10000)
classifier_output = m.predict_proba(x_plot.reshape(-1, 1))[:,1]
y_plot = uniform_density * (classifier_output)/(1.-classifier_output)

plt.plot(x_plot, y_plot)
sns.distplot(x, bins=np.linspace(np.min(x), np.max(x), 10), kde=False, norm_hist=True)
plt.show()

print('Area under density curve is {0}'.format( np.trapz(y_plot, x_plot)))
```
