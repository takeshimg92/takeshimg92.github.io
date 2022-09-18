# Forest plots and regression models are great Exploratory Data Analysis tools

> Consistent with this view, we believe, is a clear demand that pictures based on exploration of data should _force_ their messages upon us. Pictures that emphasize what we already know - "security blankets" to reassure us - are frequently not worth the space they take. Pictures that have to be gone over with a reading glass to see the main point are wasteful of time and adequate of effect. The **greatest value of a picture** is when it _forces_ us to notice **what we never expected to see**.

~John Tukey, _Exploratory Data Analysis_ (emphasis in original)

> ...there are rarely any widely accepted, nearly right models that can be used with real data. By default, the true enterprise is description. Most everything else is puffery.

Richard Berk, [_What You Can and Can’t Properly Do with Regression_](http://www.public.asu.edu/~gasweete/crj604/readings/2010-Berk%20(what%20you%20can%20and%20can't%20do%20with%20regression).pdf)

> Think of a series of models, starting with the too-simple and continuing through to the hopelessly messy. Generally it’s a good idea to start simple. Or start complex if you’d like, but prepare to quickly drop things out and move to the simpler model to help understand what’s going on. Working with simple models is not a research goal—in the problems we work on, we usually find complicated models more believable—but rather a technique to help understand the fitting process

Andrew Gelman and Jennifer Hill, _Data Analysis Using Regression and Multilevel/Hierarchical Models_

# EDA has no clear-cut recipe

Most analysis guides act as if you have the model in mind before you run the analysis, but really there's a feedback loop between exploring the data and fitting models, which we call "exploration"

- Organizations need to know how things are going, and how to improve them
- Often, these do not start as well-formed questions, but rather as vague "what is going on" questions
- What groups appear to be driving the outcome of interest?
- What does this suggest about the causal relationships at work?

# Why do EDA? (What are examples of EDA insights)

- Build intuition about potential intervention points (points of leverage)
- Check assumptions we have about which variables are associated with the outcome
- Which customers are having positive/negative experiences?
- Which groups are providing the best monetization?
- Who is most likely to convert?

- Challenges: many variables, maybe many subgroups, hard to intuit the "unique impact" of each


# Regression is a great descriptive tool
- Berk: Describing the data is most of what we do
- Coefficients represent partial correlations
- Dummy encoding shows us extreme subgroups

# Isn't this a fishing expedition?
- Isn't everything
- From the Type I (FWER/FDR perspective, we can do some stuff); Data splitting
- Or just live that Bayes life

# Examples: What is associated with high income?

```bash
#https://archive.ics.uci.edu/ml/datasets/Census+Income
curl https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data > census.csv
```

```python
import pandas as pd
from statsmodels.api import formula as smf

df = pd.read_csv('census.csv')
df.columns = 'age workclass fnlwgt education education_num marital_status occupation relationship race sex capital_gain capital_loss hours_per_week native_country high_income'.split(' ')
df['high_income'] = df['high_income'].apply(lambda x: 1 if x == ' >50K' else 0)

model = smf.logit('high_income ~ age + workclass + education + marital_status + age:workclass', df)
results = model.fit()
```

# Why isn't this a causal interpretation? When might it be?

# What else might go into an EDA?

# Appendix: Python functions for forest plots


```python
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def forestplot(middle, lower, upper, names):
  df = pd.DataFrame({'mid': middle,
                     'low': lower,
                     'high': upper,
                     'name': names})
  df['position'] = -np.arange(len(df))
  plt.scatter(df['mid'], df['position'])
  plt.scatter(df['low'], df['position'], marker='|', color='black')
  plt.scatter(df['high'], df['position'], marker='|', color='black')
  plt.hlines(df['position'], df['low'], df['high'])
  plt.yticks(df['position'], df['name'])
    
def forestplot_sorted( middle, lower, upper, names, colormap):
  df = pd.DataFrame({'mid': middle,
                     'low': lower,
                     'high': upper,
                     'name': names})
  df = df.sort_values('mid')
  df['position'] = -np.arange(len(df))
  colors = colormap(np.linspace(0, 1, len(df)))
  plt.scatter(df['mid'], df['position'], color=colors)
  plt.scatter(df['low'], df['position'], color=colors, marker='|')
  plt.scatter(df['high'], df['position'], color=colors, marker='|')
  plt.hlines(df['position'], df['low'], df['high'], color=colors)
  plt.yticks(df['position'], df['name'])
    
def forestplot_grouped(middle, lower, upper, names, colormap, groups):
  df = pd.DataFrame({'mid': middle,
                     'low': lower,
                     'high': upper,
                     'name': names,
                     'groups': groups})
  unique_groups = list(set(df['groups']))
  color_lookup = {g: c for g, c in zip(unique_groups, colormap(np.linspace(0, 1, len(unique_groups))))}
  colors = [color_lookup[g] for g in groups]
  df['position'] = -np.arange(len(df))
  plt.scatter(df['mid'], df['position'], color=colors)
  plt.scatter(df['low'], df['position'], color=colors, marker='|')
  plt.scatter(df['high'], df['position'], color=colors, marker='|')
  plt.hlines(df['position'], df['low'], df['high'], color=colors)
  plt.yticks(df['position'], df['name'])

forestplot([0, 1, 2, 3], [-1, 0, 1, 2], [1, 2, 3, 4], ['a', 'b', 'c', 'd'])
plt.show()

forestplot_sorted([0, 1, 2, 3], [-1, 0, 1, 2], [1, 2, 3, 4], ['a', 'b', 'c', 'd'], plt.cm.plasma)
plt.show()

forestplot_grouped([0, 1, 2, 3], [-1, 0, 1, 2], [1, 2, 3, 4], ['a', 'b', 'c', 'd'], plt.cm.plasma, [0, 0, 1, 2])
plt.show()
```


```python
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def group_covariates(terms, cols):
  groups = -np.ones(len(cols))
  g = 0
  for i, c in enumerate(cols):
    if c[:len(terms[g])] != terms[g]: # Check first part of string
      g +=1
    groups[i] = g
  return groups.astype('int')

def clean_categorical_name(n):
  i = n.index('[')
  return n[i+3:-1]
  
def is_level(group_name, col_name):
  return group_name != col_name

def forestplot(model, fit_results, alpha=.05, cols_to_include=None, bonferroni_correct=False):
  if bonferroni_correct:
    a = alpha / len(fit_results.params)
  else:
    a = alpha
  summary_matrix = pd.DataFrame({'point': fit_results.params,
                                 'low': fit_results.conf_int(a)[0],
                                 'high': fit_results.conf_int(a)[1],
                                 'name': model.data.design_info.column_names,
                                 'position': -np.arange(len(fit_results.params))})
  terms = model.data.design_info.term_names
  n_terms = len(terms)
  term_group = group_covariates(terms, summary_matrix['name'])
  summary_matrix['term'] = [terms[g] for g in term_group]
  term_colors = plt.cm.rainbow(np.linspace(0, 1, n_terms))
  summary_matrix['color'] = [term_colors[g] for g in term_group]
  summary_matrix['clean_name'] = [clean_categorical_name(c) if is_level(t, c) else c for t, c in summary_matrix[['term', 'name']].values]
  if cols_to_include is None:
    cols = set(terms)
  else:
    cols = set(cols_to_include)
  summary_matrix = summary_matrix[summary_matrix['term'].apply(lambda x: x in cols)]
  plt.scatter(summary_matrix['point'], summary_matrix['position'], c=summary_matrix['color'])
  for p, l, h, c in summary_matrix[['position', 'low', 'high', 'color']].values:
    plt.plot([l, h], [p, p], c=c)
  plt.axvline(0, linestyle='dotted', color='black')
  plt.yticks(summary_matrix['position'], summary_matrix['clean_name'])
```

```
import patsy
import pandas as pd

df = pd.DataFrame({'X': [1, 0, 1, 2, 3], 'Z': [1, 0, 1, 2, 3], 'y': 0})
endog, exog = patsy.dmatrices('y ~ X + C(X) + bs(X, df=3) + X:Z + C(Z) + C(X):C(Z)', df)

term_names = [t.name() for t in exog.design_info.terms]
term_slices = list(exog.design_info.term_slices.values())

factor_names = [evf.name() for evf in exog.design_info.factor_infos.keys()]
factor_levels = [fi.categories for fi in exog.design_info.factor_infos.values()]
factor_lookup = {n: l for n, l in zip(factor_names, factor_levels) if l is not None}
```
