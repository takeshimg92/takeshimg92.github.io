Not a free lunch, but a very cheap one: You should include pre-treatment covariates in your A/B test analysis

We'll show it with some simulations of experiments with binary treatments and continuous normally distributed outcomes:
1. No extra variable, y ~ t
2. One extra variable, y ~ t vs y ~ t + x
3. One unobserved extra variable but an observed ancestor, y ~ t vs y ~ t + x and y ~ t vs y ~ t + z (where z is the parent of x)
4. One irrelevant extra variable
5. One incorrectly-controlled-for extra variable (DO NOT CONDITION ON POST-TREATMENT VARIABLES, conditioning on a variable which is a result of x)
n = 100, alpha = 1, delta=0.5

Consider non-normal outcomes? Heteroskedasticity?

Maybe this is just one dataset and we just get better at seeing it

https://pubmed.ncbi.nlm.nih.gov/26921693/

http://www.stat.cmu.edu/~cshalizi/TALR/TALR.pdf - 7.1.5

https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf
https://booking.ai/how-booking-com-increases-the-power-of-online-experiments-with-cuped-995d186fff1d

# Including pretreatment measurements gives us more efficient estimates of treatment effects in A/B tests

## Example

## Simulations

```python
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.api import formula as smf
from tqdm import tqdm

sample_size = 100
t_p = 0.5
a_g = 0.0
a_s = 1.0
a_x = 2
b_x = 1
s_x = 1
a_y_p = -1
b_y_p = 0.5
s_y_p = 2
a_y = 3
b_y = 1
s_y = 1
a_c = 0
b_c = 1
s_c = 1
d = 1

def generate_data():
  t = np.random.binomial(1, t_p, sample_size)
  g = np.random.normal(a_g, a_s, sample_size)
  x = np.random.normal(a_x + b_x * g, s_x)
  y_p = np.random.normal(a_y_p + b_y_p * x**5, s_y_p)
  y = np.random.normal(a_y + d * t + b_y * x ** 3, s_y)
  c = np.random.normal(a_c + b_c * y, s_c)
  i = np.random.normal(0, 1, sample_size)
  return pd.DataFrame({'t': t, 'g': g, 'x': x, 'y_p': y_p, 'y': y, 'c': c, 'i': i})

sim_scenarios = ['Baseline', 'Known cause', 'Previous observation', 'Cause of cause', 'Irrelevant', 'Kitchen Sink', 'Bad control']
sim_results = []

for _ in tqdm(range(1000)):
  sim_df = generate_data()
  baseline = smf.ols('y ~ t', sim_df).fit().params['t']
  known_cause = smf.ols('y ~ t + x', sim_df).fit().params['t']
  prev = smf.ols('y ~ t + y_p', sim_df).fit().params['t']
  cause_of_cause  = smf.ols('y ~ t + g', sim_df).fit().params['t']
  kitchen_sink  = smf.ols('y ~ t + x + y_p + g + i', sim_df).fit().params['t']
  bad_control = smf.ols('y ~ t + c', sim_df).fit().params['t']
  irrelevant = smf.ols('y ~ t + i', sim_df).fit().params['t']
  sim_results.append([baseline, known_cause, prev, cause_of_cause, irrelevant, kitchen_sink, bad_control])

sim_df = pd.DataFrame(sim_results, columns=sim_scenarios)
for col in sim_df.columns:
  sns.distplot(sim_df[col], label=col, hist=False)
plt.axvline(d, label='True treatment effect')
plt.xlabel('Treatment effect')
plt.legend()
plt.show()
```

## Why this works - smaller residual size vs larger parameter count

# What happens if we pick a bad pretreatment measurement?

Bad: it is uncorrelated --> No benefit, but no harm

Bad: it is actually

# : Even a really good pretreatment measurement doesn't solve the problem of a missing confounder

```python
import numpy as np
from statsmodels.api import formula as smf
import pandas as pd
from scipy.special import expit
from matplotlib import pyplot as plt
import seaborn as sns

n = 1000

s_c = 1
s_p = 1
s_y = 1

a_p = 0
b_cp = 1
a_y = 0
b_ty = 1
b_cy = 1

c = np.random.normal(0, s_c, n)
t = np.random.binomial(1, expit(c))
p = np.random.normal(a_p + b_cp*c, s_p)
y = np.random.normal(a_y + b_ty*t + b_cy*c, s_y)

df = pd.DataFrame({'c': c, 't': t, 'p': p, 'y': y})

print(smf.ols('y ~ t', df).fit().summary())
print(smf.ols('y ~ t + c', df).fit().summary())
print(smf.ols('y ~ t + p', df).fit().summary())

plt.scatter(df[df['t'] == 1]['c'], df[df['t'] == 1]['y'])
plt.scatter(df[df['t'] == 0]['c'], df[df['t'] == 0]['y'])
plt.show()

plt.scatter(df[df['t'] == 1]['p'], df[df['t'] == 1]['y'])
plt.scatter(df[df['t'] == 0]['p'], df[df['t'] == 0]['y'])
plt.show()
```

Can we say that it's an improvement over the naive analysis? How much?
