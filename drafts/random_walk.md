http://e.sci.osaka-cu.ac.jp/yoshino/download/rw/

[https://github.com/jbrownlee/Datasets/blob/master/monthly-robberies.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-robberies.csv)

```python

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-robberies.csv') # IID I think
y = df['Robberies']

#df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-mean-temp.csv') # IID
#y = df['Temperature']

#df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly_champagne_sales.csv') # IID
#y = df['Sales']

#df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv') # Looks more random walk than IID
#y = df['Passengers']

t = list(df.index)

#diff_y = y - y.shift(12)
#diff_y = y / y.shift(12) - 1
diff_y = np.log(y) - np.log(y.shift(12))

def gen_iid_noise(size, sigma): # Note that this is Gaussian IID noise, not white noise
  return np.random.normal(0, sigma, size=size)

def gen_random_walk(size, sigma):
  return np.cumsum(np.random.normal(0, sigma, size=size))
  
for i in range(100):
  plt.plot(t[12:], gen_random_walk(len(t)-12, np.std(diff_y)), color='orange') # Analytical version: Expanding bands of something involving sigma and the square root of t
  
plt.plot(t, diff_y, color='blue')
plt.show()

for i in range(100):
  plt.plot(t[12:], gen_iid_noise(len(t)-12, np.std(diff_y)), color='orange') # Analytical version: constant bands of 2*sigma or so
  
plt.plot(t, diff_y, color='blue')
plt.show()

# What separates these two models? One is the cumulative sum of the other
# Estimates of sigma, drift, predictions at each point - implement both in python/pandas with shift and window functions
# Gelman style measures of similarity of observed data to simulated data: Quantiles of observed vs simulated for summary stats, observation at each time point
# Model comparison? Select model based on leave-one-out sequential CV/avg error
# Include a measure of surprise, like a Z-score (distance from mean in standard deviations) or Normal quantile

low = -2*np.std(diff_y)
high = 2*np.std(diff_y)
plt.axhline(low)
plt.axhline(high)
plt.plot(t, diff_y)
plt.show()

diff_y_second_diff = diff_y - diff_y.shift(1)
sd = diff_y_second_diff.rolling(24).std()
low = diff_y.shift(1) - 2*sd
high = diff_y.shift(1) + 2*sd
plt.plot(t, low)
plt.plot(t, diff_y)
plt.plot(t, high)
plt.show()

diff_y_second_diff = diff_y - diff_y.shift(1)
sd = diff_y_second_diff.rolling(24).std()
drift = diff_y_second_diff.rolling(24).mean()
low = diff_y.shift(1) + drift - 2*sd
high = diff_y.shift(1) + drift + 2*sd
plt.plot(t, low)
plt.plot(t, diff_y)
plt.plot(t, high)
plt.show()

# Attempt to falsify random walk hypothesis
# How do we attempt to falsify the "normal-shaped noise" assumption? Is there a non-parametric version with chebyshev's inequality
# Probably but I bet its too wide
sns.distplot(diff_y_second_diff.dropna(), fit=norm)
plt.show()
```
