# Time Series data is full of cyclic patterns

Airline data

```
curl https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv --output airline.csv
```

```python
import pandas as pd
import numpy as np
from scipy.signal import periodogram
from statsmodels.api import formula as smf
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('airline.csv')
df['t'] = np.arange(len(df)) # The values are already sorted

plt.plot(df['t'], df['Passengers'])
plt.title('Airline passengers by month')
plt.ylabel('Total passengers')
plt.xlabel('Month')
plt.axvline(48, linestyle='dotted')
plt.axvline(48*2, linestyle='dotted')
plt.show()

df['detrended_values'] = smf.ols('np.log(Passengers) ~ t', df).fit().resid


plt.plot(df['t'], df['detrended_values'])
plt.title('Airline passengers by month, with linear detrending')
plt.ylabel('Total passengers')
plt.xlabel('Residual')
plt.axvline(48, linestyle='dotted')
plt.axvline(48*2, linestyle='dotted')
plt.show()

nfft = int(len(df['detrended_values'])*2) # Multiplier on length for nfft is a hyperparameter
periodogram_results = pd.DataFrame(np.array(periodogram(df['detrended_values'], detrend=False, nfft=nfft)).T, columns=['frequency', 'power'])
periodogram_results.sort_values('power', ascending=False, inplace=True)

plt.stem(periodogram_results['frequency'], periodogram_results['power'])
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.show()

k = 10 # So is the number of frequencies to include
top_k_frequencies = periodogram_results['frequency'].iloc[:k]

#term_formula = 'np.sin(t / {0})'
term_formula = 'np.sin(t / {0}) + np.cos(t / {0})'
#term_formula = 'np.sin(t / {0}) + np.sin((t-np.pi/2) / {0}) + np.sin((t-np.pi) / {0}) + np.sin((t-3*np.pi/2) / {0})'
model_spec = 'np.log(Passengers) ~' + ' + '.join(['t'] + [term_formula.format(i) for i in top_k_frequencies])


sinusoid_model = smf.ols(model_spec, df)
sinusoid_fit = sinusoid_model.fit()

plt.plot(sinusoid_fit.predict(df))
plt.plot(np.log(df['Passengers']))
plt.show()
```

# Modeling cyclic behavior with sinusoidal models: Trend + Seasonality + Noise

$y_t \sim \alpha + \beta t + \sum\limits_i \gamma_i sin(\lambda_i t) + \eta_i cos(\lambda_i t)$

# Removing the trend

Get residuals of lin-log a + bt model, or detrend by differencing

# What frequencies should I include? The Periodogram

Extract top k frequencies from periodogram; fit a model with 2k + 2 parameters

# Visualizing the components of our model

plot trend, cyclic part, residual part

# Building a forecast for airline usage

Our previous model had 12+21 (month+year) parameters, and no clear forecasting method

Check cross validation, and in-sample residuals vs time

# Summary: Cooking up a sinusoid model

- Detrend
- Extract top frequencies
- Fit sine model with 2k+2 parameters

# Appendix: Okay, what the _hell_ is a Fourier Transform

3blue1brown

chatfield 2004, ch. 7

# Appendix: The road not taken - detrending by differencing

It's mostly the same: Log, difference, then do the sine thing

```
detrended_values = np.diff(np.log(df['Passengers']))

plt.plot(df['t'].iloc[1:], detrended_values)
plt.title('First order difference of log passenger count')
plt.ylabel('Total passengers')
plt.xlabel('Residual')
plt.axvline(48, linestyle='dotted')
plt.axvline(48*2, linestyle='dotted')
plt.show()
```
