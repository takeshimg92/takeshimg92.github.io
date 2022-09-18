Interrupted time series analysis

Possible Example: Did Blazing Saddles kill the western?

Key idea: In the absence of an ironclad causal inference method (experiment, clear list of counfounders, instrumental variable), we often tend to use an informal "before vs after" look to make a guess about causal effects after an intervention is introduced. We might also include our historical knowledge of previous fluctuation levels, pre-treatment trends, and cyclic behavior, and attempt to synthesize them. This article is about ITS, the formal way of doing that.

Nonlinear extensions with B-splines

ITS Analysis

Outcome = Intercept + Cycle + Long-term trend + Impact of Treatment + Post-treatment change in trend + Noise

Views: EV + CI
Data + Expectation ( + Prediction interval)
Cycle only
Trends only
Impact only

```python
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.api import formula as smf
import numpy as np

df = pd.read_csv('https://www1.nyc.gov/assets/tlc/downloads/csv/data_reports_monthly.csv')
df['trips'] = df['         Trips Per Day         '].str.replace(',', '').astype(float)
df['date'] = df['Month/Year'].apply(lambda s: datetime.strptime(s + '-01', '%Y-%m-%d'))

daily_trip_series = df.groupby('date').sum()['trips']

daily_trip_regression_data = pd.DataFrame({'date': daily_trip_series.index, 'trips': daily_trip_series}).reset_index(drop=True).sort_values('date')
#daily_trip_regression_data = daily_trip_regression_data[daily_trip_regression_data['date'] >= '2019-01-01']
#daily_trip_regression_data = daily_trip_regression_data[daily_trip_regression_data['date'] < '2022-01-01']
daily_trip_regression_data['month'] = daily_trip_regression_data['date'].apply(lambda x: x.month)
daily_trip_regression_data['year'] = daily_trip_regression_data['date'].apply(lambda x: x.year)
daily_trip_regression_data['trend'] = np.arange(len(daily_trip_regression_data))
daily_trip_regression_data['after'] = (daily_trip_regression_data['date'] >= '2020-04-01').apply(int)
daily_trip_regression_data['trend'] = np.arange(len(daily_trip_regression_data)) * (1. - daily_trip_regression_data['after']) + (daily_trip_regression_data['after'] * np.max(np.arange(len(daily_trip_regression_data)) * (1. - daily_trip_regression_data['after']))) # Forgive me...goal is to count up to the time when the intervention happens, then stay at that value. There's probably a nice way to do it with np.clip
daily_trip_regression_data['after'].mask(daily_trip_regression_data['date'] == '2020-03-01', 1./3, inplace=True)
daily_trip_regression_data['after_trend'] = np.cumsum(daily_trip_regression_data['after'])

plt.scatter(daily_trip_regression_data['date'], daily_trip_regression_data['trips'])

#model = smf.ols('trips ~ trend + after + after_trend', daily_trip_regression_data)
model = smf.ols('trips ~ bs(trend, df=5) + after + bs(after_trend, df=5) + C(month)', daily_trip_regression_data)
result = model.fit()
plt.plot(daily_trip_regression_data['date'], result.fittedvalues)

prediction_se = result.get_prediction(daily_trip_regression_data).se_mean
plt.fill_between(daily_trip_regression_data['date'], result.fittedvalues - 2 * prediction_se, result.fittedvalues + 2 * prediction_se, alpha=.5)

obs_se = result.get_prediction(daily_trip_regression_data).se_obs
plt.plot(daily_trip_regression_data['date'], result.fittedvalues + 2 * obs_se, color='black', linestyle='dotted')
plt.plot(daily_trip_regression_data['date'], result.fittedvalues - 2 * obs_se, color='black', linestyle='dotted')

plt.show()

plt.title('Monthly cycle removed')
plt.scatter(daily_trip_regression_data['date'], daily_trip_regression_data['trips'])
daily_trip_regression_data_plot = daily_trip_regression_data.copy()
daily_trip_regression_data_plot['month'] = 6
plt.plot(daily_trip_regression_data['date'], result.predict(daily_trip_regression_data_plot))
plt.show()

plt.title('Monthly cycle only')
plt.scatter(daily_trip_regression_data['date'], daily_trip_regression_data['trips'])
daily_trip_regression_data_plot = daily_trip_regression_data.copy()
daily_trip_regression_data_plot['trend'] = 0
daily_trip_regression_data_plot['after'] = 0
daily_trip_regression_data_plot['after_trend'] = 0
plt.plot(daily_trip_regression_data['date'], result.predict(daily_trip_regression_data_plot))
plt.show()

plt.title('Counterfactual')
plt.scatter(daily_trip_regression_data['date'], daily_trip_regression_data['trips'])
daily_trip_regression_data_plot = daily_trip_regression_data.copy()
daily_trip_regression_data_plot['after'] = 0
daily_trip_regression_data_plot['after_trend'] = 0
plt.plot(daily_trip_regression_data['date'], result.predict(daily_trip_regression_data_plot))
plt.show()
```
