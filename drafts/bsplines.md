Bsplines rock and should be your go-to method for modeling smooth nonlinearities

How they work - Basis expansion and splines; Degree of spline, number of knots; start with low order and increase

Example: Non-linear relationship of NOX and Price in the Boston housing data

How to use them in Python

Graphing them with a PDP (?); plotting them somehow (what does this do if it's nonlinear: https://www.statsmodels.org/stable/generated/statsmodels.graphics.regressionplots.plot_partregress.html#statsmodels.graphics.regressionplots.plot_partregress)

Looking at CIs of the PDP

```python
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.api import formula as smf
import pandas as pd
from statsmodels.graphics.regressionplots import plot_partregress

n = 100
x = np.linspace(0, 7, n)
y = np.sin(x) + x + np.random.normal(0, 1, size=n)
data = pd.DataFrame({'x': x, 'y': y})

spline_df = 10

for spline_degree in np.arange(0, 4):
  model_spec = 'y ~ x + bs(x, df={0}, degree={1})'.format(spline_df, spline_degree)
  model = smf.ols(model_spec, data)
  result = model.fit()
  y_hat = result.fittedvalues
  
  pred_results = result.get_prediction(data)
  y_hat_se = pred_results.se_mean
  y_hat_low = y_hat - 2 * y_hat_se
  y_hat_high = y_hat + 2 * y_hat_se
  
  y_obs_se = pred_results.se_obs
  y_hat_obs_low = y_hat - 2 * y_obs_se
  y_hat_obs_high = y_hat + 2 * y_obs_se
  
  plt.scatter(x, y)
  plt.plot(x, np.sin(x) + x)
  plt.plot(x, y_hat)
  plt.fill_between(x, y_hat_low, y_hat_high, color='grey', alpha=.5)
  plt.plot(x, y_hat_obs_high)
  plt.plot(x, y_hat_obs_low)
  plt.title(model_spec)
  plt.show()
  ```
