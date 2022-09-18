Should I include these outliers in my analysis? Probably

- when the outliers contain little information about your outcome of interest

NOT

- when the outliers are annoying to analyze and you don't want to deal with them

An outlier analysis rule: Discarding outliers or choosing to winsorize does _not_ come from looking at the data, but from the substantive goals of the analysis

When you care about the body more than the tail! 

Example: We want to "raise the floor" of user activity instead of getting more juice out of the super-users. This may be a good choice! but it is a different goal from "get more user activity, however you do it"

Is the winsorized mean "robust"? In a sense - it will ignore the tail observations, so an outlying point will move the winsorized mean less. Be careful that this is actually what you want to measure, and that you're not just using it because you don't want to deal with your funny-looking data

It's a different quantity than the mean - it's not just "the mean but safer because robust"

If you think the outliers are measurement errors, why keep them in the data set
If you think the outliers are irrelevant to your analysis - why didn't you think of that before? Are you sure you're not justifying it because you think it will make your analysis easier?

In many cases, the tail observations are the _most_ important. They are your highest activity users, highest activity clients, etc - think carefully before you exclude them!

Winsorizing decreases variance and increases power

Note that we have to bootstrap - the whole algorithm is winsor + mean

```python
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import sem, describe
from scipy.stats.mstats import winsorize
import numpy as np
import pandas as pd

data = pd.Series(np.abs(np.random.standard_t(1, 1000)))
sns.distplot(data)
plt.show()

print(data.describe())

m = np.mean(data)
m_se = sem(data)
m_se_bootstrap = np.std([np.mean(np.random.choice(data, len(data))) for _ in range(10000)])

winsor_mean_99 = np.mean(winsorize(data, limits=(0, .01)))
winsor_se_direct = sem(winsorize(data, limits=(0, .01)))
winsor_se_bootstrap = np.std([np.mean(winsorize(np.random.choice(data, len(data)), limits=(0, .01))) for _ in range(10000)])

print(m, m_se, m_se_bootstrap)
print(winsor_mean_99, winsor_se_direct, winsor_se_bootstrap)
```
