
Simulation:
* Generate data (x properties)

Version 1:
* Assign pairs with LSA

Version 2:
* Assign pairs at random

Then:
* Generate y values
* Compute treatment effects both ways
* Demonstrate treatment effect heterogeneity (both ways?)

```python
from scipy.stats import norm
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

x_gen = norm(0, 2)

def f(x, T):
  return 10 + 5*np.sin(x) + T * .2 * x + norm(0, 1).rvs(2*N)
  
def dist_matrix(unit_x):
  dist_vector = pdist(np.reshape(unit_x, (len(unit_x), 1)))
  m = squareform(dist_vector)
  np.fill_diagonal(m, np.inf)
  return m
  
N = 100
population_x = x_gen.rvs(2 * N)
row_ind, col_ind = linear_sum_assignment(dist_matrix(population_x)) # Each item is in row_ind, with its pair in col_ind

pairs = {}
for i row_ind:
  if i not in pairs:
    pairs[i] = col_ind[i]

#plt.scatter(population_x, observed_y)
#plt.show()
```
