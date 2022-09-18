# How did terrorism affect the GDP of the Basque region

https://rdrr.io/cran/Synth/man/basque.html
https://economics.mit.edu/files/11870



# Synthetic control lets us create a counterfactual time series from a set of parallel control time series

https://en.wikipedia.org/wiki/Synthetic_control_method

https://mixtape.scunning.com/synthetic-control.html#synthetic-control

https://economics.mit.edu/files/11859 <-- Another of the main papers, explanation of convex combination

```
from scipy.optimize import LinearConstraint, minimize
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/matheusfacure/python-causality-handbook/master/causal-inference-for-the-brave-and-true/data/smoking.csv')

piv = df[['year', 'state', 'cigsale']].pivot(index='year', columns='state')['cigsale']

i = 30

X, y = piv.drop(i, axis=1), piv[i]

from functools import partial

def loss_w(W, X, y):
    return np.sqrt(np.mean((y - X.dot(W))**2))

from scipy.optimize import fmin_slsqp

def get_w(X, y):
    
    w_start = [1/X.shape[1]]*X.shape[1]

    weights = fmin_slsqp(partial(loss_w, X=X, y=y),
                         np.array(w_start),
                         f_eqcons=lambda x: np.sum(x) - 1,
                         bounds=[(0.0, 1.0)]*len(w_start),
                         disp=False)
    return weights
    
w_fit = get_w(X, y)

plt.plot(y.index, y)
plt.plot(y.index, np.dot(X, w_fit))
plt.show()

```

# P-value method

https://matheusfacure.github.io/python-causality-handbook/15-Synthetic-Control.html

# Posterior of the synthetic control curve with the laplace approximation

# Alternatives to synthetic control

Consider also the case of just two series, one intervened on and one not: https://bcallaway11.github.io/did/articles/multi-period-did.html
ie diff-in-diff with multiple periods

Interrupted time series

# Scratch

Alternative implementation using constraints, from CIftBaT

```
from scipy.optimize import LinearConstraint, minimize
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/matheusfacure/python-causality-handbook/master/causal-inference-for-the-brave-and-true/data/smoking.csv')

piv = df[['year', 'state', 'cigsale']].pivot(index='year', columns='state')['cigsale']

i = 30

X, y = piv.drop(i, axis=1), piv[i]

from functools import partial

def loss_w(W, X, y):
    return np.sqrt(np.mean((y - X.dot(W))**2))

from scipy.optimize import fmin_slsqp

def get_w(X, y):
    
    w_start = [1/X.shape[1]]*X.shape[1]

    weights = fmin_slsqp(partial(loss_w, X=X, y=y),
                         np.array(w_start),
                         f_eqcons=lambda x: np.sum(x) - 1,
                         bounds=[(0.0, 1.0)]*len(w_start),
                         disp=False)
    return weights
    
w_fit = get_w(X, y)

plt.plot(y.index, y)
plt.plot(y.index, np.dot(X, w_fit))
plt.show()
```


Do a block bootstrap to get SEs for each year during the test period

Blocks: https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy/6811241#6811241

Relaxing the convex hull restriction: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3192710
