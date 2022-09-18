https://training.cochrane.org/handbook/current/chapter-10#section-10-3

https://en.wikipedia.org/wiki/Inverse-variance_weighting

```python
from scipy.stats import sem
import numpy as np

l1 = np.random.normal(1, 1, 100) 
l2 = np.random.normal(1, 1, 100)
l12 = np.concatenate((l1, l2))

def combine_estimates(estimates, standard_errors):
  vars = np.array(standard_errors)**2
  estimates = np.array(estimates)
  combined_estimate = np.sum(estimates / vars) / np.sum(1. / vars)
  combined_var = 1./ np.sum(1. / vars)
  combined_se = np.sqrt(combined_var)
  return combined_estimate, combined_se

print(combine_estimates([np.mean(l1), np.mean(l2)],
                        [sem(l1), sem(l2)]))
print(np.mean(l12), sem(l12))
```

This is really a fixed-effects model; consider also a mixed-effects or hierarchical bayesian approach

Compare with a bayesian non-hierarchical approach (same result if the parameter is shared...I think) and wtih a hierarchical analysis
