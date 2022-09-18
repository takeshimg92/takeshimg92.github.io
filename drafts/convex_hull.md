```python
import numpy as np
from scipy.spatial import ConvexHull

hull = ConvexHull(np.random.normal(0, 1, [100, 10]))

def point_in_hull(point, hull, tolerance=1e-12):
    # Thanks to https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)

point_in_hull([0]*10, hull)
```
