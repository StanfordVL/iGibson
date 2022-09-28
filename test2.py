from scipy.interpolate import CubicSpline
import numpy as np

ab = np.concatenate(np.array([[[1,2], [3,4]], [[1, 2], [3, 4]]]), axis=0)
ab = ab[np.argsort(ab[:,0])]

x = ab[:,0]
y = ab[:,1]

print(np.concatenate((x, y), axis=1))
