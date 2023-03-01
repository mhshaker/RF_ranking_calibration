import Data.data_provider as dp
import numpy as np
import matplotlib.pyplot as plt

a = np.array([0, 1, 3, 1, 0])
a = np.array(a, dtype=float)

classes = [10, 20]
CL = [0, 1]

for c in CL:
    np.place(a, a==c, classes[c])

print(a)