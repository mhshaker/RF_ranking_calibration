import Data.data_provider as dp
import numpy as np
import matplotlib.pyplot as plt

a = np.array([1, 2, 3, 1, 2])

np.place(a, a==1, 10)
np.place(a, a==2, 20)

print(a)