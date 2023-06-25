import numpy as np

# Example arrays
arr1 = np.array([[1, 2], [3, 8]])
arr2 = np.array([[2, 4], [6, 8]])

all = np.concatenate((arr1, arr2))
print("all", all)