import numpy as np

# Example arrays
arr1 = np.array([1, 2, 3, 8, 5])
arr2 = np.array([2, 4, 6, 8, 5])

# Find indices of elements with the same value
indices = np.where(arr1 == arr2)

print(indices[0])
