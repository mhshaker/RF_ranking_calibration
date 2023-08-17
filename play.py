import numpy as np

# Create a sample numpy array with 0s and 1s
original_array = np.array([0, 1, 1, 0, 1, 0])

# Switch 0s to 1s and 1s to 0s using arithmetic operations
switched_array = 1 - original_array

print("Original Array:", original_array)
print("Switched Array:", switched_array)
