import numpy as np

# Create a sample NumPy array
arr = np.array([1, 2, 3, 4, 5])

# Shift the elements to the right by 2 positions
shifted_right = np.concatenate((arr[-2:], arr[:-2]))

# Shift the elements to the left by 2 positions
shifted_left = np.concatenate((arr[2:], arr[:2]))

print("Original array:", arr)
print("Shifted right by 2:", shifted_right)
print("Shifted left by 2:", shifted_left)
