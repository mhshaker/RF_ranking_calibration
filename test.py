import numpy as np



x = np.array([[1,2], [30,4]])

max_index = np.argmax(x, axis=1)

r = 5

x[list(range(len(x))), max_index] += r

print("x", x)
print("max_index", max_index)

x[_,_]