import numpy as np
from CalibrationM import convert_prob_2D

x = np.array([0.9, 0.2, 0.4])
print("x shape", x.shape)
x2 = convert_prob_2D(x)

print("x2", x2)
