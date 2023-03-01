import numpy as np
import Data.data_provider as dp
import matplotlib.pyplot as plt


# samples = 10000
# features = 20


# X, y, tp = dp.make_classification_gaussian_with_true_prob(samples, features)

# plt.hist(tp, bins=100)
# plt.show()

# print("tp", tp)
# print("y", y)

a = np.asarray([1, 2, 3])
b = np.asarray([4, 5, 6])

mean = (a + b) / 2

print(mean)