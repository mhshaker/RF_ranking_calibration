import Data.data_provider as dp
import numpy as np
import matplotlib.pyplot as plt

# samples = 10000
# features = 4

# X, y, tp = dp.make_classification_gaussian_with_true_prob(samples, features,0,1,1,2,0,1,1,2, 0)
# plt.hist(tp, bins=100)
# plt.show()

array = np.array([4,2,7,1,80])
order = array.argsort()
ranks = order.argsort()

X, y, tp = dp.make_classification_gaussian_with_true_prob(samples, features)

plt.hist(tp, bins=100)
plt.show()

# print("tp", tp)
# print("y", y)