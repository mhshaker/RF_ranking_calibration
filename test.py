import numpy as np
import Data.data_provider as dp
import matplotlib.pyplot as plt


samples = 10000
features = 20


X, y, tp = dp.make_classification_gaussian_with_true_prob(samples, features)

plt.hist(tp, bins=100)
plt.show()

# print("tp", tp)
# print("y", y)

