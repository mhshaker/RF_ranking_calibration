import numpy as np
import Data.data_provider as dp


X, y, tp = dp.make_classification_with_true_prob(10, 4)

print("tp", tp)
print("y", y)