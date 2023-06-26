import numpy as np
import Data.data_provider as dp
from sklearn.model_selection import StratifiedKFold
import Experiments.core_calib as cal

data_name = "parkinsons"
X, y = dp.load_data(data_name)
X, y, tp = dp.make_classification_gaussian_with_true_prob(1000, 2, seed=0)
data = cal.CV_split_train_calib_test(data_name, X,y,10,0,tp)

print("data", len(data))


# _, c = np.unique(y, return_counts=True)
# print("all ", c[0]/ len(y))
# print("---------------------------------")

# skf = StratifiedKFold(n_splits=3)

# s = skf.split(X, y)
# next(s)

# print("---------------------------------")
# print("---------------------------------")

# for i, (train_calib_index, test_index) in enumerate(s):
#     X_train_calib, X_test = X[train_calib_index], X[test_index]
#     y_train_calib, y_test = y[train_calib_index], y[test_index]

#     skf2 = StratifiedKFold(n_splits=3)
#     train_index, calib_index = next(skf2.split(X_train_calib, y_train_calib))
#     X_train, X_calib = X[train_index], X[calib_index]
#     y_train, y_calib = y[train_index], y[calib_index]
#     print("i", i)



