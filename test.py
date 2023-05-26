
# import Data.data_provider as dp
# from Experiments import core as cal
# from estimators.LR_estimator import LR_u as lr
# from sklearn.linear_model import LogisticRegression

# data_name = "spambase"
# X, y = dp.load_data(data_name, ".")
# data = cal.split_train_calib_test(data_name, X, y, 0.3, 0.3, 0)


# m = lr(random_state=0).fit(data["x_train"], data["y_train"])
# print("intercept before", m.intercept_[0])
# acc = m.score(data["x_test"], data["y_test"])
# print("acc", acc)
# m = m.update_intercept(data["x_calib"], data["y_calib"])
# print("intercept after ", m.intercept_[0])
# acc = m.score(data["x_test"], data["y_test"])
# print("acc", acc)

import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt

from skopt import BayesSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_digits(n_class=10, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)

# log-uniform: understand as search over p = exp(x) by varying x
opt = BayesSearchCV(
    RandomForestClassifier(),
    {
        'n_estimators': (1, 200),
        'max_depth': (1, 8),
    },
    n_iter=32,
    cv=3
)

opt.fit(X_train, y_train)

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))
