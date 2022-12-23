import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
from sklearn.datasets import make_regression
from scipy.stats import multivariate_normal
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split
from estimators.IR_RF_estimator import IR_RF
seed = 0

import matplotlib.pyplot as plt

np.random.seed(seed)
samples = 10000

X, tp = make_regression(samples) # make regression data
y = np.where(tp>0, 1, 0) # create classification labels by setting a threshold

test_size = 0.4
x_train_calib, x_test, y_train_calib, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)
x_train, x_calib, y_train, y_calib = train_test_split(x_train_calib, y_train_calib, test_size=0.5, shuffle=True, random_state=seed) 
_, _, tp_train_calib, tp_test = train_test_split(X, tp, test_size=test_size, shuffle=True, random_state=seed)
_, _, tp_train, tp_calib = train_test_split(x_train_calib, tp_train_calib, test_size=0.5, shuffle=True, random_state=seed) 

irrf = IR_RF(n_estimators=100, random_state=seed)
irrf.fit(x_train, y_train)

x_test_rank = irrf.rank(x_test, class_to_rank=1)
x_test_prob = irrf.predict_proba(x_test, laplace=1)[:, 1]

rank_sort_index = np.argsort(x_test_rank, kind="stable")
prob_sort_index = np.argsort(x_test_prob, kind="stable")
true_sort_index = np.argsort(tp_test, kind="stable")

tp_test_rank_sort = tp_test[rank_sort_index]
tp_test_prob_sort = tp_test[prob_sort_index]
tp_test_true_sort = tp_test[true_sort_index]

# print("tp_test_rank_sort", tp_test_rank_sort)
# print("tp_test_true_sort", tp_test_true_sort)

tau, p_value = kendalltau(tp_test_true_sort, tp_test_rank_sort)
print("tau", tau)

tau, p_value = kendalltau(tp_test_true_sort, tp_test_prob_sort)
print("tau", tau)
