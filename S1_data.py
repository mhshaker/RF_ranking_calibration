import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split
from estimators.IR_RF_estimator import IR_RF
seed = 0

np.random.seed(seed)
# Synthetic data with 5 dimentions and 2 classes
samples = 100

mean1 = [0, 2, 3, -1, 9]
cov1 = [[.1, 0, 0, 0, 0], 
        [0, .5, 0, 0, 0],
        [0, 0, 0.8, 0, 0],
        [0, 0, 0, .1, 0],
        [0, 0, 0, 0, .3],
        ]

mean2 = [-1, 3, 0, 2, 3]
cov2 = [[.9, 0, 0, 0, 0], 
        [0, .1, 0, 0, 0],
        [0, 0, 0.3, 0, 0],
        [0, 0, 0, .1, 0],
        [0, 0, 0, 0, .7],
        ]

x1 = np.random.multivariate_normal(mean1, cov1, samples)
x2 = np.random.multivariate_normal(mean2, cov2, samples)

x1_pdf_dif = multivariate_normal.pdf(x1, mean1, cov1) - multivariate_normal.pdf(x1, mean2, cov2)
x2_pdf_dif = multivariate_normal.pdf(x2, mean2, cov2) - multivariate_normal.pdf(x2, mean1, cov1)

X = np.concatenate([x1, x2])
y = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])
tp = np.concatenate([x1_pdf_dif, x2_pdf_dif])


test_size = 0.4
x_train_calib, x_test, y_train_calib, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)
x_train, x_calib, y_train, y_calib = train_test_split(x_train_calib, y_train_calib, test_size=0.5, shuffle=True, random_state=seed) 
_, _, tp_train_calib, tp_test = train_test_split(X, tp, test_size=test_size, shuffle=True, random_state=seed)
_, _, tp_train, tp_calib = train_test_split(x_train_calib, tp_train_calib, test_size=0.5, shuffle=True, random_state=seed) 

irrf = IR_RF(n_estimators=100, random_state=seed)
irrf.fit(x_train, y_train)

x_test_rank = irrf.rank(x_test, class_to_rank=1)

rank_sort_index = np.argsort(x_test_rank, kind="stable")
true_sort_index = np.argsort(tp_test, kind="stable")

y_test_rank_sort = y_test[rank_sort_index]
y_test_true_sort = y_test[true_sort_index]

tau, p_value = kendalltau(y_test_true_sort, y_test_rank_sort)
print("tau", tau)
