import numpy as np
import Data.data_provider as dp
from estimators.IR_RF_estimator import IR_RF
from sklearn.model_selection import train_test_split
from scipy.stats import kendalltau

seed = 0
np.random.seed(seed)
samples = 10000
X, y, tp = dp.make_classification_with_true_prob(50,2,samples,seed)

# print("x", X.shape)

### spliting data to train calib and test
test_size = 0.4
x_train_calib, x_test, y_train_calib, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)
x_train, x_calib, y_train, y_calib = train_test_split(x_train_calib, y_train_calib, test_size=0.5, shuffle=True, random_state=seed) 
_, _, tp_train_calib, tp_test = train_test_split(X, tp, test_size=test_size, shuffle=True, random_state=seed)
_, _, tp_train, tp_calib = train_test_split(x_train_calib, tp_train_calib, test_size=0.5, shuffle=True, random_state=seed) 

# print("x", x_train.shape)

irrf = IR_RF(n_estimators=100, random_state=seed)
irrf.fit(x_train, y_train)

rank1 = irrf.rank(x_test, class_to_rank=1)
rank0 = irrf.rank(x_test, class_to_rank=0)


rank1_index = np.argsort(rank1, kind="stable")
rank1_index_neg = np.argsort(-rank1, kind="stable")
rank0_index = np.argsort(rank0, kind="stable")

rank1_sort = tp_test[rank1_index]
rank1_sort_neg = tp_test[rank1_index_neg]
rank0_sort = tp_test[rank0_index]

tau, p_value = kendalltau(rank1, -rank1)

print("tau", tau)
