
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
from sklearn.calibration import calibration_curve
y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9,  1.])
prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=3)
print("prob_true", prob_true) 
print("prob_pred", prob_pred) 
