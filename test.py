
import Data.data_provider as dp
from Experiments import core as cal
from sklearn.ensemble import RandomForestClassifier
from estimators.IR_RF_estimator import IR_RF
import numpy as np

# data
data_name = "QSAR"
X, y = dp.load_data(data_name, ".")
data = cal.split_train_calib_test(data_name, X, y, 0.3, 0.3, 0)

print("---------------------------------RF")
rf = RandomForestClassifier(n_estimators=30, random_state=10).fit(data["x_train"], data["y_train"])
irrf = IR_RF(n_estimators=30, random_state=10).fit(data["x_train"], data["y_train"])

rf_prob = rf.predict_proba(data["x_test"])
rf_CT = irrf.predict_proba(data["x_test"], classifier_tree=True)
rf_PET = irrf.predict_proba(data["x_test"], classifier_tree=False)

# print("rf_prob\n", rf_prob)
# print("rf_CT\n", rf_CT)
# print("rf_PET\n", rf_PET)

if (rf_CT == rf_PET).all():
    print("Equal")
else:
    print("Not equal")

# def convert_prob_2D(prob1D):
#     prob_second_class = np.ones(len(prob1D)) - prob1D
#     prob2D = np.concatenate((prob_second_class.reshape(-1,1), prob1D.reshape(-1,1)), axis=1)
#     return prob2D


# s = np.random.uniform(0,1,5)
# # s = np.ones(1000)
# print("s", s)
# s = convert_prob_2D(s)

# s_avg = s.mean(axis=0)

# s_class = np.argmax(s, axis=1)
# s_cavg = s_class.mean()

# print("s", s)
# print("---------------------------------")
# print("s_avg", s_avg[1])
# print("---------------------------------")
# print("s_cavg", s_cavg)