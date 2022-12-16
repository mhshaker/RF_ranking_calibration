import numpy as np
from IR_RF_estimator import IR_RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import make_blobs

# data
X, y = make_blobs(n_samples=2000, n_features=2, centers=2, random_state=42, cluster_std=5.0)

# train RF
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

irrf = IR_RF(n_estimators=25, random_state=0)
irrf.fit(x_train, y_train)

rf_roc = roc_auc_score(y_test, irrf.predict_proba(x_test, laplace=0)[:, 1])
lap_rf_roc = roc_auc_score(y_test, irrf.predict_proba(x_test, laplace=1)[:, 1])
irrf_roc = roc_auc_score(y_test, irrf.rank(x_test, class_to_rank=1))

print("rf_roc     ", rf_roc)
print("lap_rf_roc ", lap_rf_roc)
print("irrf_roc   ", irrf_roc)

# r = irrf.rank(x_test[:6])

# probs_L0 = irrf.predict_proba(x_test, laplace=0)
# probs_L1 = irrf.predict_proba(x_test)

# print(probs_L0[:5])
# print("---------------------------------")
# print(probs_L1[:5])