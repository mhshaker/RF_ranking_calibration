from sklearn.isotonic import IsotonicRegression
import numpy as np
from estimators.IR_RF_estimator import IR_RF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import make_classification
import Data.data_provider as dp

# data
X, y = make_classification(n_samples=10000, n_features=40, n_informative=2, n_redundant=10, random_state=42)

# train RF
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

irrf = IR_RF(n_estimators=25, random_state=0)
irrf.fit(x_train, y_train)

iso_reg = IsotonicRegression().fit(X, y)
iso_reg.predict([.1, .2])

rf_roc = roc_auc_score(y_test, irrf.predict_proba(x_test, laplace=0)[:, 1])
lap_rf_roc = roc_auc_score(y_test, irrf.predict_proba(x_test, laplace=1)[:, 1])
irrf_roc = roc_auc_score(y_test, irrf.rank(x_test, class_to_rank=1))

print("rf_roc     ", rf_roc)
print("lap_rf_roc ", lap_rf_roc)
print("irrf_roc   ", irrf_roc)
