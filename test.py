import numpy as np
import Data.data_provider as dp
from estimators.IR_RF_estimator import IR_RF
from sklearn.dummy import DummyClassifier

data_name = "wilt"
X, y = dp.load_data(data_name)
rf = IR_RF()
rf.fit(X, y)
rf_s = rf.score(X,y)


# dummy_clf = DummyClassifier(strategy="most_frequent").fit(X, y)
# d_s = dummy_clf.score(X, y)
print("acc", np.unique(y))