import numpy as np
import Data.data_provider as dp
import Experiments.core_calib as cal

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 123

# Generate a binary classification dataset.3
X, y = dp.load_data("spambase")
print(len(X))
data_folds = cal.CV_split_train_calib_test("test", X, y, 5, RANDOM_STATE)

data = data_folds[0]
print("data", data.keys())   

# X, y = make_classification(
#     n_samples=500,
#     n_features=25,
#     n_clusters_per_class=1,
#     n_informative=15,
#     random_state=RANDOM_STATE,
# )

rf = RandomForestClassifier(
            warm_start=True,
            oob_score=True,
            max_features="sqrt",
            random_state=RANDOM_STATE,
        ).fit(data["x_train"], data["y_train"])

print("rf oob", rf.oob_decision_function_.shape)