# connect to DB and load datasets
# calibration methods to run on datasets
# calibration measures for evaluation

### road map

# implement LR-RF
    # convert normal data to ranking data
    # train RF on normal data
    # at test time fetch the lables of each leaf from ranking data
    # agregate the ranking labels using Borda method
# convert ranking to calibration using ISO
# compare to other calib approches

###

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from estimators.LR_RF_estimator import LR_RF
from estimators.LR_RF_estimator import convert_to_ranking
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import CalibrationM as calibm

from sklearn.datasets import make_classification


# import Data.data_provider as dp
# dataset = "adult"
# features, target = dp.load_data(dataset)


# load the data
from sklearn.datasets import load_digits
digits = load_digits()
features, target = digits.data, digits.target

# synthetic data
features, target = make_classification(n_samples=100_000, n_features=20, n_informative=2, n_redundant=2, random_state=42)

train_samples = 100  # Samples used for training the models
X_train, X_test, y_train, y_test = train_test_split(features, target, shuffle=False,test_size=100_000 - train_samples)


# convert lables to ranking
target_rank, y_top_rank = convert_to_ranking(features, target)

# split to train and test
x_train, x_test, y_train, y_test = train_test_split(features, y_top_rank, test_size=0.5, random_state=0)
_, _, y_train_rank, y_test_rank = train_test_split(features, target_rank, test_size=0.5, random_state=0)

# train RF
model = RandomForestClassifier(max_depth=100, n_estimators=100, random_state=0)
model.fit(x_train, y_train)
RF_probs = model.predict_proba(x_test)
RF_pred = np.argmax(RF_probs, axis=1) # which is equal to the sklearn model.predict(x_test)

# train LR_RF
model_lr = LR_RF(max_depth=100, n_estimators=100, random_state=0)
model_lr.fit(x_train, y_train, y_train_rank)
LR_RF_probs = model_lr.predict_proba(x_test)
LR_RF_pred = model_lr.predict(x_test)

# compare RF and LR_RF acc
print("acc RF       ", accuracy_score(y_test, RF_pred))
print("acc LR_RF_sk ",accuracy_score(y_test, LR_RF_pred))

print("---------------------------------")
RF_cw_ece = calibm.classwise_ECE(RF_probs, y_test)
LR_RFcw_ece = calibm.classwise_ECE(LR_RF_probs, y_test)
print("cw_ece RF    ", np.array(RF_cw_ece).mean(axis=0))
print("cw_ece LR_RF ", np.array(LR_RFcw_ece).mean(axis=0))
