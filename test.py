from LR_RF_estimator import LR_RF
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

# load the data
from sklearn.datasets import load_digits
digits = load_digits()
features, target = digits.data, digits.target

# train RF
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.5, random_state=0)

model = LR_RF(max_depth=100, n_estimators=100, random_state=0)
model.fit(x_train, y_train)

LR_RF_acc = model.score(x_test, y_test)

print("acc", LR_RF_acc)
