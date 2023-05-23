
import Data.data_provider as dp
from Experiments import core as cal
from estimators.LR_estimator import LR_u as lr
from sklearn.linear_model import LogisticRegression

data_name = "spambase"
X, y = dp.load_data(data_name, ".")
data = cal.split_train_calib_test(data_name, X, y, 0.3, 0.3, 0)


m = lr(random_state=0).fit(data["x_train"], data["y_train"])
print("intercept before", m.intercept_)
acc = m.score(data["x_test"], data["y_test"])
print("acc", acc)
m = m.update_intercept(data["x_calib"], data["y_calib"])
print("intercept after ", m.intercept_)
acc = m.score(data["x_test"], data["y_test"])
print("acc", acc)

