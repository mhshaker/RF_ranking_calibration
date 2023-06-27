import numpy as np
import Data.data_provider as dp
from estimators.IR_RF_estimator import IR_RF

data_name = "Customer_Churn"
X, y = dp.load_data(data_name)

rf = IR_RF()
rf.fit(X, y)
acc = rf.score(X,y)


print("acc", np.unique(y))