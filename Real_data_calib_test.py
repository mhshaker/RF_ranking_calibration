
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
import pandas as pd
from sklearn.model_selection import train_test_split
from estimators.IR_RF_estimator import IR_RF
from estimators.CRF_estimator import CRF_calib
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from CalibrationM import confidance_ECE, convert_prob_2D, classwise_ECE
import Data.data_provider as dp
from sklearn.calibration import _SigmoidCalibration
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

runs = 1
n_estimators=100

plot_bins = 10
test_size = 0.3

ece_score = True
brier_score = True
acc_score = True
auc_score = True

oob = False

plot = False

results_dict = {}

seed = 0
calib_methods = ["RF", "Platt" , "ISO", "Rank", "CRF"]
metrics = ["acc", "auc", "brier", "ece"]
data_list = ["spambase"]

def Venn_predictor(preds, labels):

    classes = np.unique(preds)
    n_test_samples = len(preds)
    pred_label_pairs = []

    for c in classes:
        c_pred_label_pairs = []
        for i in range(n_test_samples):
            if preds[i] == c:
                c_pred_label_pairs.append((preds[i], labels[i]))
        pred_label_pairs.append(c_pred_label_pairs)

    lower = []
    upper = []

    for c in classes:
        for cl in classes:
            cnt = 0
            for pair in pred_label_pairs[c]:
                label = pair[1]
                if label == cl:
                    cnt += 1
            
            low = cnt / (len(pred_label_pairs[c])+1)
            up = cnt+1 / (len(pred_label_pairs[c])+1)
            
            lower.append(low)
            upper.append(up)

    return lower, upper

for data in data_list:

    X, y = dp.load_data(data)

    for seed in range(runs):
        # seed = 5
        # print("seed ", seed)
        np.random.seed(seed)

        ### spliting data to train calib and test
        x_train_calib, x_test, y_train_calib, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)
        if oob:
            x_train = x_train_calib
            y_train = y_train_calib
        else:
            x_train, x_calib, y_train, y_calib = train_test_split(x_train_calib, y_train_calib, test_size=0.3, shuffle=True, random_state=seed) 

        ### training the IRRF
        irrf = IR_RF(n_estimators=n_estimators, oob_score=oob, random_state=seed)
        irrf.fit(x_train, y_train)

        ### calibration and ECE plot

        # random forest probs
        rf_p_calib = irrf.predict_proba(x_calib, laplace=1)
        rf_p_test = irrf.predict_proba(x_test, laplace=1)
        rf_d_test = np.argmax(rf_p_test,axis=1)

        lower, upper = Venn_predictor(preds=rf_d_test, labels=y_test)
        print(f'lowers: {lower} \nuppers: {upper}')