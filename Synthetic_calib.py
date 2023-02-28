
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
from CalibrationM import confidance_ECE, convert_prob_2D
import Data.data_provider as dp
from sklearn.calibration import _SigmoidCalibration
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

runs = 1
n_estimators=10

plot_bins = 10
test_size = 0.3


oob = False

plot = True

results_dict = {}

samples = 10000
features = 40
calib_methods = ["RF", "Platt" , "ISO", "Rank", "CRF"]
metrics = ["acc", "auc", "brier", "ece", "tce"]

data = "Synthetic"

for res_val in ["prob", "decision"]:
    _dict = {}
    for method in calib_methods:
        _dict[method] = []
    results_dict[data + "_" + res_val] = _dict


for metric in metrics:
    _dict = {}
    for method in calib_methods:
        _dict[method] = []
    results_dict[data + "_" + metric] = _dict


for seed in range(runs):
    # seed = 5
    # print("seed ", seed)
    np.random.seed(seed)
    X, y, tp = dp.make_classification_gaussian_with_true_prob(samples, features , seed)

    ### spliting data to train calib and test
    x_train_calib, x_test, y_train_calib, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)
    x_train, x_calib, y_train, y_calib = train_test_split(x_train_calib, y_train_calib, test_size=0.5, shuffle=True, random_state=seed) 
    _, _, tp_train_calib, tp_test = train_test_split(X, tp, test_size=test_size, shuffle=True, random_state=seed)
    _, _, tp_train, tp_calib = train_test_split(x_train_calib, tp_train_calib, test_size=0.5, shuffle=True, random_state=seed) 

    # print("tp_test ", tp_test.shape)
    # exit()

    ### training the IRRF
    irrf = IR_RF(n_estimators=n_estimators, oob_score=oob, random_state=seed)
    irrf.fit(x_train, y_train)

    ### calibration and ECE plot

    # random forest probs
    rf_p_calib = irrf.predict_proba(x_calib, laplace=1)
    rf_p_test = irrf.predict_proba(x_test, laplace=1)
    results_dict[data + "_prob"]["RF"] = rf_p_test
    results_dict[data + "_decision"]["RF"] = np.argmax(rf_p_test,axis=1)

    # Platt scaling on RF
    plat_calib = _SigmoidCalibration().fit(rf_p_calib[:,1], y_calib)
    plat_p_test = convert_prob_2D(plat_calib.predict(rf_p_test[:,1]))
    results_dict[data + "_prob"]["Platt"] = plat_p_test
    results_dict[data + "_decision"]["Platt"] = np.argmax(plat_p_test,axis=1)

    # ISO calibration on RF
    iso_calib = IsotonicRegression(out_of_bounds='clip').fit(rf_p_calib[:,1], y_calib)
    iso_p_test = convert_prob_2D(iso_calib.predict(rf_p_test[:,1]))
    results_dict[data + "_prob"]["ISO"] = iso_p_test
    results_dict[data + "_decision"]["ISO"] = np.argmax(iso_p_test,axis=1)


    # Ranking with the RF
    x_calib_rank = irrf.rank(x_calib, class_to_rank=1, train_rank=True)
    x_test_rank = irrf.rank_refrence(x_test, class_to_rank=1)

    # RF ranking + ISO
    iso_rank = IsotonicRegression(out_of_bounds='clip').fit(x_calib_rank, y_calib) 
    rank_p_test = convert_prob_2D(iso_rank.predict(x_test_rank))
    results_dict[data + "_prob"]["Rank"] = rank_p_test
    results_dict[data + "_decision"]["Rank"] = np.argmax(rank_p_test,axis=1)

    # CRF calibrator
    crf_calib = CRF_calib(learning_method="sig_brior").fit(rf_p_calib[:,1], y_calib)
    crf_p_test = crf_calib.predict(rf_p_test[:,1])
    results_dict[data + "_prob"]["CRF"] = crf_p_test
    results_dict[data + "_decision"]["CRF"] = np.argmax(crf_p_test,axis=1)

    if "acc" in metrics:
        for method in calib_methods:
            results_dict[data + "_acc"][method].append(accuracy_score(y_test, results_dict[data + "_decision"][method]))

    if "auc" in metrics:
        for method in calib_methods:
            fpr, tpr, thresholds = roc_curve(y_test, results_dict[data + "_prob"][method][:,1])
            results_dict[data + "_auc"][method].append(auc(fpr, tpr))

    if "ece" in metrics:
        for method in calib_methods:
            results_dict[data + "_ece"][method].append(confidance_ECE(results_dict[data + "_prob"][method], y_test, bins=plot_bins))

    if "brier" in metrics:
        for method in calib_methods:
            results_dict[data + "_brier"][method].append(brier_score_loss(y_test, results_dict[data + "_prob"][method][:,1]))

    if "tce" in metrics:
        for method in calib_methods:
            results_dict[data + "_tce"][method].append(mean_squared_error(tp_test, results_dict[data + "_prob"][method][:,1]))
    
    if plot:

        for method in calib_methods:
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.scatter(tp_test, results_dict[data + "_prob"][method][:,1], marker='.', label=method)
            plt.xlabel("True probability")
            plt.ylabel("Predicted probability")
            plt.legend()
            plt.savefig(f"./results/Synthetic/plots/{method}.png")
            plt.close()

print(f"data {data} done")

# save results as txt
for metric in metrics:
    txt = "Data"
    for method in calib_methods:
        txt += "," + method
    txt += "\n"+ data
    for method in calib_methods:
        txt += "," + str(np.array(results_dict[data+ "_" +metric][method]).mean())
    txt_data = StringIO(txt)
    df = pd.read_csv(txt_data, sep=",")
    df.set_index('Data', inplace=True)
    mean_res = df.mean()
    if metric == "ece" or metric == "brier" or metric == "tce":
        df_rank = df.rank(axis=1, ascending = True)
    else:
        df_rank = df.rank(axis=1, ascending = False)

    mean_rank = df_rank.mean()
    # df.loc["Mean"] = mean_res
    df.loc["Rank"] = mean_rank
    df.to_csv(f"./results/Synthetic/{data}_DataCalib_{metric}.csv",index=False)
    print("---------------------------------", metric)
    print(df)
