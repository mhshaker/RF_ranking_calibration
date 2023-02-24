
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
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from CalibrationM import confidance_ECE, convert_prob_2D, classwise_ECE
import Data.data_provider as dp
from sklearn.calibration import _SigmoidCalibration
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

runs = 10
n_estimators=10

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
calib_methods = ["RF", "Platt" , "ISO", "Rank"]
metrics = ["acc", "auc", "brier", "ece"]
data_list = ["spambase", "climate", "QSAR", "bank", "climate", "parkinsons", "vertebral", "ionosphere", "diabetes", "breast", "blod"]
# data_list = ["parkinsons", "vertebral"]

for data in data_list:

    X, y = dp.load_data(data)

    for metric in metrics:
        _dict = {}
        for method in calib_methods:
            _dict[method] = []
        results_dict[data + "_" + metric] = _dict

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

        # Platt scaling on RF
        sig_rf = _SigmoidCalibration().fit(rf_p_calib[:,1], y_calib)
        rf_cp_sig_test = convert_prob_2D(sig_rf.predict(rf_p_test[:,1]))
        rf_d_platt_test = np.argmax(rf_cp_sig_test,axis=1)

        # ISO calibration on RF
        iso_rf = IsotonicRegression(out_of_bounds='clip').fit(rf_p_calib[:,1], y_calib)
        rf_cp_test = convert_prob_2D(iso_rf.predict(rf_p_test[:,1]))
        rf_d_iso_test = np.argmax(rf_cp_test,axis=1)

        # Ranking with the RF
        x_calib_rank = irrf.rank(x_calib, class_to_rank=1, train_rank=True)
        x_test_rank = irrf.rank_refrence(x_test, class_to_rank=1)

        # RF ranking + ISO
        iso_rank = IsotonicRegression(out_of_bounds='clip').fit(x_calib_rank, y_calib) 
        irrf_cp_test = convert_prob_2D(iso_rank.predict(x_test_rank))
        rf_d_rank_test = np.argmax(irrf_cp_test,axis=1)

        if "acc" in metrics:
            results_dict[data + "_acc"]["RF"].append(accuracy_score(y_test, rf_d_test))
            results_dict[data + "_acc"]["Platt"].append(accuracy_score(y_test, rf_d_platt_test))
            results_dict[data + "_acc"]["ISO"].append(accuracy_score(y_test, rf_d_iso_test))
            results_dict[data + "_acc"]["Rank"].append(accuracy_score(y_test, rf_d_rank_test))

        if "auc" in metrics:
            fpr, tpr, thresholds = roc_curve(y_test, rf_p_test[:,1])
            results_dict[data + "_auc"]["RF"].append(auc(fpr, tpr))
            fpr, tpr, thresholds = roc_curve(y_test, rf_cp_sig_test[:,1])
            results_dict[data + "_auc"]["Platt"].append(auc(fpr, tpr))
            fpr, tpr, thresholds = roc_curve(y_test, rf_cp_test[:,1])
            results_dict[data + "_auc"]["ISO"].append(auc(fpr, tpr))
            fpr, tpr, thresholds = roc_curve(y_test, irrf_cp_test[:,1])
            results_dict[data + "_auc"]["Rank"].append(auc(fpr, tpr))

        if "ece" in metrics:
            results_dict[data + "_ece"]["RF"].append(confidance_ECE(rf_p_test, y_test, bins=plot_bins))
            results_dict[data + "_ece"]["Platt"].append(confidance_ECE(rf_cp_sig_test, y_test, bins=plot_bins))
            results_dict[data + "_ece"]["ISO"].append(confidance_ECE(rf_cp_test, y_test, bins=plot_bins))
            results_dict[data + "_ece"]["Rank"].append(confidance_ECE(irrf_cp_test, y_test, bins=plot_bins))

        if "brier" in metrics:
            results_dict[data + "_brier"]["RF"].append(brier_score_loss(y_test, rf_p_test[:,1]))
            results_dict[data + "_brier"]["Platt"].append(brier_score_loss(y_test,rf_cp_sig_test[:,1]))
            results_dict[data + "_brier"]["ISO"].append(brier_score_loss(y_test, rf_cp_test[:,1]))
            results_dict[data + "_brier"]["Rank"].append(brier_score_loss(y_test, irrf_cp_test[:,1]))

        if plot:
            tp, pp = calibration_curve(y_test, rf_p_test[:,1], n_bins=plot_bins)
            tp_iso, pp_iso = calibration_curve(y_test, rf_cp_test, n_bins=plot_bins)
            tp_sig, pp_sig = calibration_curve(y_test, rf_cp_sig_test, n_bins=plot_bins)
            tp_irrf, pp_irrf = calibration_curve(y_test, irrf_cp_test, n_bins=plot_bins)
            
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.plot(tp, pp, marker='.', label="RF")
            # plt.plot(tp_sig, pp_sig, marker='.', label="RF+sig")
            plt.plot(tp_iso, pp_iso, marker='.', label="RF+iso")
            plt.plot(tp_irrf, pp_irrf, marker='.', label="RF+rank+ios", c="black")
            plt.xlabel("True probability")
            plt.ylabel("Mean predicted probability")
            plt.legend()
            plt.show()

            plt.hist(pp_iso, color="green")
            plt.ylabel("Count")
            plt.xlabel("Mean predicted probability")

            plt.show()
            plt.hist(pp,color="black")
            plt.ylabel("Count")
            plt.xlabel("Mean predicted probability")
            plt.show()
    print(f"data {data} done")

# save results as txt
for metric in metrics:
    txt = "Data"
    for method in calib_methods:
        txt += "," + method
    for data in data_list:
        txt += "\n"+ data
        for method in calib_methods:
            txt += "," + str(np.array(results_dict[data+ "_" +metric][method]).mean())
    txt_data = StringIO(txt)
    df = pd.read_csv(txt_data, sep=",")
    df.set_index('Data', inplace=True)
    mean_res = df.mean()
    if metric == "ece" or metric == "brier":
        df_rank = df.rank(axis=1, ascending = True)
    else:
        df_rank = df.rank(axis=1, ascending = False)

    mean_rank = df_rank.mean()
    df.loc["Mean"] = mean_res
    df.loc["Rank"] = mean_rank
    df.to_csv(f"results/RealDataCalib_{metric}.csv",index=False)
    print("---------------------------------", metric)
    print(df)
