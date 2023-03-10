
### Impots

import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
import pandas as pd
import Data.data_provider as dp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from estimators.IR_RF_estimator import IR_RF
from estimators.CRF_estimator import CRF_calib
from estimators.Elkan_estimator import Elkan_calib
from estimators.Venn_estimator import Venn_calib
from estimators.VA_estimator import VA_calib
from estimators.TLR_estimator import treeLR_calib
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration
from betacal import BetaCalibration

from CalibrationM import confidance_ECE, convert_prob_2D
from sklearn.metrics import brier_score_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

### Parameters

runs = 1
n_estimators=100

plot_bins = 10
test_size = 0.3


oob = False

plot = True
save_results = False

results_dict = {}

samples = 3000
features = 40
calib_methods = ["RF", "Platt" , "ISO", "Rank", "CRF", "prank", "Venn", "VA", "Beta", "Elkan", "tlr"]
# calib_methods = ["RF", "tlr"]
metrics = ["acc", "auc", "brier", "ece", "tce"]

run_name = "Samples"


def split_train_calib_test(name, X, y, test_size, calib_size, seed=0, tp=np.zeros(10)):
    ### spliting data to train calib and test
    x_train_calib, x_test, y_train_calib, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)
    x_train, x_calib, y_train, y_calib = train_test_split(x_train_calib, y_train_calib, test_size=calib_size, shuffle=True, random_state=seed)
    if not tp.all() == 0: 
        _, _, tp_train_calib, tp_test = train_test_split(X, tp, test_size=test_size, shuffle=True, random_state=seed)
        _, _, tp_train, tp_calib = train_test_split(x_train_calib, tp_train_calib, test_size=0.5, shuffle=True, random_state=seed)

    if tp.all() == 0: 
        data = {"name":name, "x_train": x_train, "x_calib":x_calib, "x_test":x_test, "y_train":y_train, "y_calib":y_calib, "y_test":y_test}

    else:
        data = {"name":name, "x_train": x_train, "x_calib":x_calib, "x_test":x_test, "y_train":y_train, "y_calib":y_calib, "y_test":y_test, "tp_train":tp_train, "tp_calib":tp_calib, "tp_test":tp_test}
    return data

def calibration(RF, data, calib_methods, metrics):

    # for res_val in ["prob", "decision"]:
    #     _dict = {}
    #     for method in calib_methods:
    #         _dict[method] = []
    #     results_dict[data["name"] + "_" + res_val] = _dict


    for metric in metrics:
        for method in calib_methods:
            results_dict[data["name"] + "_" + method + "_" + metric] = []

    # random forest probs
    rf_p_calib = RF.predict_proba(data["x_calib"], laplace=1)
    rf_p_test = RF.predict_proba(data["x_test"], laplace=1)
    results_dict[data["name"] + "_RF_prob"] = rf_p_test
    results_dict[data["name"] + "_RF_decision"] = np.argmax(rf_p_test,axis=1)

    # Platt scaling on RF
    if "Platt" in calib_methods:
        plat_calib = _SigmoidCalibration().fit(rf_p_calib[:,1], data["y_calib"])
        plat_p_test = convert_prob_2D(plat_calib.predict(rf_p_test[:,1]))
        results_dict[data["name"] + "_Platt_prob"] = plat_p_test
        results_dict[data["name"] + "_Platt_decision"] = np.argmax(plat_p_test,axis=1)

    # ISO calibration on RF
    if "ISO" in calib_methods:
        iso_calib = IsotonicRegression(out_of_bounds='clip').fit(rf_p_calib[:,1], data["y_calib"])
        iso_p_test = convert_prob_2D(iso_calib.predict(rf_p_test[:,1]))
        results_dict[data["name"] + "_ISO_prob"] = iso_p_test
        results_dict[data["name"] + "_ISO_decision"] = np.argmax(iso_p_test,axis=1)

    # RF ranking + ISO
    if "Rank" in calib_methods:
        x_calib_rank = RF.rank(data["x_calib"], class_to_rank=1, train_rank=True)
        x_test_rank = RF.rank_refrence(data["x_test"], class_to_rank=1)

        iso_rank = IsotonicRegression(out_of_bounds='clip').fit(x_calib_rank, data["y_calib"]) 
        rank_p_test = convert_prob_2D(iso_rank.predict(x_test_rank))
        results_dict[data["name"] + "_Rank_prob"] = rank_p_test
        results_dict[data["name"] + "_Rank_decision"] = np.argmax(rank_p_test,axis=1)

    # perfect rank + ISO
    if "prank" in calib_methods:
        iso_rank = IsotonicRegression(out_of_bounds='clip').fit(data["tp_calib"], data["y_calib"]) 
        rank_p_test = convert_prob_2D(iso_rank.predict(data["tp_test"]))
        results_dict[data["name"] + "_prank_prob"] = rank_p_test
        results_dict[data["name"] + "_prank_decision"] = np.argmax(rank_p_test,axis=1)

    # CRF calibrator
    if "CRF" in calib_methods:
        crf_calib = CRF_calib(learning_method="sig_brior").fit(rf_p_calib[:,1], data["y_calib"])
        crf_p_test = crf_calib.predict(rf_p_test[:,1])
        results_dict[data["name"] + "_CRF_prob"] = crf_p_test
        results_dict[data["name"] + "_CRF_decision"] = np.argmax(crf_p_test,axis=1)

    # Venn calibrator
    if "Venn" in calib_methods:
        ven_calib = Venn_calib().fit(rf_p_calib, data["y_calib"])
        ven_p_test = ven_calib.predict(rf_p_test)
        results_dict[data["name"] + "_Venn_prob"] = ven_p_test
        results_dict[data["name"] + "_Venn_decision"] = np.argmax(ven_p_test,axis=1)

    # Venn abers
    if "VA" in calib_methods:       
        VA = VA_calib().fit(rf_p_calib[:,1], data["y_calib"])
        va_p_test = VA.predict(rf_p_test[:,1])
        results_dict[data["name"] + "_VA_prob"] = va_p_test
        results_dict[data["name"] + "_VA_decision"] = np.argmax(va_p_test,axis=1)

    # Beta calibration
    if "Beta" in calib_methods:
        beta_calib = BetaCalibration(parameters="abm").fit(rf_p_calib[:,1], data["y_calib"])
        beta_p_test = convert_prob_2D(beta_calib.predict(rf_p_test[:,1]))
        results_dict[data["name"] + "_Beta_prob"] = beta_p_test
        results_dict[data["name"] + "_Beta_decision"] = np.argmax(beta_p_test,axis=1)

    # Elkan calibration
    if "Elkan" in calib_methods:
        elkan_calib = Elkan_calib().fit(data["y_train"], data["y_calib"])
        elkan_p_test = elkan_calib.predict(rf_p_test[:,1])
        results_dict[data["name"] + "_Elkan_prob"] = elkan_p_test
        results_dict[data["name"] + "_Elkan_decision"] = np.argmax(elkan_p_test,axis=1)

    # tree LR calib
    if "tlr" in calib_methods:
        tlr_calib = treeLR_calib().fit(RF, data["x_train"] ,data["y_train"], data["x_calib"], data["y_calib"])
        tlr_p_test = tlr_calib.predict(data["x_test"])
        results_dict[data["name"] + "_tlr_prob"] = tlr_p_test
        results_dict[data["name"] + "_tlr_decision"] = np.argmax(tlr_p_test,axis=1)


    if "acc" in metrics:
        for method in calib_methods:
            results_dict[data["name"] + "_" + method +"_acc"].append(accuracy_score(data["y_test"], results_dict[data["name"] + "_" + method +"_decision"]))

    if "auc" in metrics:
        for method in calib_methods:
            fpr, tpr, thresholds = roc_curve(data["y_test"], results_dict[data["name"] + "_" + method +"_prob"][:,1])
            results_dict[data["name"] + "_" + method +"_auc"].append(auc(fpr, tpr))

    if "ece" in metrics:
        for method in calib_methods:
            results_dict[data["name"] + "_" + method +"_ece"].append(confidance_ECE(results_dict[data["name"] + "_" + method +"_prob"], data["y_test"], bins=plot_bins))

    if "brier" in metrics:
        for method in calib_methods:
            results_dict[data["name"] + "_" + method +"_brier"].append(brier_score_loss(data["y_test"], results_dict[data["name"] + "_" + method +"_prob"][:,1]))

    if "logloss" in metrics:
        for method in calib_methods:
            results_dict[data["name"] + "_" + method +"_logloss"].append(brier_score_loss(data["y_test"], results_dict[data["name"] + "_" + method +"_prob"][:,1]))

    if "tce" in metrics:
        for method in calib_methods:
            results_dict[data["name"] + "_" + method +"_tce"].append(mean_squared_error(data["tp_test"], results_dict[data["name"] + "_" + method +"_prob"][:,1]))

    return results_dict


def update_runs(ref_dict, new_dict):

    # print("---------------------------------")
    # print("new_dict", new_dict)

    if ref_dict == {}:
        return new_dict.copy()
    
    res_dict = ref_dict.copy()
    for k in new_dict.keys():
        if "_prob" in k or "_decision" in k:
            continue
        res_dict[k] = ref_dict[k] + new_dict[k]

    return res_dict    

def mean_and_ranking_table(results_dict, metrics, calib_methods, data_list):
    # save results as txt
    df_dict = {}
    res = ""
    for metric in metrics:
        txt = "Data"
        for method in calib_methods:
            txt += "," + method

        for data in data_list:
            txt += "\n"+ data
            for method in calib_methods:
                txt += "," + str(np.array(results_dict[data+ "_" + method + "_"+ metric]).mean())

        txt_data = StringIO(txt)
        df = pd.read_csv(txt_data, sep=",")
        df.set_index('Data', inplace=True)
        mean_res = df.mean()
        if metric == "ece" or metric == "brier" or metric == "tce" or metric == "logloss":
            df_rank = df.rank(axis=1, ascending = True)
        else:
            df_rank = df.rank(axis=1, ascending = False)

        mean_rank = df_rank.mean()
        df.loc["Mean"] = mean_res
        df.loc["Rank"] = mean_rank
        df_dict[metric] = df
        if save_results:
            df.to_csv(f"./results/Synthetic/{data}_DataCalib_{metric}.csv",index=True)
        # print("---------------------------------", metric)
        # print(df)
        res += f"--------------------------------- {metric}\n"
        res += str(df)
    return df_dict

def exp_mean_rank_through_time(exp_df_all, exp_df, exp_value, value="rank", exp_test="Calibration"):
    value_index = -1
    if value == "mean":
        value_index = -2
    for k in exp_df.keys():
        table = exp_df[k]
        calib_values = table.iloc[value_index].to_dict()
        calib_values[exp_test] = exp_value
        exp_df_all[k] = pd.concat([exp_df_all[k], (pd.DataFrame([calib_values]))])
    return exp_df_all
