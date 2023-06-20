
### Impots
import os
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
from estimators.boot_calib import Boot_calib
from estimators.bin_calib import Bin_calib
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration
from betacal import BetaCalibration
from sklearn.linear_model import LinearRegression

from CalibrationM import confidance_ECE, convert_prob_2D
from sklearn.metrics import brier_score_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc


tvec = np.linspace(0.01, 0.99, 990)


def split_train_calib_test(name, X, y, test_size, calib_size, orig_seed=0, tp=np.zeros(10)):
    ### spliting data to train calib and test
    for i in range(1000, 1100): # the for loop is to make sure the calib train and test split all consist of both classes of the binary dataset
        seed = i + orig_seed
        x_train_calib, x_test, y_train_calib, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)
        x_train, x_calib, y_train, y_calib = train_test_split(x_train_calib, y_train_calib, test_size=calib_size, shuffle=True, random_state=seed)
        if not tp.all() == 0: 
            _, _, tp_train_calib, tp_test = train_test_split(X, tp, test_size=test_size, shuffle=True, random_state=seed)
            _, _, tp_train, tp_calib = train_test_split(x_train_calib, tp_train_calib, test_size=calib_size, shuffle=True, random_state=seed)

        if tp.all() == 0: 
            data = {"name":name, "x_train": x_train, "x_calib":x_calib, "x_test":x_test, "y_train":y_train, "y_calib":y_calib, "y_test":y_test}

        else:
            data = {"name":name, "x_train": x_train, "x_calib":x_calib, "x_test":x_test, "y_train":y_train, "y_calib":y_calib, "y_test":y_test, "tp_train":tp_train, "tp_calib":tp_calib, "tp_test":tp_test}
        
        if len(np.unique(data["y_calib"])) > 1 and len(np.unique(data["y_test"])) > 1 and len(np.unique(data["y_train"])) > 1:
            break

    return data

def calibration(RF, data, params):
    data_name = data["name"]
    calib_methods = params["calib_methods"] 
    metrics = params["metrics"]
    # the retuen is a dict with all the metrics results as well as RF probs and every calibration method decision for every test data point
    # the structure of the keys in the dict is data_calibMethod_metric
    results_dict = {}
    for metric in metrics:
        for method in calib_methods:
            results_dict[data["name"] + "_" + method + "_" + metric] = []

    # random forest probs
    rf_p_calib = RF.predict_proba(data["x_calib"])
    rf_p_test = RF.predict_proba(data["x_test"])
    results_dict[data["name"] + "_RF_prob"] = rf_p_test
    results_dict[data["name"] + "_RF_prob_train"] = RF.predict_proba(data["x_train"])
    results_dict[data["name"] + "_RF_prob_calib"] = rf_p_calib
    results_dict[data["name"] + "_RF_decision"] = np.argmax(rf_p_test,axis=1)

    # all input probs to get the fit calib model
    method = "bin"
    if method in calib_methods:
        rf_p_train = results_dict[data["name"] + "_RF_prob_train"]
        bc = Bin_calib(params["ece_bins"]).fit(rf_p_train[:,1], data["y_train"], rf_p_calib[:,1], data["y_calib"])
        bin_p_test = convert_prob_2D(bc.predict(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = bin_p_test
        results_dict[f"{data_name}_{method}_decision"] = np.argmax(bin_p_test,axis=1)
    
    method = "RF_boot"
    if method in calib_methods:
        rf_tree_test = RF.predict_proba(data["x_test"], return_tree_prob=True)
        bc = Boot_calib(boot_count=params["boot_count"], bootstrap_size= params["boot_size"])
        bc_p_test = bc.predict(rf_tree_test)
        results_dict[f"{data_name}_{method}_prob"] = bc_p_test
        results_dict[f"{data_name}_{method}_decision"] = np.argmax(bc_p_test,axis=1)
    
    method = "RF_ens"
    if method in calib_methods:
        bc = Boot_calib(boot_count=params["boot_count"])
        bc_p_test = bc.predict_ens(data["x_test"], data["x_train"], data["y_train"], RF)
        results_dict[f"{data_name}_{method}_prob"] = bc_p_test
        results_dict[f"{data_name}_{method}_decision"] = np.argmax(bc_p_test,axis=1)

    method = "RF_ensbin"
    if method in calib_methods:
        rf_tree_test = RF.predict_proba(data["x_test"])
        bc = Boot_calib(boot_count=params["boot_count"]).fit(data["x_train"], data["y_train"], RF)
        bc_p_test = convert_prob_2D(bc.predict_ens2(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = bc_p_test
        results_dict[f"{data_name}_{method}_decision"] = np.argmax(bc_p_test,axis=1)

    method = "RF_CT"
    if method in calib_methods:
        rf_ct_test = RF.predict_proba(data["x_test"], classifier_tree=True)
        results_dict[f"{data_name}_{method}_prob"] = rf_ct_test
        results_dict[f"{data_name}_{method}_decision"] = np.argmax(rf_ct_test,axis=1)

    method = "RF_Laplace"
    if method in calib_methods:
        rf_lap_test = RF.predict_proba(data["x_test"], laplace=1)
        results_dict[f"{data_name}_{method}_prob"] = rf_lap_test
        results_dict[f"{data_name}_{method}_decision"] = np.argmax(rf_lap_test,axis=1)

    method = "Line"
    if method in calib_methods:
        lr_calib = LinearRegression().fit(rf_p_calib, data["y_calib"])
        y_pred_clipped = np.clip(lr_calib.predict(rf_p_test), 0, 1)
        lr_p_test = convert_prob_2D(y_pred_clipped)
        results_dict[f"{data_name}_{method}_prob"] = lr_p_test
        results_dict[f"{data_name}_{method}_decision"] = np.argmax(lr_p_test,axis=1)
        results_dict[f"{data_name}_{method}_fit"] = np.clip(lr_calib.predict(convert_prob_2D(tvec)), 0, 1)

    method = "Platt"
    if method in calib_methods:
        plat_calib = _SigmoidCalibration().fit(rf_p_calib[:,1], data["y_calib"])
        plat_p_test = convert_prob_2D(plat_calib.predict(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = plat_p_test
        results_dict[f"{data_name}_{method}_decision"] = np.argmax(plat_p_test,axis=1)
        results_dict[f"{data_name}_{method}_fit"] = plat_calib.predict(tvec)

    # ISO calibration on RF
    method = "ISO"
    if method in calib_methods:
        iso_calib = IsotonicRegression(out_of_bounds='clip').fit(rf_p_calib[:,1], data["y_calib"])
        iso_p_test = convert_prob_2D(iso_calib.predict(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = iso_p_test
        results_dict[f"{data_name}_{method}_decision"] = np.argmax(iso_p_test,axis=1)
        results_dict[f"{data_name}_{method}_fit"] = iso_calib.predict(tvec)

    # RF ranking + ISO
    method = "Rank"
    if method in calib_methods:
        x_calib_rank = RF.rank(data["x_calib"], class_to_rank=1, train_rank=True)
        x_test_rank = RF.rank_refrence(data["x_test"], class_to_rank=1)

        iso_rank = IsotonicRegression(out_of_bounds='clip').fit(x_calib_rank, data["y_calib"]) 
        rank_p_test = convert_prob_2D(iso_rank.predict(x_test_rank))
        results_dict[f"{data_name}_{method}_prob"] = rank_p_test
        results_dict[f"{data_name}_{method}_decision"] = np.argmax(rank_p_test,axis=1)
        # tvec_rank = RF.rank_refrence(data["x_test"], class_to_rank=1)
        # results_dict[data["name"] + "_Rank_fit"] = iso_rank.predict(tvec_rank)

    # perfect rank + ISO
    method = "prank"
    if method in calib_methods:
        iso_rank = IsotonicRegression(out_of_bounds='clip').fit(data["tp_calib"], data["y_calib"]) 
        rank_p_test = convert_prob_2D(iso_rank.predict(data["tp_test"]))
        results_dict[f"{data_name}_{method}_prob"] = rank_p_test
        results_dict[f"{data_name}_{method}_decision"] = np.argmax(rank_p_test,axis=1)

    # CRF calibrator
    method = "CRF"
    if method in calib_methods:
        crf_calib = CRF_calib(learning_method="sig_brior").fit(rf_p_calib[:,1], data["y_calib"])
        crf_p_test = crf_calib.predict(rf_p_test[:,1])
        results_dict[f"{data_name}_{method}_prob"] = crf_p_test
        results_dict[f"{data_name}_{method}_decision"] = np.argmax(crf_p_test,axis=1)
        results_dict[f"{data_name}_{method}_fit"] = crf_calib.predict(tvec)[:,1]

    # Venn calibrator
    method = "Venn"
    if method in calib_methods:
        ven_calib = Venn_calib().fit(rf_p_calib, data["y_calib"])
        ven_p_test = ven_calib.predict(rf_p_test)
        results_dict[f"{data_name}_{method}_prob"] = ven_p_test
        results_dict[f"{data_name}_{method}_decision"] = np.argmax(ven_p_test,axis=1)
        results_dict[f"{data_name}_{method}_fit"] = ven_calib.predict(convert_prob_2D(tvec))[:,1]

    # Venn abers
    method = "VA"
    if method in calib_methods:
        VA = VA_calib().fit(rf_p_calib[:,1], data["y_calib"])
        va_p_test = convert_prob_2D(VA.predict(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = va_p_test
        results_dict[f"{data_name}_{method}_decision"] = np.argmax(va_p_test,axis=1)
        results_dict[f"{data_name}_{method}_fit"] = VA.predict(tvec)

    # Beta calibration
    method = "Beta"
    if method in calib_methods:
        beta_calib = BetaCalibration(parameters="abm").fit(rf_p_calib[:,1], data["y_calib"])
        beta_p_test = convert_prob_2D(beta_calib.predict(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = beta_p_test
        results_dict[f"{data_name}_{method}_decision"] = np.argmax(beta_p_test,axis=1)
        results_dict[f"{data_name}_{method}_fit"] = beta_calib.predict(tvec)

    # Elkan calibration
    method = "Elkan"
    if method in calib_methods:
        elkan_calib = Elkan_calib().fit(data["y_train"], data["y_calib"])
        elkan_p_test = elkan_calib.predict(rf_p_test[:,1])
        results_dict[f"{data_name}_{method}_prob"] = elkan_p_test
        results_dict[f"{data_name}_{method}_decision"] = np.argmax(elkan_p_test,axis=1)
        results_dict[f"{data_name}_{method}_fit"] = elkan_calib.predict(tvec)[:,1]

    # tree LR calib
    method = "tlr"
    if method in calib_methods:
        tlr_calib = treeLR_calib().fit(RF, data["x_train"] ,data["y_train"], data["x_calib"], data["y_calib"])
        tlr_p_test = tlr_calib.predict(data["x_test"])
        results_dict[f"{data_name}_{method}_prob"] = tlr_p_test
        results_dict[f"{data_name}_{method}_decision"] = np.argmax(tlr_p_test,axis=1)
        # results_dict[data["name"] + "_tlr_fit"] = tlr_calib.predict(convert_prob_2D(tvec))[:,1]

    # for key in results_dict:
    #     print("key ", key)   


    if "acc" in metrics:
        for method in calib_methods:
            results_dict[f"{data_name}_{method}_acc"].append(accuracy_score(data["y_test"], results_dict[f"{data_name}_{method}_decision"]))

    if "auc" in metrics:
        for method in calib_methods:
            fpr, tpr, thresholds = roc_curve(data["y_test"], results_dict[data["name"] + "_" + method +"_prob"][:,1])
            results_dict[f"{data_name}_{method}_auc"].append(auc(fpr, tpr))

    if "ece" in metrics:
        for method in calib_methods:
            results_dict[f"{data_name}_{method}_ece"].append(confidance_ECE(results_dict[f"{data_name}_{method}_prob"], data["y_test"], bins=params["ece_bins"]))

    if "brier" in metrics:
        for method in calib_methods:
            results_dict[f"{data_name}_{method}_brier"].append(brier_score_loss(data["y_test"], results_dict[f"{data_name}_{method}_prob"][:,1]))

    if "logloss" in metrics:
        for method in calib_methods:
            results_dict[f"{data_name}_{method}_logloss"].append(brier_score_loss(data["y_test"], results_dict[f"{data_name}_{method}_prob"][:,1]))

    if "tce" in metrics:
        for method in calib_methods:
            results_dict[f"{data_name}_{method}_tce"].append(mean_squared_error(data["tp_test"], results_dict[f"{data_name}_{method}_prob"][:,1]))

    return results_dict


def model_calibration(models, data, metrics, plot_bins = 10):

    # the retuen is a dict with all the metrics results as well as RF probs and every calibration method decision for every test data point
    # the structure of the keys in the dict is data_calibMethod_metric
    results_dict = {}
    for metric in metrics:
        for model in models:
            results_dict[data["name"] + "_" + model + "_" + metric] = []

    # random forest probs
    for model_key in models:
        if model_key == "RF_l":
            p_test = models[model_key].predict_proba(data["x_test"], laplace=1)
        elif model_key == "RF_ct":
            p_test = models[model_key].predict_proba(data["x_test"], classifier_tree=True)
        else:
            p_test = models[model_key].predict_proba(data["x_test"])
        results_dict[data["name"] + f"_{model_key}_prob"] = p_test
        results_dict[data["name"] + f"_{model_key}_decision"] = np.argmax(p_test,axis=1)


    if "acc" in metrics:
        for model in models:
            results_dict[data["name"] + "_" + model +"_acc"].append(accuracy_score(data["y_test"], results_dict[data["name"] + "_" + model +"_decision"]))

    if "auc" in metrics:
        for model in models:
            fpr, tpr, thresholds = roc_curve(data["y_test"], results_dict[data["name"] + "_" + model +"_prob"][:,1])
            results_dict[data["name"] + "_" + model +"_auc"].append(auc(fpr, tpr))

    if "ece" in metrics:
        for model in models:
            results_dict[data["name"] + "_" + model +"_ece"].append(confidance_ECE(results_dict[data["name"] + "_" + model +"_prob"], data["y_test"], bins=plot_bins))

    if "brier" in metrics:
        for model in models:
            results_dict[data["name"] + "_" + model +"_brier"].append(brier_score_loss(data["y_test"], results_dict[data["name"] + "_" + model +"_prob"][:,1]))

    if "logloss" in metrics:
        for model in models:
            results_dict[data["name"] + "_" + model +"_logloss"].append(brier_score_loss(data["y_test"], results_dict[data["name"] + "_" + model +"_prob"][:,1]))

    if "tce" in metrics:
        for model in models:
            results_dict[data["name"] + "_" + model +"_tce"].append(mean_squared_error(data["tp_test"], results_dict[data["name"] + "_" + model +"_prob"][:,1]))

    return results_dict


def update_runs(ref_dict, new_dict):

    # calib results for every run for the same dataset is aggregated in this function 
    # (ex. acc of every run will be available as an array with the data_calibMethod_metric key)
    # the _prob and _decision keys in the returned result is not meaningfull

    if ref_dict == {}:
        res_dict = new_dict.copy()
        for k in new_dict.keys():
            if "_prob" in k or "_decision" in k:
                del res_dict[k]
                continue
        return res_dict
    
    res_dict = ref_dict.copy()
    # print("new_dict keys", len(new_dict.keys()))
    # print("res_dict keys", len(res_dict.keys()))
    # print("---------------------------------")
    for k in ref_dict.keys():
        if "_prob" in k or "_decision" in k:
            del res_dict[k]
            continue
        res_dict[k] = ref_dict[k] + new_dict[k]

    return res_dict    



def mean_and_ranking_table(results_dict, metrics, calib_methods, data_list, save_results=False, mean_and_rank=True, std=False):
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

        if std:
            txt = "Data"
            for method in calib_methods:
                txt += "," + method + "_std"

            for data in data_list:
                txt += "\n"+ data
                for method in calib_methods:
                    txt += "," + str(np.array(results_dict[data+ "_" + method + "_"+ metric]).std())

            txt_data = StringIO(txt)
            df_std = pd.read_csv(txt_data, sep=",")
            df_std.set_index('Data', inplace=True)
            df_dict[metric + "_std"] = df_std
        if mean_and_rank:
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

def plot_probs(exp_data_name, probs, data, calib_methods, run_index, ref_plot_name="RF", hist_plot=False, calib_plot=False):
    
    for method in calib_methods:

        plt.plot([0, 1], [0, 1], linestyle='--')
        colors = ['black', 'red']
        colors_mean = ['orange', 'blue']
        plt.scatter(data["tp_test"], probs[f"{exp_data_name}_{method}_prob"][:,1], marker='.', c=[colors[c] for c in data["y_test"].astype(int)]) # Calibrated probs
        plt.scatter(data["tp_test"], probs[f"{exp_data_name}_{ref_plot_name}_prob"][:,1], marker='.', c=[colors[c] for c in data["y_test"].astype(int)], alpha=0.1) # faded RF probs

        # plt.scatter(data["tp_train"], probs[f"{exp_data_name}_{ref_plot_name}_prob_train"][:,1], marker='.', c=[colors[c] for c in data["y_train"].astype(int)]) # RF train probs 


        # ################## Just to test vertical and horisantal averaging
        # bin_means, bin_edges, binnumber = binned_statistic(data["tp_test"], probs[f"{exp_data_name}_{ref_plot_name}_prob"][:,1], bins=100) # Mean of the calibrated probs
        # plt.scatter((bin_edges[:-1] + bin_edges[1:])/2, bin_means, label='binned statistic of data')
        # v_tce = mean_squared_error((bin_edges[:-1] + bin_edges[1:])/2, bin_means)

        # bin_means, bin_edges, binnumber = binned_statistic(probs[f"{exp_data_name}_{method}_prob"][:,1], data["tp_test"], bins=100) # Horizantal Mean of the calibrated probs
        # plt.scatter((bin_edges[:-1] + bin_edges[1:])/2, bin_means, label='binned statistic of data')
        # h_tce = mean_squared_error((bin_edges[:-1] + bin_edges[1:])/2, bin_means)
        # ##################
        
        calib_tce = mean_squared_error(data["tp_test"], probs[f"{exp_data_name}_{method}_prob"][:,1]) # calculate TCE to add to the calib method plot
        
        if (method == "ISO" or method == "CRF" or method == "Line" or method == "Platt" or method =="Beta" or method =="VA") and calib_plot:
            plt.plot(tvec, probs[f"{exp_data_name}_{method}_fit"], c="blue")
        plt.xlabel(f"True probability")
        plt.ylabel("Predicted probability")

        # Add legend
        red_patch = plt.plot([],[], marker='o', markersize=10, color='red', linestyle='')[0]
        black_patch = plt.plot([],[], marker='o', markersize=10, color='black', linestyle='')[0]
        calib_patch = plt.plot([],[], marker='_', markersize=15, color='blue', linestyle='')[0]
        plt.legend((red_patch, black_patch, calib_patch), ('Class 0', 'Class 1', method + f" (TCE {calib_tce:0.5f})"))
        path = f"../../results/Synthetic/plots/{run_index}/{method}"
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f"{path}/{method}_{exp_data_name}.png")
        plt.close()

        if hist_plot:
            plt.hist(probs[f"{exp_data_name}_{method}_prob"][:,1], bins=50)
            plt.xlabel(f"probability output of {method}")
            plt.savefig(f"{path}/{method}_{exp_data_name}_hist.png")
            plt.close()

