
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
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration
from betacal import BetaCalibration
from sklearn.linear_model import LinearRegression

from CalibrationM import confidance_ECE, convert_prob_2D
from sklearn.metrics import brier_score_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

# ### Parameters

# runs = 1
# n_estimators=100

# plot_bins = 10
# test_size = 0.3


# oob = False

# plot = True
# save_results = False

# results_dict = {}

# samples = 3000
# features = 40
# calib_methods = ["RF", "Platt" , "ISO", "Rank", "CRF", "prank", "Venn", "VA", "Beta", "Elkan", "tlr"]
# # calib_methods = ["RF", "tlr"]
# metrics = ["acc", "auc", "brier", "ece", "tce"]

# run_name = "Samples"

tvec = np.linspace(0.01, 0.99, 990)
calib_methods = ["RF", "Platt" , "ISO", "Rank", "CRF", "VA", "Beta", "Elkan", "tlr", "Line"]
metrics = ["acc", "auc", "brier", "logloss", "ece", "tce"]


def split_train_calib_test(name, X, y, test_size, calib_size, orig_seed=0, tp=np.zeros(10)):
    ### spliting data to train calib and test
    for i in range(1000, 1100): # the for loop is to make sure the calib train and test split all consist of both classes of the binary dataset
        seed = i + orig_seed
        x_train_calib, x_test, y_train_calib, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)
        x_train, x_calib, y_train, y_calib = train_test_split(x_train_calib, y_train_calib, test_size=calib_size, shuffle=True, random_state=seed)
        if not tp.all() == 0: 
            _, _, tp_train_calib, tp_test = train_test_split(X, tp, test_size=test_size, shuffle=True, random_state=seed)
            _, _, tp_train, tp_calib = train_test_split(x_train_calib, tp_train_calib, test_size=0.5, shuffle=True, random_state=seed)

        if tp.all() == 0: 
            data = {"name":name, "x_train": x_train, "x_calib":x_calib, "x_test":x_test, "y_train":y_train, "y_calib":y_calib, "y_test":y_test}

        else:
            data = {"name":name, "x_train": x_train, "x_calib":x_calib, "x_test":x_test, "y_train":y_train, "y_calib":y_calib, "y_test":y_test, "tp_train":tp_train, "tp_calib":tp_calib, "tp_test":tp_test}
        
        if len(np.unique(data["y_calib"])) > 1 and len(np.unique(data["y_test"])) > 1 and len(np.unique(data["y_train"])) > 1:
            break

    return data

def calibration(RF, data, calib_methods, metrics, plot_bins = 10, laplace=1):

    # the retuen is a dict with all the metrics results as well as RF probs and every calibration method decision for every test data point
    # the structure of the keys in the dict is data_calibMethod_metric
    results_dict = {}
    for metric in metrics:
        for method in calib_methods:
            results_dict[data["name"] + "_" + method + "_" + metric] = []

    # random forest probs
    rf_p_calib = RF.predict_proba(data["x_calib"], laplace=laplace)
    rf_p_test = RF.predict_proba(data["x_test"], laplace=laplace)
    results_dict[data["name"] + "_RF_prob"] = rf_p_test
    results_dict[data["name"] + "_RF_decision"] = np.argmax(rf_p_test,axis=1)

    # all input probs to get the fit calib model

    # Platt scaling on RF
    if "Line" in calib_methods:
        lr_calib = LinearRegression().fit(rf_p_calib, data["y_calib"])
        y_pred_clipped = np.clip(lr_calib.predict(rf_p_test), 0, 1)
        lr_p_test = convert_prob_2D(y_pred_clipped)
        results_dict[data["name"] + "_Line_prob"] = lr_p_test
        results_dict[data["name"] + "_Line_decision"] = np.argmax(lr_p_test,axis=1)
        results_dict[data["name"] + "_Line_fit"] = np.clip(lr_calib.predict(convert_prob_2D(tvec)), 0, 1)
    if "Platt" in calib_methods:
        plat_calib = _SigmoidCalibration().fit(rf_p_calib[:,1], data["y_calib"])
        plat_p_test = convert_prob_2D(plat_calib.predict(rf_p_test[:,1]))
        results_dict[data["name"] + "_Platt_prob"] = plat_p_test
        results_dict[data["name"] + "_Platt_decision"] = np.argmax(plat_p_test,axis=1)
        results_dict[data["name"] + "_Platt_fit"] = plat_calib.predict(tvec)

    # ISO calibration on RF
    if "ISO" in calib_methods:
        iso_calib = IsotonicRegression(out_of_bounds='clip').fit(rf_p_calib[:,1], data["y_calib"])
        iso_p_test = convert_prob_2D(iso_calib.predict(rf_p_test[:,1]))
        results_dict[data["name"] + "_ISO_prob"] = iso_p_test
        results_dict[data["name"] + "_ISO_decision"] = np.argmax(iso_p_test,axis=1)
        results_dict[data["name"] + "_ISO_fit"] = iso_calib.predict(tvec)

    # RF ranking + ISO
    if "Rank" in calib_methods:
        x_calib_rank = RF.rank(data["x_calib"], class_to_rank=1, train_rank=True)
        x_test_rank = RF.rank_refrence(data["x_test"], class_to_rank=1)

        iso_rank = IsotonicRegression(out_of_bounds='clip').fit(x_calib_rank, data["y_calib"]) 
        rank_p_test = convert_prob_2D(iso_rank.predict(x_test_rank))
        results_dict[data["name"] + "_Rank_prob"] = rank_p_test
        results_dict[data["name"] + "_Rank_decision"] = np.argmax(rank_p_test,axis=1)
        # tvec_rank = RF.rank_refrence(data["x_test"], class_to_rank=1)
        # results_dict[data["name"] + "_Rank_fit"] = iso_rank.predict(tvec_rank)

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
        results_dict[data["name"] + "_CRF_fit"] = crf_calib.predict(tvec)[:,1]

    # Venn calibrator
    if "Venn" in calib_methods:
        ven_calib = Venn_calib().fit(rf_p_calib, data["y_calib"])
        ven_p_test = ven_calib.predict(rf_p_test)
        results_dict[data["name"] + "_Venn_prob"] = ven_p_test
        results_dict[data["name"] + "_Venn_decision"] = np.argmax(ven_p_test,axis=1)
        results_dict[data["name"] + "_Venn_fit"] = ven_calib.predict(convert_prob_2D(tvec))[:,1]

    # Venn abers
    if "VA" in calib_methods:
        VA = VA_calib().fit(rf_p_calib[:,1], data["y_calib"])
        va_p_test = convert_prob_2D(VA.predict(rf_p_test[:,1]))
        results_dict[data["name"] + "_VA_prob"] = va_p_test
        results_dict[data["name"] + "_VA_decision"] = np.argmax(va_p_test,axis=1)
        results_dict[data["name"] + "_VA_fit"] = VA.predict(tvec)

    # Beta calibration
    if "Beta" in calib_methods:
        beta_calib = BetaCalibration(parameters="abm").fit(rf_p_calib[:,1], data["y_calib"])
        beta_p_test = convert_prob_2D(beta_calib.predict(rf_p_test[:,1]))
        results_dict[data["name"] + "_Beta_prob"] = beta_p_test
        results_dict[data["name"] + "_Beta_decision"] = np.argmax(beta_p_test,axis=1)
        results_dict[data["name"] + "_Beta_fit"] = beta_calib.predict(tvec)

    # Elkan calibration
    if "Elkan" in calib_methods:
        elkan_calib = Elkan_calib().fit(data["y_train"], data["y_calib"])
        elkan_p_test = elkan_calib.predict(rf_p_test[:,1])
        results_dict[data["name"] + "_Elkan_prob"] = elkan_p_test
        results_dict[data["name"] + "_Elkan_decision"] = np.argmax(elkan_p_test,axis=1)
        results_dict[data["name"] + "_Elkan_fit"] = elkan_calib.predict(tvec)[:,1]

    # tree LR calib
    if "tlr" in calib_methods:
        tlr_calib = treeLR_calib().fit(RF, data["x_train"] ,data["y_train"], data["x_calib"], data["y_calib"])
        tlr_p_test = tlr_calib.predict(data["x_test"])
        results_dict[data["name"] + "_tlr_prob"] = tlr_p_test
        results_dict[data["name"] + "_tlr_decision"] = np.argmax(tlr_p_test,axis=1)
        # results_dict[data["name"] + "_tlr_fit"] = tlr_calib.predict(convert_prob_2D(tvec))[:,1]



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

def plot_probs(exp_data_name, probs, data, calib_methods, run_index, hist_plot=False, calib_plot=False):
    for method in calib_methods:

        plt.plot([0, 1], [0, 1], linestyle='--')
        colors = ['black', 'red']
        plt.scatter(data["tp_test"], probs[f"{exp_data_name}_{method}_prob"][:,1], marker='.', c=[colors[c] for c in data["y_test"].astype(int)])
        plt.scatter(data["tp_test"], probs[f"{exp_data_name}_RF_prob"][:,1], marker='.', c=[colors[c] for c in data["y_test"].astype(int)], alpha=0.1)
        if method != "RF" and method != "Rank" and method != "prank" and method != "tlr" and calib_plot:
            plt.plot(tvec, probs[f"{exp_data_name}_{method}_fit"], c="blue")
        plt.xlabel("True probability")
        plt.ylabel("Predicted probability")

        # Add legend
        red_patch = plt.plot([],[], marker='o', markersize=10, color='red', linestyle='')[0]
        black_patch = plt.plot([],[], marker='o', markersize=10, color='black', linestyle='')[0]
        calib_patch = plt.plot([],[], marker='_', markersize=15, color='blue', linestyle='')[0]
        plt.legend((red_patch, black_patch, calib_patch), ('Class 0', 'Class 1', method))
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

        # fig, ax = plt.subplots()
        # colors = ['black', 'red']
        # scatter = ax.scatter(data["tp_test"], probs[f"{exp_data_name}_{method}_prob"][:,1], c=[colors[c] for c in data["y_test"].astype(int)])
        # # produce a legend with the unique colors from the scatter
        # legend1 = ax.legend('Class 0', 'Class 1',loc="upper left", title="Classes")
        # ax.plot([0, 1], [0, 1], linestyle='--')
        # if method != "RF" and method != "Rank" and method != "prank" and method != "tlr":
        #     ax.plot(np.linspace(0.01, 0.99, 99), probs[f"{exp_data_name}_{method}_fit"], label=f"{method}")
        # ax.add_artist(legend1)
        # plt.xlabel("True probability")
        # plt.ylabel("Predicted probability")
        # path = f"../../results/Synthetic/plots/{run_index}/{method}"
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # plt.savefig(f"{path}/{method}_{exp_data_name}.png")
        # plt.close()


