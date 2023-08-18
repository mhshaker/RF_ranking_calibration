import Experiments.core_exp as exp
import Experiments.core_calib as cal
from joblib import Parallel, delayed
import numpy as np

def test_run_exp(): # test to see if run_exp parallel processing works


    params = {
        "seed": 0,
        "runs": 3,
        "path": ".",
        "exp_name": "split_test",
        "split": "CV", #CV و random_split
        "cv_folds": 3,
        "plot_data": False,
        "plot": False,
        "data_name": "parkinsons",

        "calib_methods": ["RF", "DT"], #, "RF_ens_k", "RF_ens_r", "Platt", "ISO", "Beta", "CRF", "VA"],
        "metrics": ["acc", "ece", "brier"], 

        # calib param
        "bin_strategy": "uniform",
        "ece_bins": 20,
        "boot_size": 1000,
        "boot_count": 5,


        # RF hyper opt
        "hyper_opt": "Manual", #"Default", "Manual"
        "opt_cv":5, 
        "opt_n_iter":10,
        "opt_top_K": 5,
        "search_space": {
                        "n_estimators": [100],
                        "max_depth": [2,3,4,5,6,7,8,10,15,20,30,40,50,60,100],
                        "criterion": ["gini", "entropy"],
                        "max_features": ["sqrt", "log2"],
                        "min_samples_split": [2,3,4,5],
                        "min_samples_leaf": [1,2,3],
                        "oob_score": [True]
                        },
        "oob": True,
        "n_estimators": 100,

    }

    exp_key = "depth"
    exp_values = [2,3,4]

    exp_res, data_list = exp.run_exp(exp_key, exp_values, params)

    exp_keys = list(exp_res.keys())
    expected_keys = ['2_RF_acc', '2_DT_acc', '2_RF_ece', '2_DT_ece', '2_RF_brier', '2_DT_brier', '3_RF_acc', '3_DT_acc', '3_RF_ece', '3_DT_ece', '3_RF_brier', '3_DT_brier', '4_RF_acc', '4_DT_acc', '4_RF_ece', '4_DT_ece', '4_RF_brier', '4_DT_brier']
    exp_keys.sort()
    expected_keys.sort()

    assert exp_keys == expected_keys
    # assert len(exp_res["2_RF_prob"]) == 9 # this test is removed because when plot is false, there is no _prob in the dict to save memory
    assert len(exp_res["2_RF_ece"]) == 9 # if plot set to true then it will be 1
    assert len(exp_res["2_RF_acc"]) == 9
    assert len(exp_res["2_RF_brier"]) == 9
    assert data_list == ['2', '3', '4']


def test_multi_run_avg(): # test to see if the avg of multi splits of a data are correct
    params = {
        "seed": 0,
        "runs": 3,
        "exp_name": "calib_avg",
        "split": "CV", #CV و random_split
        "cv_folds": 5,
        "plot": False,

        "calib_methods": ["RF", "Beta", "CRF", "VA"],
        "metrics": ["acc", "ece", "brier"], 

        "data_name": "synthetic",
        "data_size": 1000,
        "n_features": 2,
        "plot_data" : False,

        "class1_mean_min":0, 
        "class1_mean_max":1,
        "class2_mean_min":1, 
        "class2_mean_max":3, 

        # calib param
        "bin_strategy": "uniform",
        "ece_bins": 20,
        "boot_size": 1000,
        "boot_count": 5,

        # RF hyper opt
        "hyper_opt": True, #"Default", "Manual"
        "opt_cv":5, 
        "opt_n_iter":1,
        "opt_top_K": 5,
        "search_space": {"n_estimators": [100], "oob_score": [False]},
        "oob": False,
    }

    data_runs = exp.load_data_runs(params, "expname")

    # first run with normal data
    res_list = Parallel(n_jobs=-1)(delayed(cal.calibration)(data, params) for data, params in zip(data_runs, np.repeat(params, len(data_runs))))
    res_runs = {} # results for each data set will be saved in here.
    for res in res_list: # res_runs is a dict of all the metrics which are a list of results of multiple runs 
        res_runs = cal.update_runs(res_runs, res) # calib results for every run for the same dataset is aggregated in res_runs (ex. acc of every run as an array)

    # second run with y_test labels switched

    for data in data_runs:
        data["y_test"] = 1 - data["y_test"]
    # data_runs[0]["y_test"] = np.zeros(len(data_runs[0]["y_test"]))

    res_list2 = Parallel(n_jobs=-1)(delayed(cal.calibration)(data, params) for data, params in zip(data_runs, np.repeat(params, len(data_runs))))
    res_runs2 = {} # results for each data set will be saved in here.
    for res in res_list2: # res_runs is a dict of all the metrics which are a list of results of multiple runs 
        res_runs2 = cal.update_runs(res_runs2, res) # calib results for every run for the same dataset is aggregated in res_runs (ex. acc of every run as an array)

    sum_all_RF = np.array(res_runs["expname_RF_acc"]) + np.array(res_runs2["expname_RF_acc"])
    sum_all_Beta = np.array(res_runs["expname_Beta_acc"]) + np.array(res_runs2["expname_Beta_acc"])
    sum_all_CRF = np.array(res_runs["expname_CRF_acc"]) + np.array(res_runs2["expname_CRF_acc"])
    sum_all_VA = np.array(res_runs["expname_VA_acc"]) + np.array(res_runs2["expname_VA_acc"])

    assert np.all(sum_all_RF == 1)
    assert np.all(sum_all_Beta == 1)
    assert np.all(sum_all_CRF == 1)
    assert np.all(sum_all_VA == 1)

# test_multi_run_avg()

