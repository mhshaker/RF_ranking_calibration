import Experiments.core_calib as cal
import Experiments.core_exp as exp


def test_run_exp(): # test to see if run_exp parallel processing works


    params = {
        "seed": 0,
        "runs": 3,
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
    expected_keys = ['2_RF_prob', '2_DT_prob', '2_RF_acc', '2_DT_acc', '2_RF_ece', '2_DT_ece', '2_RF_brier', '2_DT_brier', '3_RF_prob', '3_DT_prob', '3_RF_acc', '3_DT_acc', '3_RF_ece', '3_DT_ece', '3_RF_brier', '3_DT_brier', '4_RF_prob', '4_DT_prob', '4_RF_acc', '4_DT_acc', '4_RF_ece', '4_DT_ece', '4_RF_brier', '4_DT_brier']
    exp_keys.sort()
    expected_keys.sort()

    assert exp_keys == expected_keys
    assert len(exp_res["2_RF_prob"]) == 9
    assert len(exp_res["2_RF_ece"]) == 9 # if plot set to true then it will be 1
    assert len(exp_res["2_RF_acc"]) == 9
    assert len(exp_res["2_RF_brier"]) == 9
    assert data_list == ['2', '3', '4']


# test_run_exp()