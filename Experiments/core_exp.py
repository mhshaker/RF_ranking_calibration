# imports
import sys
sys.path.append('../../') # to access the files in higher directories
sys.path.append('../') # to access the files in higher directories
import Data.data_provider as dp
import core_calib as cal
from estimators.IR_RF_estimator import IR_RF
from sklearn.model_selection import RandomizedSearchCV


# params
params_all = {
    # exp
    "runs": 1,
    "plot": True,
    "calib_methods": ["RF", "Platt", "ISO", "Rank", "CRF", "VA", "Beta", "Elkan", "tlr", "Line", "RF_ens_r", "RF_large", "RF_boot"],
    "metrics": ["acc", "tce", "logloss", "brier", "ece", "auc"],
    
    #data
    "data_name": "synthetic",
    "data_size": 1000,
    "n_features": 2,

    "class1_mean_min":0, 
    "class1_mean_max":1,
    "class2_mean_min":1, 
    "class2_mean_max":3, 

    "test_split": 0.3,
    "calib_split": 0.1,


    # calib param
    "ece_bins": 20,
    "boot_size": 50,
    "boot_count": 100,

    # RF hyper opt
    "hyper_opt": True,
    "opt_cv":5, 
    "opt_n_iter":10,
    "search_space": {
                    "n_estimators": [50],
                    "max_depth": [2,3,4,5,6,7,8,10,20,50,100],
                    "criterion": ["gini", "entropy"],
                    # "min_samples_split": [2,3,4,5],
                    # "min_samples_leaf": [1,2,3],
                    },
    # RF
    "depth": 4,
    "n_estimators": 10,
    "oob": False,

}


def run_exp(exp_key, exp_values, params):

    data_list = []
    calib_results_dict = {}

    for exp_param in exp_values: 
        params[exp_key] = exp_param
        # Data
        exp_data_name = str(exp_param) # data_name + "_" + 
        data_list.append(exp_data_name)
        if params["data_name"] == "synthetic":
            X, y, tp = dp.make_classification_gaussian_with_true_prob(params["data_size"], 
                                                                    params["n_features"], 
                                                                    class1_mean_min = params["class1_mean_min"], 
                                                                    class1_mean_max = params["class1_mean_max"],
                                                                    class2_mean_min = params["class2_mean_min"], 
                                                                    class2_mean_max = params["class2_mean_max"], 
                                                                    seed=0)
        else:
            X, y = dp.load_data(params["data_name"], "../../")

        data_dict = {} # results for each data set will be saved in here.
        for seed in range(params["runs"]): # running the same dataset multiple times
            # split the data
            if params["data_name"] == "synthetic":
                data = cal.split_train_calib_test(exp_data_name, X, y, params["test_split"], params["calib_split"], seed, tp)
            else:
                data = cal.split_train_calib_test(exp_data_name, X, y, params["test_split"], params["calib_split"], seed)

            # train model - hyper opt
            if params["hyper_opt"]:
                rf = IR_RF(random_state=seed)
                RS = RandomizedSearchCV(rf, params["search_space"], scoring=["accuracy"], refit="accuracy", cv=params["opt_cv"], n_iter=params["opt_n_iter"], random_state=seed)
                RS.fit(data["x_train"], data["y_train"])
                rf_best = RS.best_estimator_
                # print("hyper_opt: best params", RS.best_params_)
            else:
                rf_best = IR_RF(n_estimators=params["n_estimators"], oob_score=params["oob"], max_depth=params["depth"], random_state=seed)
                rf_best.fit(data["x_train"], data["y_train"])

                if params["exp_name"] == "trees":
                    print("trees params", rf_best.get_params())


            # calibration
            res = cal.calibration(rf_best, data, params) # res is a dict with all the metrics results as well as RF probs and every calibration method decision for every test data point
            data_dict = cal.update_runs(data_dict, res) # calib results for every run for the same dataset is aggregated in data_dict (ex. acc of every run as an array)
            if params["plot"] and params["data_name"] == "synthetic":
                cal.plot_probs(exp_data_name, res, data, params, seed, "RF", False, True) 

        calib_results_dict.update(data_dict) # merge results of all datasets together
    return calib_results_dict, data_list
        
