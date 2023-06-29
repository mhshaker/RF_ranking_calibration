# imports
import sys
sys.path.append('../../') # to access the files in higher directories
sys.path.append('../') # to access the files in higher directories
import Data.data_provider as dp
import core_calib as cal
from estimators.IR_RF_estimator import IR_RF
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier

np.random.seed(0)


# params
params_all = {
    # exp
    "runs": 1,
    "plot": True,
    "calib_methods": ["RF", "Platt", "ISO", "Rank", "CRF", "VA", "Beta", "Elkan", "tlr", "Line", "RF_boot", "RF_ens_r", "RF_large", "RF_ens_line"],
    "metrics": ["acc", "tce", "logloss", "brier", "ece", "auc"],
    
    #data
    "data_name": "synthetic",
    "data_size": 1000,
    "n_features": 2,

    "class1_mean_min":0, 
    "class1_mean_max":1,
    "class2_mean_min":1, 
    "class2_mean_max":3, 

    "cv_folds": 10,
    "test_split": 0.3,
    "calib_split": 0.1,


    # calib param
    "ece_bins": 20,
    "boot_size": 5000,
    "boot_count": 20,

    # RF hyper opt
    "hyper_opt": True,
    "opt_cv":5, 
    "opt_n_iter":10,
    "search_space": {
                    "n_estimators": [10],
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

def train_calib(data, params, seed, check_dummy=False):
    # train model - hyper opt
    if params["hyper_opt"]:
        rf = IR_RF(random_state=seed)
        RS = RandomizedSearchCV(rf, params["search_space"], scoring=["accuracy"], refit="accuracy", cv=params["opt_cv"], n_iter=params["opt_n_iter"], random_state=seed)
        RS.fit(data["x_train"], data["y_train"])
        rf_best = RS.best_estimator_
    else:
        rf_best = IR_RF(n_estimators=params["n_estimators"], max_depth=params["depth"], random_state=seed)
        rf_best.fit(data["x_train"], data["y_train"])

    if check_dummy:
        dummy_clf = DummyClassifier(strategy="most_frequent").fit(data["x_train"], data["y_train"])
        d_s = dummy_clf.score(data["x_test"], data["y_test"])
        rf_best_s = rf_best.score(data["x_test"], data["y_test"])
        l_dif = (rf_best_s - d_s ) * 100
        # print(f"data {data['name']} learn diff {rf_best_s - d_s}")
        if l_dif <= 1:
            print(f">>>>>>> data {data['name']} NOT LEARNING - learnign diff is {l_dif}")


    # calibration
    return cal.calibration(rf_best, data, params) # res is a dict with all the metrics results as well as RF probs and every calibration method decision for every test data point

    
def run_exp(exp_key, exp_values, params):

    exp_res = {}
    data_list = []

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
                                                                    seed=params["seed"])
            data_folds = cal.CV_split_train_calib_test(exp_data_name, X,y,params["cv_folds"],params["seed"],tp)
        else:
            X, y = dp.load_data(params["data_name"], "../../")
            data_folds = cal.CV_split_train_calib_test(exp_data_name, X,y,params["cv_folds"],params["seed"])

        # for data in data_folds: # running the same dataset multiple times
        res_list = Parallel(n_jobs=-1)(delayed(train_calib)(data, params, seed) for data, params, seed in zip(data_folds, np.repeat(params, params["cv_folds"]), np.repeat(params["seed"], params["cv_folds"])))
        
        res_runs = {} # results for each data set will be saved in here.
        for res in res_list:
            res_runs = cal.update_runs(res_runs, res) # calib results for every run for the same dataset is aggregated in res_runs (ex. acc of every run as an array)

        if params["plot"]: # and params["data_name"] == "synthetic":
            cal.plot_probs(exp_data_name, res_runs, data_folds, params, "RF", False, True) 
        
        if params["plot"] and params["data_name"] == "synthetic":
            tmp = params["data_name"]
            params["data_name"] = tmp + "ece"
            cal.plot_probs(exp_data_name, res_runs, data_folds, params, "RF", False, True) 
            params["data_name"] = tmp
        
        exp_res.update(res_runs) # merge results of all datasets together

    return exp_res, data_list
        
