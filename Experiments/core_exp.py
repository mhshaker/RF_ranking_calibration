# imports
import sys
import os
sys.path.append('../../') # to access the files in higher directories
sys.path.append('../') # to access the files in higher directories
import Data.data_provider as dp
import core_calib as cal
from estimators.IR_RF_estimator import IR_RF
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


np.random.seed(0)
    
def run_exp(exp_key, exp_values, params):

    exp_res = {}
    data_list = []
    
    for exp_param in exp_values: 
        params[exp_key] = exp_param
        # Data
        exp_data_name = str(exp_param) # data_name + "_" + 
        data_list.append(exp_data_name)
        res_runs = {} # results for each data set will be saved in here.
        data_runs = []
        for seed in range(params["runs"]):
            params["seed"] = seed

            if params["data_name"] == "synthetic":
                # X, y, tp = dp.make_classification_gaussian_with_true_prob(params["data_size"], 
                #                                                         params["n_features"], 
                #                                                         class1_mean_min = params["class1_mean_min"], 
                #                                                         class1_mean_max = params["class1_mean_max"],
                #                                                         class2_mean_min = params["class2_mean_min"], 
                #                                                         class2_mean_max = params["class2_mean_max"], 
                #                                                         seed=seed)
                
                X_temp, _ = make_classification(n_samples=params["data_size"], 
                            n_features= params["n_features"], 
                            n_informative= params["n_informative"], 
                            n_redundant= params["n_redundant"], 
                            n_repeated= params["n_repeated"], 
                            n_classes=2, 
                            n_clusters_per_class=2, 
                            weights=None, 
                            flip_y=0.05, 
                            class_sep=1.0, 
                            hypercube=True, 
                            shift=0.0, 
                            scale=1.0, 
                            shuffle=True, 
                            random_state=seed)
                X, y, tp = dp.x_y_q(X_temp, n_copy = params["n_copy"], seed = params["seed"])

                if params["plot_data"]:
                    colors = ['black', 'red']
                    plt.scatter(X[:,0], X[:,1], marker='.', c=[colors[c] for c in y.astype(int)]) # Calibrated probs
                    red_patch = plt.plot([],[], marker='o', markersize=10, color='red', linestyle='')[0]
                    black_patch = plt.plot([],[], marker='o', markersize=10, color='black', linestyle='')[0]
                    plt.legend((red_patch, black_patch), ('Class 1', 'Class 0'), loc='upper left')
                    plt.xlabel("X_0")
                    plt.ylabel("X_1")

                    path = f"./results/{params['exp_name']}"
                    if not os.path.exists(path):
                        os.makedirs(path)
                    plt.savefig(f"{path}/data.pdf", format='pdf', transparent=True)
                    plt.close()

                data_folds = cal.CV_split_train_calib_test(exp_data_name, X,y,params["cv_folds"],seed,tp)
            else:
                X, y = dp.load_data(params["data_name"], "../../")
                data_folds = cal.CV_split_train_calib_test(exp_data_name, X,y,params["cv_folds"],seed)
            for data in data_folds:    
                data_runs.append(data)
            # for data in data_folds: # running the same dataset multiple times
            res_list = Parallel(n_jobs=-1)(delayed(cal.calibration)(data, params) for data, params in zip(data_folds, np.repeat(params, params["cv_folds"])))
            
            for res in res_list:
                res_runs = cal.update_runs(res_runs, res) # calib results for every run for the same dataset is aggregated in res_runs (ex. acc of every run as an array)

        if params["plot"]: # and params["data_name"] == "synthetic":
            cal.plot_probs(exp_data_name, res_runs, data_runs, params, "RF", False, True) 
        
        if params["plot"] and params["data_name"] != "synthetic":
            tmp = params["data_name"]
            params["data_name"] = tmp + "ece"
            cal.plot_probs(exp_data_name, res_runs, data_runs, params, "RF", False, True) 
            params["data_name"] = tmp
            
        exp_res.update(res_runs) # merge results of all datasets together
        # print("exp_res ", exp_res)
        # print("---------------------------------")
        print(f"exp_param {exp_param} done")


    return exp_res, data_list
        
