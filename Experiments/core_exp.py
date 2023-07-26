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


def train_calib(data, params, seed, check_dummy=False):
    # train model - hyper opt
    rf = IR_RF(random_state=seed)
    RS = RandomizedSearchCV(rf, params["search_space"], scoring=["neg_brier_score"], refit="neg_brier_score", cv=params["opt_cv"], n_iter=params["opt_n_iter"], random_state=seed)
    RS.fit(data["x_train"], data["y_train"])
    rf_best = RS.best_estimator_

    opt_params = np.array(RS.cv_results_['params'])
    opt_rankings = np.array(RS.cv_results_['rank_test_neg_brier_score'])

    sorted_indices = np.argsort(opt_rankings, kind="stable")
    opt_params = opt_params[sorted_indices]
    params["opt_top_K"] = opt_params[:params["opt_top_K"]] # save the top K best performing RF params in opt_top_K
    # print("opt_top_K", params["opt_top_K"])

    if params["hyper_opt"] == "Default":
        rf_best = IR_RF(random_state=seed)
        rf_best.fit(data["x_train"], data["y_train"])
    elif params["hyper_opt"] == "Manual":
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
                            n_features=params["n_features"], 
                            n_informative=2, 
                            n_redundant=0, 
                            n_repeated=0, 
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
                X, y, tp = dp.x_y_q(X_temp, n_copy=params["n_copy"])

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
            res_list = Parallel(n_jobs=-1)(delayed(train_calib)(data, params, seed) for data, params, seed in zip(data_folds, np.repeat(params, params["cv_folds"]), np.repeat(seed, params["cv_folds"])))
            
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

    return exp_res, data_list
        
