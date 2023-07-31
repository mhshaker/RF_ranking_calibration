
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
from scipy.stats import binned_statistic
from CalibrationM import confidance_ECE, convert_prob_2D
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

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

def CV_split_train_calib_test(name, X, y, folds=10, seed=0, tp=np.full(10,-1)):
    
    data_folds = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    for train_calib_index, test_index in skf.split(X, y):
        data = {"name": name }
        data["X"] = X
        data["y"] = y
        data["x_train_calib"], data["x_test"] = X[train_calib_index], X[test_index]
        data["y_train_calib"], data["y_test"] = y[train_calib_index], y[test_index]
        if not tp.all() == -1:
            data["tp"] = tp
            data["tp_train_calib"], data["tp_test"] = tp[train_calib_index], tp[test_index]

        skf2 = StratifiedKFold(n_splits=folds-1, shuffle=True, random_state=seed)
        train_index, calib_index = next(skf2.split(data["x_train_calib"], data["y_train_calib"]))
        data["x_train"], data["x_calib"] = data["x_train_calib"][train_index], data["x_train_calib"][calib_index]
        data["y_train"], data["y_calib"] = data["y_train_calib"][train_index], data["y_train_calib"][calib_index]
        if not tp.all() == -1:
            data["tp_train"], data["tp_calib"] = data["tp_train_calib"][train_index], data["tp_train_calib"][calib_index]
        
        data_folds.append(data)

    return data_folds


def calibration(RF, data, params):
    data_name = data["name"]
    calib_methods = params["calib_methods"] 
    metrics = params["metrics"]
    # the retuen is a dict with all the metrics results as well as RF probs and every calibration method decision for every test data point
    # the structure of the keys in the dict is data_calibMethod_metric
    results_dict = {}

    # random forest probs
    rf_p_calib = RF.predict_proba(data["x_calib"])
    rf_p_test = RF.predict_proba(data["x_test"])
    results_dict[data["name"] + "_RF_prob"] = rf_p_test
    results_dict[data["name"] + "_RF_prob_train"] = RF.predict_proba(data["x_train"])
    results_dict[data["name"] + "_RF_cstar"] = RF.predict_proba(data["X"])
    results_dict[data["name"] + "_RF_prob_calib"] = rf_p_calib
    results_dict[data["name"] + "_RF_decision"] = np.argmax(rf_p_test,axis=1)

    bc = Boot_calib(boot_count=params["boot_count"])
    RF_ens_p_test = bc.predict_ens(data["x_test"], data["x_train"], data["y_train"], RF)
    RF_ens_p_calib = bc.predict_ens(data["x_calib"], data["x_train"], data["y_train"], RF)
    RF_ens_p_c = bc.predict_ens(data["X"], data["x_train"], data["y_train"], RF)

    RF_ens_k_p_test = bc.predict_ens_params(data["x_test"], data["x_train"], data["y_train"], params["opt_top_K"], params["seed"])
    RF_ens_k_p_calib = bc.predict_ens_params(data["x_calib"], data["x_train"], data["y_train"], params["opt_top_K"], params["seed"])
    RF_ens_k_p_c = bc.predict_ens_params(data["X"], data["x_train"], data["y_train"], params["opt_top_K"], params["seed"])

    RF_large_p_test = bc.predict_largeRF(data["x_test"], data["x_train"], data["y_train"], RF)
    RF_large_p_calib = bc.predict_largeRF(data["x_calib"], data["x_train"], data["y_train"], RF)
    RF_large_p_c = bc.predict_largeRF(data["X"], data["x_train"], data["y_train"], RF)

    # all input probs to get the fit calib model

    ### models

    method = "DT"
    if method in calib_methods:
        dt = DecisionTreeClassifier(random_state=params["seed"], max_depth=params["depth"]).fit(data["x_train"], data["y_train"])
        dt_p_test = dt.predict_proba(data["x_test"])
        results_dict[f"{data_name}_{method}_prob"] = dt_p_test
        results_dict[f"{data_name}_{method}_cstar"] = dt.predict_proba(data["X"])

    method = "LR"
    if method in calib_methods:
        lr = LogisticRegression(random_state=params["seed"]).fit(data["x_train"], data["y_train"])
        lr_p_test = lr.predict_proba(data["x_test"])
        results_dict[f"{data_name}_{method}_prob"] = lr_p_test
        results_dict[f"{data_name}_{method}_cstar"] = lr.predict_proba(data["X"])

    method = "SVM"
    if method in calib_methods:
        svm = SVC(probability=True, random_state=params["seed"]).fit(data["x_train"], data["y_train"])
        svm_p_test = svm.predict_proba(data["x_test"])
        results_dict[f"{data_name}_{method}_prob"] = svm_p_test
        results_dict[f"{data_name}_{method}_cstar"] = svm.predict_proba(data["X"])

    ### full data (train + calib)
    method = "RF_d"
    if method in calib_methods:
        rf_d = IR_RF(n_estimators=params["n_estimators"], random_state=params["seed"]).fit(data["x_train_calib"], data["y_train_calib"])
        rf_d_p_test = rf_d.predict_proba(data["x_test"])
        rf_d_p_calib = rf_d.predict_proba(data["x_calib"])
        results_dict[f"{data_name}_{method}_prob"] = rf_d_p_test
        results_dict[f"{data_name}_{method}_cstar"] = rf_d.predict_proba(data["X"])

    method = "RF_opt"
    if method in calib_methods:
        # train model - hyper opt
        if params["hyper_opt"]:
            rf_f = IR_RF(random_state=params["seed"])
            RS_f = RandomizedSearchCV(rf_f, params["search_space"], scoring=["neg_brier_score"], refit="neg_brier_score", cv=params["opt_cv"], n_iter=params["opt_n_iter"], random_state=params["seed"])
            RS_f.fit(data["x_train_calib"], data["y_train_calib"])
            RF_f = RS_f.best_estimator_
        else:
            RF_f = IR_RF(n_estimators=params["n_estimators"], max_depth=params["depth"], random_state=params["seed"])
            RF_f.fit(data["x_train_calib"], data["y_train_calib"])

        rff_p_test = RF_f.predict_proba(data["x_test"])
        results_dict[f"{data_name}_{method}_prob"] = rff_p_test
        results_dict[f"{data_name}_{method}_cstar"] = RF_f.predict_proba(data["X"])
    
    method = "RF_ens_r"
    if method in calib_methods:
        RF_ens_p_test_fd = bc.predict_ens(data["x_test"], data["x_train_calib"], data["y_train_calib"], RF)
        results_dict[f"{data_name}_{method}_prob"] = RF_ens_p_test_fd
        results_dict[f"{data_name}_{method}_cstar"] = bc.predict_ens(data["X"], data["x_train_calib"], data["y_train_calib"], RF)

    method = "RF_ens_k"
    if method in calib_methods:
        RF_ens_k_p_test_fd = bc.predict_ens_params(data["x_test"], data["x_train_calib"], data["y_train_calib"], params["opt_top_K"], params["seed"])
        results_dict[f"{data_name}_{method}_prob"] = RF_ens_k_p_test_fd
        results_dict[f"{data_name}_{method}_cstar"] = bc.predict_ens_params(data["X"], data["x_train_calib"], data["y_train_calib"], params["opt_top_K"], params["seed"])

    method = "RF_large"
    if method in calib_methods:
        RF_large_p_test_fd = bc.predict_largeRF(data["x_test"], data["x_train_calib"], data["y_train_calib"], RF)
        results_dict[f"{data_name}_{method}_prob"] = RF_large_p_test_fd
        results_dict[f"{data_name}_{method}_cstar"] = bc.predict_largeRF(data["X"], data["x_train_calib"], data["y_train_calib"], RF)

    

    method = "Platt"
    if method in calib_methods:
        plat_calib = _SigmoidCalibration().fit(rf_p_calib[:,1], data["y_calib"])
        plat_p_test = convert_prob_2D(plat_calib.predict(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = plat_p_test
        results_dict[f"{data_name}_{method}_fit"] = plat_calib.predict(tvec)
        results_dict[f"{data_name}_{method}_cstar"] = convert_prob_2D(plat_calib.predict(results_dict[data["name"] + "_RF_cstar"][:,1]))

    # ISO calibration on RF
    method = "ISO"
    if method in calib_methods:
        iso_calib = IsotonicRegression(out_of_bounds='clip').fit(rf_p_calib[:,1], data["y_calib"])
        iso_p_test = convert_prob_2D(iso_calib.predict(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = iso_p_test
        results_dict[f"{data_name}_{method}_fit"] = iso_calib.predict(tvec)
        results_dict[f"{data_name}_{method}_cstar"] = convert_prob_2D(iso_calib.predict(results_dict[data["name"] + "_RF_cstar"][:,1]))

    # CRF calibrator
    method = "CRF"
    if method in calib_methods:
        crf_calib = CRF_calib(learning_method="sig_brior").fit(rf_p_calib[:,1], data["y_calib"])
        crf_p_test = crf_calib.predict(rf_p_test[:,1])
        results_dict[f"{data_name}_{method}_prob"] = crf_p_test
        results_dict[f"{data_name}_{method}_fit"] = crf_calib.predict(tvec)[:,1]
        results_dict[f"{data_name}_{method}_cstar"] = convert_prob_2D(crf_calib.predict(results_dict[data["name"] + "_RF_cstar"][:,1]))

    # Venn abers
    method = "VA"
    if method in calib_methods:
        VA = VA_calib().fit(rf_p_calib[:,1], data["y_calib"])
        va_p_test = convert_prob_2D(VA.predict(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = va_p_test
        results_dict[f"{data_name}_{method}_fit"] = VA.predict(tvec)
        results_dict[f"{data_name}_{method}_cstar"] = convert_prob_2D(VA.predict(results_dict[data["name"] + "_RF_cstar"][:,1]))


    # Beta calibration
    method = "Beta"
    if method in calib_methods:
        beta_calib = BetaCalibration(parameters="abm").fit(rf_p_calib[:,1], data["y_calib"])
        beta_p_test = convert_prob_2D(beta_calib.predict(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = beta_p_test
        results_dict[f"{data_name}_{method}_fit"] = beta_calib.predict(tvec)
        results_dict[f"{data_name}_{method}_cstar"] = convert_prob_2D(beta_calib.predict(results_dict[data["name"] + "_RF_cstar"][:,1]))

    # tree LR calib
    method = "tlr"
    if method in calib_methods:
        tlr_calib = treeLR_calib().fit(RF, data["x_train"] ,data["y_train"], data["x_calib"], data["y_calib"])
        tlr_p_test = tlr_calib.predict(data["x_test"])
        results_dict[f"{data_name}_{method}_prob"] = tlr_p_test
        # results_dict[data["name"] + "_tlr_fit"] = tlr_calib.predict(convert_prob_2D(tvec))[:,1]
        results_dict[f"{data_name}_{method}_cstar"] = tlr_calib.predict(data["X"])

    ### RF_ens
    method = "RF_ens_line"
    if method in calib_methods:
        lr_calib = LinearRegression().fit(RF_ens_p_calib, data["y_calib"])
        y_pred_clipped = np.clip(lr_calib.predict(RF_ens_p_test), 0, 1)
        ebl_p_test = convert_prob_2D(y_pred_clipped)
        results_dict[f"{data_name}_{method}_prob"] = ebl_p_test

    method = "RF_ens_CRF"
    if method in calib_methods:
        crf_calib = CRF_calib(learning_method="sig_brior").fit(RF_ens_p_calib[:,1], data["y_calib"])
        ebl_p_test = crf_calib.predict(RF_ens_p_test[:,1])
        results_dict[f"{data_name}_{method}_prob"] = ebl_p_test

    method = "RF_ens_Platt"
    if method in calib_methods:
        plat_calib = _SigmoidCalibration().fit(RF_ens_p_calib[:,1], data["y_calib"])
        ebl_p_test = convert_prob_2D(plat_calib.predict(RF_ens_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = ebl_p_test

    method = "RF_ens_ISO"
    if method in calib_methods:
        iso_calib = IsotonicRegression(out_of_bounds='clip').fit(RF_ens_p_calib[:,1], data["y_calib"])
        ebl_p_test = convert_prob_2D(iso_calib.predict(RF_ens_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = ebl_p_test

    method = "RF_ens_Beta"
    if method in calib_methods:
        beta_calib = BetaCalibration(parameters="abm").fit(RF_ens_p_calib[:,1], data["y_calib"])
        ebl_p_test = convert_prob_2D(beta_calib.predict(RF_ens_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = ebl_p_test

    ### RF_large
    method = "RF_large_line"
    if method in calib_methods:
        lr_calib = LinearRegression().fit(RF_large_p_calib, data["y_calib"])
        y_pred_clipped = np.clip(lr_calib.predict(RF_large_p_test), 0, 1)
        ebl_p_test = convert_prob_2D(y_pred_clipped)
        results_dict[f"{data_name}_{method}_prob"] = ebl_p_test

    method = "RF_large_CRF"
    if method in calib_methods:
        crf_calib = CRF_calib(learning_method="sig_brior").fit(RF_large_p_calib[:,1], data["y_calib"])
        ebl_p_test = crf_calib.predict(RF_large_p_test[:,1])
        results_dict[f"{data_name}_{method}_prob"] = ebl_p_test

    method = "RF_large_Platt"
    if method in calib_methods:
        plat_calib = _SigmoidCalibration().fit(RF_large_p_calib[:,1], data["y_calib"])
        ebl_p_test = convert_prob_2D(plat_calib.predict(RF_large_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = ebl_p_test

    method = "RF_large_ISO"
    if method in calib_methods:
        iso_calib = IsotonicRegression(out_of_bounds='clip').fit(RF_large_p_calib[:,1], data["y_calib"])
        ebl_p_test = convert_prob_2D(iso_calib.predict(RF_large_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = ebl_p_test

    method = "RF_large_Beta"
    if method in calib_methods:
        beta_calib = BetaCalibration(parameters="abm").fit(RF_large_p_calib[:,1], data["y_calib"])
        ebl_p_test = convert_prob_2D(beta_calib.predict(RF_large_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = ebl_p_test

    ### RF_ens_k
    method = "RF_ens_k_Platt"
    if method in calib_methods:
        plat_calib = _SigmoidCalibration().fit(RF_ens_k_p_calib[:,1], data["y_calib"])
        ebl_p_test = convert_prob_2D(plat_calib.predict(RF_ens_k_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = ebl_p_test

    method = "Line"
    if method in calib_methods:
        lr_calib = LinearRegression().fit(rf_p_calib, data["y_calib"])
        y_pred_clipped = np.clip(lr_calib.predict(rf_p_test), 0, 1)
        lr_p_test = convert_prob_2D(y_pred_clipped)
        results_dict[f"{data_name}_{method}_prob"] = lr_p_test
        results_dict[f"{data_name}_{method}_fit"] = np.clip(lr_calib.predict(convert_prob_2D(tvec)), 0, 1)


    method = "Platt_d"
    if method in calib_methods:
        rf_d_platt = IR_RF(n_estimators=params["n_estimators"], random_state=params["seed"]).fit(data["x_train"], data["y_train"])
        rf_d_p_test = rf_d.predict_proba(data["x_test"])
        rf_d_p_calib = rf_d_platt.predict_proba(data["x_calib"])
        plat_calib = _SigmoidCalibration().fit(rf_d_p_calib[:,1], data["y_calib"])
        plat_p_test = convert_prob_2D(plat_calib.predict(rf_d_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = plat_p_test
        results_dict[f"{data_name}_{method}_fit"] = plat_calib.predict(tvec)

    # RF ranking + ISO
    method = "Rank"
    if method in calib_methods:
        x_calib_rank = RF.rank(data["x_calib"], class_to_rank=1, train_rank=True)
        x_test_rank = RF.rank_refrence(data["x_test"], class_to_rank=1)

        iso_rank = IsotonicRegression(out_of_bounds='clip').fit(x_calib_rank, data["y_calib"]) 
        rank_p_test = convert_prob_2D(iso_rank.predict(x_test_rank))
        results_dict[f"{data_name}_{method}_prob"] = rank_p_test
        # tvec_rank = RF.rank_refrence(data["x_test"], class_to_rank=1)
        # results_dict[data["name"] + "_Rank_fit"] = iso_rank.predict(tvec_rank)

    # perfect rank + ISO
    method = "prank"
    if method in calib_methods:
        iso_rank = IsotonicRegression(out_of_bounds='clip').fit(data["tp_calib"], data["y_calib"]) 
        rank_p_test = convert_prob_2D(iso_rank.predict(data["tp_test"]))
        results_dict[f"{data_name}_{method}_prob"] = rank_p_test

    # Venn calibrator
    method = "Venn"
    if method in calib_methods:
        ven_calib = Venn_calib().fit(rf_p_calib, data["y_calib"])
        ven_p_test = ven_calib.predict(rf_p_test)
        results_dict[f"{data_name}_{method}_prob"] = ven_p_test
        results_dict[f"{data_name}_{method}_fit"] = ven_calib.predict(convert_prob_2D(tvec))[:,1]

    # Elkan calibration
    method = "Elkan"
    if method in calib_methods:
        elkan_calib = Elkan_calib().fit(data["y_train"], data["y_calib"])
        elkan_p_test = elkan_calib.predict(rf_p_test[:,1])
        results_dict[f"{data_name}_{method}_prob"] = elkan_p_test
        results_dict[f"{data_name}_{method}_fit"] = elkan_calib.predict(tvec)[:,1]

    method = "true"
    if method in calib_methods:
        true_p_test = convert_prob_2D(bc.true_prob_ens(data["x_test"], data["y_test"], data["x_train"], data["y_train"], RF))
        results_dict[f"{data_name}_{method}_prob"] = true_p_test

    method = "bin"
    if method in calib_methods:
        rf_p_train = results_dict[data["name"] + "_RF_prob_train"]
        bc_bin = Bin_calib(params["ece_bins"]).fit(rf_p_train[:,1], data["y_train"], rf_p_calib[:,1], data["y_calib"])
        bin_p_test = convert_prob_2D(bc_bin.predict(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = bin_p_test
    
    method = "RF_boot"
    if method in calib_methods:
        rf_tree_test = RF.predict_proba(data["x_test"], return_tree_prob=True)
        bc_boot = Boot_calib(boot_count=params["boot_count"], bootstrap_size= params["boot_size"])
        bc_p_test = bc_boot.predict_boot(rf_tree_test)
        results_dict[f"{data_name}_{method}_prob"] = bc_p_test

    method = "RF_ens_boot"
    if method in calib_methods:
        bc_ensboot = Boot_calib(boot_count=params["boot_count"], bootstrap_size= params["boot_size"])
        bc_p_test = bc_ensboot.predict_ens_boot(data["x_test"], data["x_train"], data["y_train"], RF)
        results_dict[f"{data_name}_{method}_prob"] = bc_p_test

    method = "RF_ensbin"
    if method in calib_methods:
        rf_tree_test = RF.predict_proba(data["x_test"])
        bc_ensbin = Boot_calib(boot_count=params["boot_count"]).fit(data["x_train"], data["y_train"], RF)
        bc_p_test = convert_prob_2D(bc_ensbin.predict_ens2(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = bc_p_test

    method = "RF_CT"
    if method in calib_methods:
        rf_ct_test = RF.predict_proba(data["x_test"], classifier_tree=True)
        results_dict[f"{data_name}_{method}_prob"] = rf_ct_test

    method = "RF_Laplace"
    if method in calib_methods:
        rf_lap_test = RF.predict_proba(data["x_test"], laplace=1)
        results_dict[f"{data_name}_{method}_prob"] = rf_lap_test

    method = "RF_ens_p"
    if method in calib_methods:
        bc_p_test = bc.predict_ens(data["x_test"], data["x_train"], data["y_train"], RF, param_change=True)
        results_dict[f"{data_name}_{method}_prob"] = bc_p_test


    if "acc" in metrics:
        for method in calib_methods:
            results_dict[f"{data_name}_{method}_acc"] = accuracy_score(data["y_test"], np.argmax(results_dict[f"{data_name}_{method}_prob"],axis=1))

    if "auc" in metrics:
        for method in calib_methods:
            fpr, tpr, thresholds = roc_curve(data["y_test"], results_dict[data["name"] + "_" + method +"_prob"][:,1])
            results_dict[f"{data_name}_{method}_auc"] = auc(fpr, tpr)

    if "ece" in metrics:
        for method in calib_methods:
            pt, pp = calibration_curve(data["y_test"], results_dict[f"{data_name}_{method}_prob"][:,1], n_bins=params["ece_bins"], strategy=params["bin_strategy"])
            results_dict[f"{data_name}_{method}_ece"] = mean_squared_error(pt, pp)
            # results_dict[f"{data_name}_{method}_ece"] = confidance_ECE(results_dict[f"{data_name}_{method}_prob"], data["y_test"], bins=params["ece_bins"])

    if "brier" in metrics:
        for method in calib_methods:
            results_dict[f"{data_name}_{method}_brier"] = brier_score_loss(data["y_test"], results_dict[f"{data_name}_{method}_prob"][:,1])

    if "logloss" in metrics:
        for method in calib_methods:
            results_dict[f"{data_name}_{method}_logloss"] = log_loss(data["y_test"], results_dict[f"{data_name}_{method}_prob"][:,1])

    if "tce" in metrics:
        for method in calib_methods:
            results_dict[f"{data_name}_{method}_tce"] = mean_squared_error(data["tp_test"], results_dict[f"{data_name}_{method}_prob"][:,1])

    if "BS" in metrics:

        for method in calib_methods:
            # calculate C_test
            u = np.unique(results_dict[f"{data_name}_{method}_cstar"][:,1])
            c_test_calib_method = data["tp"].copy()
            for v in u:
                e_index = np.argwhere(results_dict[f"{data_name}_{method}_cstar"][:,1] == v)
                e_labels_mean = data["tp"][e_index].mean()
                c_test_calib_method[e_index] = e_labels_mean
            
            CL = mean_squared_error(c_test_calib_method, results_dict[f"{data_name}_{method}_cstar"][:,1])
            GL = mean_squared_error(data["tp"], c_test_calib_method)
            IL = mean_squared_error(data["tp"], data["y"])
            BS = mean_squared_error(data["y"], results_dict[f"{data_name}_{method}_cstar"][:,1])
            results_dict[f"{data_name}_{method}_CL"] = CL
            results_dict[f"{data_name}_{method}_GL"] = GL
            results_dict[f"{data_name}_{method}_IL"] = IL
            results_dict[f"{data_name}_{method}_BS"] = CL + GL + IL
            results_dict[f"{data_name}_{method}_BS2"] = BS

    return results_dict

def update_runs(ref_dict, new_dict):

    # calib results for every run for the same dataset is aggregated in this function 
    # (ex. acc of every run will be available as an array with the data_calibMethod_metric key)
    # the _prob and _decision keys in the returned result is not meaningfull

    if ref_dict == {}:
        res_dict = new_dict.copy()
        for k in new_dict.keys():
            res_dict[k] = [res_dict[k]]
        return res_dict
    
    res_dict = ref_dict.copy()
    # print("new_dict keys", len(new_dict.keys()))
    # print("res_dict keys", len(res_dict.keys()))
    # print("---------------------------------")
    for k in ref_dict.keys():
        res_dict[k] = ref_dict[k] + [new_dict[k]]

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

def find_nearest_index(arr, X):
    sorted_arr = np.sort(arr)
    absolute_diff = np.abs(sorted_arr - X)
    nearest_index = np.argmin(absolute_diff)
    return nearest_index

def predict_bin(prob_true, prob_pred, Y):
    calib_prob = []
    for y in Y:
        index = find_nearest_index(prob_pred, y)
        p = prob_true[index]
        calib_prob.append(p)
    calib_prob = np.array(calib_prob)
    return calib_prob

def plot_probs(exp_data_name, probs_runs, data_runs, params, ref_plot_name="RF", hist_plot=False, calib_plot=False, corrct_ece=True):

    calib_methods = params["calib_methods"]

    # concatinate all runs
    for method in calib_methods:

        all_run_probs = np.zeros(1)
        for prob in probs_runs[f"{exp_data_name}_{method}_prob"]:
            if len(all_run_probs) == 1:
                all_run_probs = prob
            else:
                all_run_probs = np.concatenate((all_run_probs, prob))

        all_run_probs_ref = np.zeros(1)
        for prob in probs_runs[f"{exp_data_name}_{ref_plot_name}_prob"]:
            if len(all_run_probs_ref) == 1:
                all_run_probs_ref = prob
            else:
                all_run_probs_ref = np.concatenate((all_run_probs_ref, prob))

        all_run_y = np.zeros(1)
        for data in data_runs:
            if len(all_run_y) == 1:
                all_run_y = data["y_test"]
            else:
                all_run_y = np.concatenate((all_run_y, data["y_test"]))

        if params["data_name"] == "synthetic":
            all_run_tp = np.zeros(1)
            for data in data_runs:
                if len(all_run_tp) == 1:
                    all_run_tp = data["tp_test"]
                else:
                    all_run_tp = np.concatenate((all_run_tp, data["tp_test"]))
        
        plt.plot([0, 1], [0, 1], linestyle='--')
        colors = ['black', 'red']
        colors_mean = ['orange', 'blue']
        if params["data_name"] == "synthetic":
            plt.scatter(all_run_tp, all_run_probs[:,1], marker='.', c=[colors[c] for c in all_run_y.astype(int)]) # Calibrated probs
            plt.scatter(all_run_tp, all_run_probs_ref[:,1], marker='.', c=[colors[c] for c in all_run_y.astype(int)], alpha=0.1) # faded RF probs
        else:
            prob_true, prob_pred = calibration_curve(all_run_y, all_run_probs[:,1], n_bins=params["ece_bins"], strategy=params["bin_strategy"])
            plt.scatter(prob_true, prob_pred, marker='.', c='darkblue') # Calibrated probs
            prob_true_ref, prob_pred_ref = calibration_curve(all_run_y, all_run_probs_ref[:,1], n_bins=params["ece_bins"], strategy=params["bin_strategy"])
            plt.scatter(prob_true_ref, prob_pred_ref, marker='.', alpha=0.2, c="gray") # Calibrated probs
            
        # plt.scatter(data["tp_train"], probs[f"{exp_data_name}_{ref_plot_name}_prob_train"][:,1], marker='.', c=[colors[c] for c in data["y_train"].astype(int)]) # RF train probs 


        # ################## Just to test vertical and horisantal averaging
        # bin_means, bin_edges, binnumber = binned_statistic(data["tp_test"], probs[f"{exp_data_name}_{ref_plot_name}_prob"][:,1], bins=100) # Mean of the calibrated probs
        # true_binded = (bin_edges[:-1] + bin_edges[1:])/2
        # plt.scatter(true_binded, bin_means, label='binned statistic of data')
        # v_tce = mean_squared_error((bin_edges[:-1] + bin_edges[1:])/2, bin_means)

        # y = predict_bin(true_binded, bin_means, probs[f"{exp_data_name}_{method}_prob"][:,1])
        # plt.scatter(data["tp_test"], y, marker='.', c=[colors[c] for c in all_run_y.astype(int)]) # Calibrated probs


        # bin_means, bin_edges, binnumber = binned_statistic(probs[f"{exp_data_name}_{method}_prob"][:,1], data["tp_test"], bins=100) # Horizantal Mean of the calibrated probs
        # plt.scatter((bin_edges[:-1] + bin_edges[1:])/2, bin_means, label='binned statistic of data')
        # h_tce = mean_squared_error((bin_edges[:-1] + bin_edges[1:])/2, bin_means)
        # ##################
        if params["data_name"] == "synthetic":
            calib_tce = mean_squared_error(all_run_tp, all_run_probs[:,1]) # calculate TCE to add to the calib method plot
            calib_tce_ref = mean_squared_error(all_run_tp, all_run_probs_ref[:,1]) # calculate TCE to add to the calib method plot
            tce_txt = f" (TCE {calib_tce:0.5f})"
            tce_ref = f" (TCE {calib_tce_ref:0.5f})"
        else:
            # print("exp_data_name", exp_data_name)
            # print("method", method)
            # print("prob shape", probs[f"{exp_data_name}_{method}_prob"].shape)
            calib_ece = mean_squared_error(prob_true, prob_pred) # confidance_ECE(all_run_probs, all_run_y, bins=params["ece_bins"])
            calib_ece_ref = mean_squared_error(prob_true_ref, prob_pred_ref) # confidance_ECE(all_run_probs_ref, all_run_y, bins=params["ece_bins"])
            if corrct_ece:
                probs_runs[f"{exp_data_name}_{method}_ece"] = [calib_ece]

            ece_txt = f" (ECE {calib_ece:0.5f})"
            ece_ref = f" (ECE {calib_ece_ref:0.5f})"
        # calib_tce = mean_squared_error(data["tp_test"], y) # calculate TCE to add to the calib method plot
        
        if (method == "ISO" or method == "CRF" or method == "Line" or method == "Platt" or method =="Beta" or method =="VA") and calib_plot:
            plt.plot(probs_runs[f"{exp_data_name}_{method}_fit"][0], tvec, c="blue")
        plt.xlabel(f"True probability")
        plt.ylabel("Predicted probability")

        # Add legend
        if params["data_name"] == "synthetic":
            red_patch = plt.plot([],[], marker='o', markersize=10, color='red', linestyle='')[0]
            black_patch = plt.plot([],[], marker='o', markersize=10, color='black', linestyle='')[0]
            calib_patch = plt.plot([],[], marker='_', markersize=15, color='blue', linestyle='')[0]
            if method == ref_plot_name:
                plt.legend((red_patch, black_patch, calib_patch), ('Class 1', 'Class 0', method + tce_txt), loc='upper left')
            else:
                plt.legend((red_patch, black_patch, calib_patch), ('Class 1', 'Class 0', method + tce_txt+ " RF"+ tce_ref), loc='upper left')
        else:
            orchid_patch = plt.plot([],[], marker='o', markersize=10, color='darkblue', linestyle='')[0]
            gray_patch = plt.plot([],[], marker='o', markersize=10, color='gray', linestyle='')[0]
            calib_patch = plt.plot([],[], marker='_', markersize=15, color='blue', linestyle='')[0]
            plt.legend((orchid_patch, gray_patch, calib_patch), (method + ece_txt, 'RF' + ece_ref, method + " fit"), loc='upper left')
        path = f"./results/{params['exp_name']}/{method}"
        if not os.path.exists(path):
            os.makedirs(path)

        plt.title(params['exp_name'] + f" {exp_data_name}")

        if params["data_name"] == "synthetic":
            plt.savefig(f"{path}/{exp_data_name}_{method}_s.pdf", format='pdf', transparent=True)
        else:
            plt.savefig(f"{path}/{exp_data_name}_{method}.pdf", format='pdf', transparent=True)
        plt.close()

        if hist_plot:
            plt.hist(all_run_probs[:,1], bins=50)
            plt.xlabel(f"probability output of {method}")
            plt.savefig(f"{path}/{method}_{exp_data_name}_hist.pdf", format='pdf', transparent=True)
            plt.close()

