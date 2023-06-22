
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

runs = 5
n_estimators=10

plot_bins = 10
test_size = 0.3


oob = False

plot = True
save_results = False

results_dict = {}

samples = 3000
features = 4
calib_methods = ["RF", "Platt" , "ISO", "Rank", "CRF", "VA", "Beta", "Elkan", "tlr"] # "prank", "Venn"
# calib_methods = ["RF", "tlr"]
metrics = ["acc", "auc", "brier", "ece"] # , "tce"

run_name = "Samples"


data_list = []
for exp in [30]: #[100, 200, 500, 1000, 2000, 5000, 10000, 50000]:#[2,5,10,20,40,80,100]:
    data = run_name + str(exp)
    data_list.append(data)

    for res_val in ["prob", "decision"]:
        _dict = {}
        for method in calib_methods:
            _dict[method] = []
        results_dict[data + "_" + res_val] = _dict


    for metric in metrics:
        _dict = {}
        for method in calib_methods:
            _dict[method] = []
        results_dict[data + "_" + metric] = _dict

    X, y, tp = dp.make_classification_gaussian_with_true_prob(exp, features, 0)

    for seed in range(runs):
        # seed = 5
        # print("seed ", seed)
        np.random.seed(seed)
        
        ### spliting data to train calib and test
        x_train_calib, x_test, y_train_calib, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)
        x_train, x_calib, y_train, y_calib = train_test_split(x_train_calib, y_train_calib, test_size=0.5, shuffle=True, random_state=seed) 
        _, _, tp_train_calib, tp_test = train_test_split(X, tp, test_size=test_size, shuffle=True, random_state=seed)
        _, _, tp_train, tp_calib = train_test_split(x_train_calib, tp_train_calib, test_size=0.5, shuffle=True, random_state=seed) 

        # print("tp_test ", tp_test.shape)
        # exit()

        ### training the IRRF
        irrf = IR_RF(n_estimators=n_estimators, oob_score=oob, random_state=seed)
        irrf.fit(x_train, y_train)

        ### calibration and ECE plot

        # random forest probs
        rf_p_calib = irrf.predict_proba(x_calib, laplace=1)
        rf_p_test = irrf.predict_proba(x_test, laplace=1)
        results_dict[data + "_prob"]["RF"] = rf_p_test
        results_dict[data + "_decision"]["RF"] = np.argmax(rf_p_test,axis=1)

        # Platt scaling on RF
        if "Platt" in calib_methods:
            plat_calib = _SigmoidCalibration().fit(rf_p_calib[:,1], y_calib)
            plat_p_test = convert_prob_2D(plat_calib.predict(rf_p_test[:,1]))
            results_dict[data + "_prob"]["Platt"] = plat_p_test
            results_dict[data + "_decision"]["Platt"] = np.argmax(plat_p_test,axis=1)

        # ISO calibration on RF
        if "ISO" in calib_methods:
            iso_calib = IsotonicRegression(out_of_bounds='clip').fit(rf_p_calib[:,1], y_calib)
            iso_p_test = convert_prob_2D(iso_calib.predict(rf_p_test[:,1]))
            results_dict[data + "_prob"]["ISO"] = iso_p_test
            results_dict[data + "_decision"]["ISO"] = np.argmax(iso_p_test,axis=1)


        # RF ranking + ISO
        if "Rank" in calib_methods:
            x_calib_rank = irrf.rank(x_calib, class_to_rank=1, train_rank=True)
            x_test_rank = irrf.rank_refrence(x_test, class_to_rank=1)

            iso_rank = IsotonicRegression(out_of_bounds='clip').fit(x_calib_rank, y_calib) 
            rank_p_test = convert_prob_2D(iso_rank.predict(x_test_rank))
            results_dict[data + "_prob"]["Rank"] = rank_p_test
            results_dict[data + "_decision"]["Rank"] = np.argmax(rank_p_test,axis=1)

        # perfect rank + ISO
        if "p_rank" in calib_methods:
            iso_rank = IsotonicRegression(out_of_bounds='clip').fit(tp_calib, y_calib) 
            rank_p_test = convert_prob_2D(iso_rank.predict(tp_test))
            results_dict[data + "_prob"]["p_rank"] = rank_p_test
            results_dict[data + "_decision"]["p_rank"] = np.argmax(rank_p_test,axis=1)

        # CRF calibrator
        if "CRF" in calib_methods:
            crf_calib = CRF_calib(learning_method="sig_brior").fit(rf_p_calib[:,1], y_calib)
            crf_p_test = crf_calib.predict(rf_p_test[:,1])
            results_dict[data + "_prob"]["CRF"] = crf_p_test
            results_dict[data + "_decision"]["CRF"] = np.argmax(crf_p_test,axis=1)

        # Venn calibrator
        if "Venn" in calib_methods:
            ven_calib = Venn_calib().fit(rf_p_calib, y_calib)
            ven_p_test = ven_calib.predict(rf_p_test)
            results_dict[data + "_prob"]["Venn"] = ven_p_test
            results_dict[data + "_decision"]["Venn"] = np.argmax(ven_p_test,axis=1)

        # Venn abers
        if "VA" in calib_methods:       
            VA = VA_calib().fit(rf_p_calib[:,1], y_calib)
            va_p_test = VA.predict(rf_p_test[:,1])
            results_dict[data + "_prob"]["VA"] = va_p_test
            results_dict[data + "_decision"]["VA"] = np.argmax(va_p_test,axis=1)

        # Beta calibration
        if "Beta" in calib_methods:
            beta_calib = BetaCalibration(parameters="abm").fit(rf_p_calib[:,1], y_calib)
            beta_p_test = convert_prob_2D(beta_calib.predict(rf_p_test[:,1]))
            results_dict[data + "_prob"]["Beta"] = beta_p_test
            results_dict[data + "_decision"]["Beta"] = np.argmax(beta_p_test,axis=1)

        # Elkan calibration
        if "Elkan" in calib_methods:
            elkan_calib = Elkan_calib().fit(y_train, y_calib)
            elkan_p_test = elkan_calib.predict(rf_p_test[:,1])
            results_dict[data + "_prob"]["Elkan"] = elkan_p_test
            results_dict[data + "_decision"]["Elkan"] = np.argmax(elkan_p_test,axis=1)

        # tree LR calib
        if "tlr" in calib_methods:
            tlr_calib = treeLR_calib().fit(irrf, x_train ,y_train, x_calib, y_calib)
            tlr_p_test = tlr_calib.predict(x_test)
            results_dict[data + "_prob"]["tlr"] = tlr_p_test
            results_dict[data + "_decision"]["tlr"] = np.argmax(tlr_p_test,axis=1)



        if "acc" in metrics:
            for method in calib_methods:
                results_dict[data + "_acc"][method].append(accuracy_score(y_test, results_dict[data + "_decision"][method]))

        if "auc" in metrics:
            for method in calib_methods:
                fpr, tpr, thresholds = roc_curve(y_test, results_dict[data + "_prob"][method][:,1])
                results_dict[data + "_auc"][method].append(auc(fpr, tpr))

        if "ece" in metrics:
            for method in calib_methods:
                results_dict[data + "_ece"][method].append(confidance_ECE(results_dict[data + "_prob"][method], y_test, bins=plot_bins))

        if "brier" in metrics:
            for method in calib_methods:
                results_dict[data + "_brier"][method].append(brier_score_loss(y_test, results_dict[data + "_prob"][method][:,1]))

        if "tce" in metrics:
            for method in calib_methods:
                results_dict[data + "_tce"][method].append(mean_squared_error(tp_test, results_dict[data + "_prob"][method][:,1]))
        
        if plot:

            for method in calib_methods:
                plt.plot([0, 1], [0, 1], linestyle='--')
                plt.scatter(tp_test, results_dict[data + "_prob"][method][:,1], marker='.', c=y_test, label=method)
                plt.xlabel("True probability")
                plt.ylabel("Predicted probability")
                plt.legend()
                plt.savefig(f"./results/Synthetic/plots/{seed}_{data}_{method}.png")
                plt.close()

    print(f"data {data} done")

# save results as txt
for metric in metrics:
    txt = "Data"
    for method in calib_methods:
        txt += "," + method

    for data in data_list:
        txt += "\n"+ data
        for method in calib_methods:
            txt += "," + str(np.array(results_dict[data+ "_" +metric][method]).mean())

    txt_data = StringIO(txt)
    df = pd.read_csv(txt_data, sep=",")
    df.set_index('Data', inplace=True)
    mean_res = df.mean()
    if metric == "ece" or metric == "brier" or metric == "tce":
        df_rank = df.rank(axis=1, ascending = True)
    else:
        df_rank = df.rank(axis=1, ascending = False)

    mean_rank = df_rank.mean()
    df.loc["Mean"] = mean_res
    df.loc["Rank"] = mean_rank
    if save_results:
        df.to_csv(f"./results/Synthetic/{data}_DataCalib_{metric}.csv",index=True)
    print("---------------------------------", metric)
    print(df)

print(results_dict)