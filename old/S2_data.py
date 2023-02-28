import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
from sklearn.datasets import make_regression
from scipy.stats import multivariate_normal
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split
from estimators.IR_RF_estimator import IR_RF
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from CalibrationM import confidance_ECE, convert_prob_2D
import Data.data_provider as dp

runs = 10
n_estimators= 100
samples = 10000
n_features = 100

plot_bins = 10

calib = True
kendal = False
plot_data = False
plot_true_rank = False

tree_rank_tau = []
rf_rank_tau = []
rf_prob_tau = []

y_test_list = []
rf_p_test_list = []
rf_cp_test_list = [] 
irrf_cp_test_list = []
true_cp_test_list = []

tp_list, pp_list = [], []
tp_iso_list, pp_iso_list = [], []
tp_irrf_list, pp_irrf_list = [], [] 
tp_true_list, pp_true_list = [], []

ECE_rf_list = []
ECE_iso_list = []
ECE_irrf_list = []
ECE_true_list = []

run_name = "G_ECE values 10k"

print("run_name", run_name)

for seed in range(runs):
    # seed = 3

    print("seed ", seed)
    np.random.seed(seed)

    ### Synthetic data generation
    X, y, tp = dp.make_classification_with_true_prob(samples, 2 , seed)
    # X, y, tp = dp.make_classification_with_true_prob2(samples, n_features,2, seed)
   
    # X, y, tp = dp.make_classification_with_true_prob(n_features,2,samples, seed)
    if plot_data:
        plt.scatter(X[:,0],tp, c=y)
        plt.xlabel("x")
        plt.ylabel("true rank")
        plt.show()
        exit()
    ### spliting data to train calib and test
    test_size = 0.4
    x_train_calib, x_test, y_train_calib, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)
    x_train, x_calib, y_train, y_calib = train_test_split(x_train_calib, y_train_calib, test_size=0.5, shuffle=True, random_state=seed) 
    _, _, tp_train_calib, tp_test = train_test_split(X, tp, test_size=test_size, shuffle=True, random_state=seed)
    _, _, tp_train, tp_calib = train_test_split(x_train_calib, tp_train_calib, test_size=0.5, shuffle=True, random_state=seed) 

    ### training the IRRF
    irrf = IR_RF(n_estimators=n_estimators, random_state=seed)
    irrf.fit(x_train, y_train)
    print("test score", irrf.score(x_test, y_test))


    if kendal:

        # ranking and getting prob for x_test
        x_test_rank = irrf.rank(x_test, class_to_rank=1)
        x_test_rank0 = irrf.rank(x_test, class_to_rank=0)
        x_test_prob = irrf.predict_proba(x_test, laplace=0)[:, 1]
        x_test_rank_tree = irrf.rank(x_test, class_to_rank=1, return_tree_rankings=True)[10]

        # sorting x_test based on ranking, prob, and true rankings
        rank_sort_index = np.argsort(x_test_rank, kind="stable")
        rank0_sort_index = np.argsort(x_test_rank0, kind="stable")
        prob_sort_index = np.argsort(x_test_prob, kind="stable")
        true_sort_index = np.argsort(tp_test, kind="stable")
        tree_sort_index = np.argsort(x_test_rank_tree, kind="stable")

        tp_test_rank_sort = tp_test[rank_sort_index]
        tp_test_rank0_sort = tp_test[rank0_sort_index]
        tp_test_prob_sort = tp_test[prob_sort_index]
        tp_test_true_sort = tp_test[true_sort_index]
        tp_test_tree_sort = tp_test[tree_sort_index]

        ### ranking performance of the IRRF compared to true rankings

        # class 1 vs class 0 in RF rank
        tau, p_value = kendalltau(tp_test_rank_sort, tp_test_rank0_sort)
        # tree_rank_tau.append(tau)
        # print("rank1 vs rank0 tau", tau)
        # exit()

        # True vs Tree rank
        tau, p_value = kendalltau(tp_test_true_sort, tp_test_tree_sort)
        tree_rank_tau.append(tau)
        # print("true vs tree_rank tau", tau)

        # True vs RF rank
        tau, p_value = kendalltau(tp_test_true_sort, tp_test_rank_sort)
        rf_rank_tau.append(tau)
        print("---------------------------------")
        print("true vs rank p_value", p_value)

        # True vs RF prob rank
        tau, p_value = kendalltau(tp_test_true_sort, tp_test_prob_sort)
        rf_prob_tau.append(tau)
        print("true vs RFprob p_value", p_value)

    if calib:
        ### calibration and ECE plot

        ## Normal random forest
        rf_p_calib = irrf.predict_proba(x_calib, laplace=0)
        rf_p_test = irrf.predict_proba(x_test, laplace=0)

        ## Random Forest + ISO
        iso_rf = IsotonicRegression(out_of_bounds='clip').fit(rf_p_calib[:,1], y_calib)
        rf_cp_test = iso_rf.predict(rf_p_test[:,1])

        ## Random Forest + Rrank + ISO
        x_calib_rank = irrf.rank(x_calib, class_to_rank=1, train_rank=True)
        x_test_rank = irrf.rank_refrence(x_test, class_to_rank=1)

        x_calib_rank_norm = x_calib_rank / x_calib_rank.max()
        x_test_rank_norm = x_test_rank / x_calib_rank.max() # normalize test data with max of the calib data to preserve the scale
    
        iso_rank = IsotonicRegression(out_of_bounds='clip').fit(x_calib_rank_norm, y_calib) 
        irrf_cp_test = iso_rank.predict(x_test_rank_norm)

        ## True Rank  + ISO
        tp_calib_norm = tp_calib / tp_calib.max()
        tp_test_norm = tp_test / tp_calib.max() # normalize test data with max of the calib data to preserve the scale

        iso_true = IsotonicRegression(out_of_bounds='clip').fit(tp_calib_norm, y_calib)
        true_cp_test = iso_true.predict(tp_test_norm)


        # sig = _SigmoidCalibration().fit(x_calib_rank, y_calib)
        # irrf_cp_test_sig = iso.predict(x_test_rank)

        y_test_list.append(y_test)

        rf_p_test_list.append(rf_p_test[:,1])
        rf_cp_test_list.append(rf_cp_test)
        irrf_cp_test_list.append(irrf_cp_test)
        true_cp_test_list.append(true_cp_test)
        
        # tp, pp = calibration_curve(y_test, rf_p_test[:,1], n_bins=plot_bins)
        # tp_iso, pp_iso = calibration_curve(y_test, rf_cp_test, n_bins=plot_bins)
        # tp_irrf, pp_irrf = calibration_curve(y_test, irrf_cp_test, n_bins=plot_bins)
        # tp_true, pp_true = calibration_curve(y_test, true_cp_test, n_bins=plot_bins)

        ece_rf = confidance_ECE(rf_p_test, y_test)
        ece_iso = confidance_ECE(convert_prob_2D(rf_cp_test), y_test)
        ece_irrf = confidance_ECE(convert_prob_2D(irrf_cp_test), y_test)
        ece_true = confidance_ECE(convert_prob_2D(true_cp_test), y_test)



        ECE_rf_list.append(ece_rf)
        ECE_iso_list.append(ece_iso)
        ECE_irrf_list.append(ece_irrf)
        ECE_true_list.append(ece_true)



if kendal:
    print("true vs tree_rank tau", np.array(tree_rank_tau).mean())
    print("true vs rank tau", np.array(rf_rank_tau).mean())
    print("true vs RFprob tau", np.array(rf_prob_tau).mean())

if calib:

    print("normal ece   ", np.array(ECE_rf_list).mean())
    print("iso ece      ", np.array(ECE_iso_list).mean())
    print("IRRF iso ece ", np.array(ECE_irrf_list).mean())
    print("True iso ece ", np.array(ECE_true_list).mean())


    
    y_test_all = np.array(y_test_list).reshape(-1)
    rf_p_test_all = np.array(rf_p_test_list).reshape(-1)
    rf_cp_test_all = np.array(rf_cp_test_list).reshape(-1)
    irrf_cp_test_all = np.array(irrf_cp_test_list).reshape(-1)
    true_cp_test_all = np.array(true_cp_test_list).reshape(-1)

    # plt.hist(rf_p_test_all, bins=100)
    # plt.show()
    # plt.hist(rf_cp_test_all, bins=100)
    # plt.show()
    # plt.hist(irrf_cp_test_all, bins=100)
    # plt.show()
    # plt.hist(true_cp_test_all, bins=100)
    # plt.show()

    # exit()



    print("---------------------------------")
    
    ece_rf = confidance_ECE(convert_prob_2D(rf_p_test_all), y_test_all)
    ece_iso = confidance_ECE(convert_prob_2D(rf_cp_test_all), y_test_all)
    ece_irrf = confidance_ECE(convert_prob_2D(irrf_cp_test_all), y_test_all)
    ece_true = confidance_ECE(convert_prob_2D(true_cp_test_all), y_test_all)

    print("normal ece   ", ece_rf)
    print("iso ece      ", ece_iso)
    print("IRRF iso ece ", ece_irrf)
    print("True iso ece ", ece_true)

    tp, pp = calibration_curve(y_test_all, rf_p_test_all, n_bins=plot_bins)
    tp_iso, pp_iso = calibration_curve(y_test_all, rf_cp_test_all, n_bins=plot_bins)
    tp_irrf, pp_irrf = calibration_curve(y_test_all, irrf_cp_test_all, n_bins=plot_bins)
    tp_true, pp_true = calibration_curve(y_test_all, true_cp_test_all, n_bins=plot_bins)

    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot model reliability
    plt.plot(tp, pp, marker='.', label="RF")
    plt.plot(tp_iso, pp_iso, marker='.', label="RF+iso")
    plt.plot(tp_irrf, pp_irrf, marker='.', label="RF+rank+ios", c="black")
    if plot_true_rank:
        plt.plot(tp_true, pp_true, marker='.', label="RF+true+ios", c="blue")
    # plt.plot(tp_irrf_sig, pp_irrf_sig, marker='.', label="RF+rank+sig", c="blue")
    plt.xlabel("True probability")
    plt.ylabel("Predicted probability")

    plt.legend()
    # plt.show()
    plt.savefig("calib_plot.png", transparent=True)
    # plt.savefig('calib_plot_v.eps', format='eps', transparent=True)
