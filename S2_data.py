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

runs = 10
n_estimators=100
samples = 10000

plot_bins = 10

calib = True
kendal = False

tree_rank_tau = []
rf_rank_tau = []
rf_prob_tau = []

fop_list, mpv_list = [], []
fop_iso_list, mpv_iso_list = [], []
fop_irrf_list, mpv_irrf_list = [], [] 
fop_true_list, mpv_true_list = [], []

ECE_rf_list = []
ECE_iso_list = []
ECE_irrf_list = []

seed = 0
for seed in range(runs):
    print("seed ", seed)
    np.random.seed(seed)

    ### Synthetic data generation

    X, tp = make_regression(samples) # make regression data
    y = np.where(tp>0, 1, 0) # create classification labels by setting a threshold

    ### spliting data to train calib and test
    test_size = 0.4
    x_train_calib, x_test, y_train_calib, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)
    x_train, x_calib, y_train, y_calib = train_test_split(x_train_calib, y_train_calib, test_size=0.5, shuffle=True, random_state=seed) 
    _, _, tp_train_calib, tp_test = train_test_split(X, tp, test_size=test_size, shuffle=True, random_state=seed)
    _, _, tp_train, tp_calib = train_test_split(x_train_calib, tp_train_calib, test_size=0.5, shuffle=True, random_state=seed) 

    ### training the IRRF
    irrf = IR_RF(n_estimators=n_estimators, random_state=seed)
    irrf.fit(x_train, y_train)

    # small test
    # x_test_prob = irrf.rank(x_test)
    # x_test_prob = irrf.predict_proba(x_test, laplace=1, return_tree_prob=True)
    # # print("x_test_prob ", x_test_prob[:,:,1])
    # exit()

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
        tau, p_value = kendalltau(tp_test_rank_sort, tp_test_rank0_sort)
        # tree_rank_tau.append(tau)
        # print("rank1 vs rank0 tau", tau)
        # exit()


        tau, p_value = kendalltau(tp_test_true_sort, tp_test_tree_sort)
        tree_rank_tau.append(tau)
        # print("true vs tree_rank tau", tau)


        tau, p_value = kendalltau(tp_test_true_sort, tp_test_rank_sort)
        rf_rank_tau.append(tau)
        # print("true vs rank tau", tau)

        tau, p_value = kendalltau(tp_test_true_sort, tp_test_prob_sort)
        rf_prob_tau.append(tau)
        # print("true vs RFprob tau", tau)

    if calib:
        ### calibration and ECE plot
        rf_p_calib = irrf.predict_proba(x_calib, laplace=0)
        rf_p_test = irrf.predict_proba(x_test, laplace=0)

        iso_rf = IsotonicRegression().fit(rf_p_calib[:,1], y_calib)
        rf_cp_test = iso_rf.predict(rf_p_test[:,1])


        x_calib_rank = irrf.rank(x_calib, class_to_rank=1, train_rank=True)
        # x_test_rank = irrf.rank(x_test, class_to_rank=1)
        x_test_rank = irrf.rank_refrence(x_test, class_to_rank=1)

        # x_calib_rank_norm = x_calib_rank / x_calib_rank.max()
        # x_test_rank_norm = x_test_rank / x_test_rank.max()

        # print("x_calib_rank", x_calib_rank.shape)
        # plt.plot(np.sort(x_calib_rank))

        # plt.hist(x_calib_rank, 100)
        # plt.show()
        # exit()

        iso_rank = IsotonicRegression().fit(x_calib_rank, y_calib) 
        irrf_cp_test = iso_rank.predict(x_test_rank)

        # tp_calib_norm = tp_calib / tp_calib.max()
        # tp_test_norm = tp_test / tp_test.max()
        iso_true = IsotonicRegression().fit(tp_calib, y_calib)
        true_cp_test = iso_true.predict(tp_test)


        # sig = _SigmoidCalibration().fit(x_calib_rank, y_calib)
        # irrf_cp_test_sig = iso.predict(x_test_rank)

        fop, mpv = calibration_curve(y_test, rf_p_test[:,1], n_bins=plot_bins)
        fop_iso, mpv_iso = calibration_curve(y_test, rf_cp_test, n_bins=plot_bins)
        fop_irrf, mpv_irrf = calibration_curve(y_test, irrf_cp_test, n_bins=plot_bins)
        fop_true, mpv_true = calibration_curve(y_test, true_cp_test, n_bins=plot_bins)

        ece_rf = confidance_ECE(rf_p_test, y_test)
        ece_iso = confidance_ECE(convert_prob_2D(rf_cp_test), y_test)
        ece_irrf = confidance_ECE(convert_prob_2D(irrf_cp_test), y_test)

        # print("ece_rf", ece_rf)
        # print("ece_iso", ece_iso)
        # print("ece_irrf", ece_irrf)

        ECE_rf_list.append(ece_rf)
        ECE_iso_list.append(ece_iso)
        ECE_irrf_list.append(ece_irrf)

        # print("text", fop.shape)
        # print("text", mpv.shape)
        # print("text", fop_iso.shape)
        # print("text", mpv_iso.shape)
        # print("text", fop_irrf.shape)
        # print("text", mpv_irrf.shape)
        # print("text", fop_true)
        # print("text", mpv_true)

        # fop_list[seed] = fop
        # mpv_list[seed] = mpv
        # fop_iso_list[seed] = fop_iso
        # mpv_iso_list[seed] = mpv_iso
        # fop_irrf_list[seed] = fop_irrf
        # mpv_irrf_list[seed] = mpv_irrf
        # fop_true_list[seed] = fop_true
        # mpv_true_list[seed] = mpv_true

        # fop_irrf_sig, mpv_irrf_sig = calibration_curve(y_test, irrf_cp_test_sig, n_bins=10)


if kendal:
    print("true vs tree_rank tau", np.array(tree_rank_tau).mean())
    print("true vs rank tau", np.array(rf_rank_tau).mean())
    print("true vs RFprob tau", np.array(rf_prob_tau).mean())

if calib:

    print("normal ece   ", np.array(ECE_rf_list).mean())
    print("iso ece      ", np.array(ECE_iso_list).mean())
    print("IRRF iso ece ", np.array(ECE_irrf_list).mean())

    exit()

    # # plot perfectly calibrated
    # plt.plot([0, 1], [0, 1], linestyle='--')
    # # plot model reliability
    # plt.plot(mpv_list.mean(axis=0), fop_list.mean(axis=0), marker='.', label="RF")
    # plt.plot(fop_iso_list.mean(axis=0), mpv_iso_list.mean(axis=0), marker='.', label="RF+iso")
    # plt.plot(fop_irrf_list.mean(axis=0), mpv_irrf_list.mean(axis=0), marker='.', label="RF+rank+ios", c="black")
    # # plt.plot(fop_true, mpv_true, marker='.', label="RF+true+ios", c="red")
    # # plt.plot(fop_irrf_sig, mpv_irrf_sig, marker='.', label="RF+rank+sig", c="blue")
    # plt.legend()
    # plt.show()
    # # plt.savefig("calib_plot.png")

    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot model reliability
    plt.plot(mpv, fop, marker='.', label="RF")
    plt.plot(fop_iso, mpv_iso, marker='.', label="RF+iso")
    plt.plot(fop_irrf, mpv_irrf, marker='.', label="RF+rank+ios", c="black")
    # plt.plot(fop_true, mpv_true, marker='.', label="RF+true+ios", c="red")
    # plt.plot(fop_irrf_sig, mpv_irrf_sig, marker='.', label="RF+rank+sig", c="blue")
    plt.legend()
    plt.show()
    # plt.savefig("calib_plot.png")
